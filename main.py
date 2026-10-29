from __future__ import annotations
"""
SatAgent – Closed-Loop Multi-Agent Satellite Image Restoration Pipeline
========================================================================

Three-Agent Mixture-of-Experts (MoE) Architecture:

  Agent 1 – Perception Engine   : VLM-driven degradation diagnosis + execution plan
  Agent 2 – Restoration Engine  : Dynamic routing (Local / Global path) + model execution
  Agent 3 – QA Engine           : NIQE / entropy rollback architecture

Entry point:
  python main.py --image <path> --profile <yaml> [--mock-vlm]
"""

import os
import sys
import json
import shutil
import warnings
import re
from pathlib import Path

# Suppress noisy requests/urllib3 version mismatch warnings in demo runs.
# Keep message-based filter to avoid importing `requests` just for the warning class.
warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 .* or chardet .* doesn't match a supported version.*",
)

# Fix relative python path executing module from outside
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.profile.manager import ProfileManager
from core.profile.selector import SatModelSelector
from core.perception.agent import PerceptionAgentOrchestrator, TASK_PHASE_SPECS
from core.perception.expert_iqa import ExpertIQAEvaluator
from core.perception.qa_engine import QAEngine
from executor import executor

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    warnings.warn("cv2/numpy unavailable – local mask crop+stitch path disabled")

# Color-fix post-processing (wavelet preferred for satellite spectral fidelity)
try:
    from utils.color_fix import wavelet_color_fix, cv2_to_pil, pil_to_cv2
    import cv2 as _cv2_cf
    _COLOR_FIX_AVAILABLE = True
except ImportError:
    _COLOR_FIX_AVAILABLE = False
    warnings.warn("color_fix not available – skipping post-processing color correction")


# ---------------------------------------------------------------------------
# Internal task-key mapping (VLM plan string → executor subtask key)
# ---------------------------------------------------------------------------
TASK_TO_SUBTASK: dict[str, str] = {
    # Phase 1
    "cloud removal":               "cloud_removal",
    "thick cloud removal":         "cloud_removal",
    "dehazing":                    "dehazing",
    "global dehazing":             "dehazing",
    "shadow correction":           "shadow_correction",
    # Phase 2
    "sensor denoising":            "denoising",
    "thermal/iso denoising":       "denoising",
    "thermal denoising":           "denoising",
    "iso denoising":               "denoising",
    "sensor deblurring":           "deblurring",
    "orbital motion deblurring":   "deblurring",
    "motion deblurring":           "deblurring",
    "destriping":                  "destriping",
    "destriping / dead-pixel inpainting": "destriping",
    "destriping / dead-pixel":     "destriping",
    "compression artifact removal": "artifact_removal",
    "artifact removal":            "artifact_removal",
    # Phase 3
    "pan-sharpening":              "pan_sharpen",
    "fusion":                      "pan_sharpen",
    "multi-temporal fusion":       "temporal_fusion",
    # Phase 4
    "super-resolution":            "super_resolution",
    "super-resolution (2x)":       "super_resolution",
    "super-resolution (4x)":       "super_resolution",
    "upscaling to 4k":             "super_resolution",
}

# Backward compatibility for legacy/aliased model ids.
MODEL_ALIASES: dict[str, str] = {
    "fbcnn": "fbcnn_blind",
    "evssm": "EVSSM",
}

# Canonical full-stack execution order matching the SatAgent paper spec.
FULL_MOE_TASK_ORDER: list[str] = [
    "cloud removal",                  # 1A
    "dehazing",                       # 1B
    "shadow correction",              # 1C
    "sensor denoising",               # 2A
    "sensor deblurring",              # 2B
    "destriping",                     # 2C
    "compression artifact removal",   # 2D
    "pan-sharpening",                 # 3A
    "multi-temporal fusion",          # 3B (profile-gated)
    "super-resolution",               # 4
]


def _build_full_moe_plan(profile_manager: ProfileManager) -> list[dict]:
    """
    Deterministic full MoE plan generator.
    Useful for paper/demo runs where all phases must be executed in order,
    independent of VLM diagnosis confidence.
    """
    mode = profile_manager.config.restoration.preference
    allow_temporal = profile_manager.config.restoration.enable_temporal_fusion
    allowed = set(profile_manager.config.restoration.allowed_agenda)

    steps: list[dict] = []
    for task in FULL_MOE_TASK_ORDER:
        if task == "multi-temporal fusion" and not allow_temporal:
            continue
        if task not in allowed and not (
            task.startswith("super-resolution") and "super-resolution" in allowed
        ):
            continue

        task_spec = TASK_PHASE_SPECS.get(task, {})
        routing = str(task_spec.get("routing_path", "global"))
        primary_model = SatModelSelector.get_model(task, mode)
        fallback_model = SatModelSelector.get_fallback_model(task, primary_model, mode) or ""
        candidates = SatModelSelector.get_model_candidates(task)
        mask_tool = SatModelSelector.get_mask_tool(task) if routing == "local" else None

        steps.append(
            {
                "step_id": len(steps) + 1,
                "phase_id": str(task_spec.get("phase_id", "unknown")),
                "phase_name": str(task_spec.get("phase_name", "Unclassified")),
                "stage_task": task,
                "routing_path": routing,
                "assigned_model": primary_model,
                "fallback_model": fallback_model,
                "model_candidates": candidates,
                "benchmark_datasets": list(task_spec.get("datasets", [])),
                "masking": {
                    "required": routing == "local",
                    "mask_tool": mask_tool or "",
                    "stitching": "poisson" if routing == "local" else "",
                },
                "execution_mode": mode,
            }
        )

    return steps


def _tool_names_for_subtask(subtask: str) -> list[str]:
    return [tool.tool_name for tool in executor.toolbox_router.get(subtask, [])]


def _resolve_tool_name(
    subtask: str,
    requested_model: str | None,
    fallback_to_primary: bool = True,
) -> str | None:
    tools = _tool_names_for_subtask(subtask)
    if not tools:
        return None
    if requested_model:
        req = requested_model.strip()
        req_lower = req.lower()
        # Exact
        if req in tools:
            return req
        # Case-insensitive
        for tool in tools:
            if tool.lower() == req_lower:
                return tool
        # Aliases
        aliased = MODEL_ALIASES.get(req_lower)
        if aliased and aliased in tools:
            return aliased
    return tools[0] if fallback_to_primary else None


def _maybe_force_model(subtask_key: str, current_model: str) -> str:
    """
    Allow one-off model overrides via environment variables.
    Example: SATA_FORCE_MODEL_SUPER_RESOLUTION=hat_psnr
    """
    env_key = f"SATA_FORCE_MODEL_{subtask_key.upper()}"
    forced = os.environ.get(env_key, "").strip()
    if not forced:
        return current_model
    tools = _tool_names_for_subtask(subtask_key)
    for tool in tools:
        if tool.lower() == forced.lower():
            return tool
    print(f"    [!] Forced model '{forced}' not found for subtask '{subtask_key}', using {current_model}.")
    return current_model


def _lookup_tool(subtask_key: str, tool_name: str):
    for tool in executor.toolbox_router.get(subtask_key, []):
        if tool.tool_name == tool_name:
            return tool
    return None


def _detect_alias_backend(tool) -> str | None:
    sp = getattr(tool, "script_path", None)
    if not sp or not Path(sp).exists():
        return None
    try:
        txt = Path(sp).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(r"ALIAS_BACKEND\s*=\s*['\"]([^'\"]+)['\"]", txt)
    if m:
        return m.group(1)
    return None


def _extract_ckpt_paths_from_opts(opts: list[str]) -> list[Path]:
    keys = {
        "--ckpt", "--ckpts", "--model_path", "--weight", "--weights",
        "--ckpt_path", "--pretrained_path", "--ram_ft_path", "--ram_path",
    }
    out: list[Path] = []
    i = 0
    while i < len(opts):
        tok = str(opts[i])
        key = tok.split("=", 1)[0]
        if key in keys:
            if "=" in tok:
                val = tok.split("=", 1)[1]
            elif i + 1 < len(opts):
                val = str(opts[i + 1])
                i += 1
            else:
                val = ""
            if val and str(val).strip().lower() not in {"none", "null", "na", "n/a"}:
                out.append(Path(val).expanduser())
        i += 1
    return out


def _run_plan_preflight(
    execution_plan: list[dict],
    image_path: str,
    output_dir: str | None,
    tool_run_gpu_id: int | None,
    fail_on_alias: bool = False,
    smoke: bool = False,
) -> tuple[bool, list[dict]]:
    """
    Preflight selected models before long pipeline execution.
    - Validates wrapper/script existence
    - Flags alias wrappers
    - Verifies checkpoint paths referenced by CLI options exist
    - Optional tiny-image smoke execution
    """
    report: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # Tiny preflight image.
    if not _CV2_AVAILABLE:
        return False, [{"level": "error", "reason": "cv2 unavailable for preflight image creation"}]
    src = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if src is None:
        return False, [{"level": "error", "reason": f"input unreadable: {image_path}"}]
    tiny = cv2.resize(src, (128, 128), interpolation=cv2.INTER_AREA)
    base_dir = Path(output_dir).resolve() if output_dir else Path(image_path).resolve().parent
    pre_dir = base_dir / "_preflight"
    pre_dir.mkdir(parents=True, exist_ok=True)
    tiny_in = pre_dir / "input_128.png"
    cv2.imwrite(str(tiny_in), tiny)

    old_timeout = os.environ.get("SATA_TOOL_CMD_TIMEOUT_SEC")
    if smoke:
        os.environ["SATA_TOOL_CMD_TIMEOUT_SEC"] = os.environ.get("SATA_PREFLIGHT_SMOKE_TIMEOUT_SEC", "45")
    try:
        for step in execution_plan:
            task = str(step.get("stage_task", "")).lower().strip()
            subtask_key = TASK_TO_SUBTASK.get(task)
            if not subtask_key:
                continue
            candidates = [
                str(step.get("assigned_model", "")).strip(),
                str(step.get("fallback_model", "")).strip(),
            ]
            for req_model in candidates:
                if not req_model:
                    continue
                resolved = _resolve_tool_name(subtask_key, req_model, fallback_to_primary=False)
                if not resolved:
                    continue
                key = (subtask_key, resolved)
                if key in seen:
                    continue
                seen.add(key)

                rec = {
                    "subtask": subtask_key,
                    "model": resolved,
                    "requested": req_model,
                    "ok": True,
                    "issues": [],
                }
                tool = _lookup_tool(subtask_key, resolved)
                if tool is None:
                    rec["ok"] = False
                    rec["issues"].append("tool_not_registered")
                    report.append(rec)
                    continue

                sp = getattr(tool, "script_path", None)
                if sp is None or not Path(sp).exists():
                    rec["ok"] = False
                    rec["issues"].append("script_missing")
                    report.append(rec)
                    continue

                alias = _detect_alias_backend(tool)
                is_alias = bool(alias)
                if alias:
                    rec["alias_backend"] = alias
                    rec["issues"].append(f"alias_wrapper:{alias}")
                    if fail_on_alias:
                        rec["ok"] = False

                # Build ckpt references by evaluating tool options on tiny paths.
                tiny_out = pre_dir / f"{subtask_key}_{resolved}_out.png"
                setattr(tool, "input_path", tiny_in)
                setattr(tool, "output_path", tiny_out)
                setattr(tool, "run_gpu_id", tool_run_gpu_id)
                try:
                    opts = [str(x) for x in tool._get_cmd_opts()]
                except Exception as e:
                    rec["ok"] = False
                    rec["issues"].append(f"opts_error:{e}")
                    report.append(rec)
                    continue

                ckpts = _extract_ckpt_paths_from_opts(opts)
                for ck in ckpts:
                    ck_abs = ck if ck.is_absolute() else (Path.cwd() / ck)
                    if not ck_abs.exists():
                        # For alias wrappers, checkpoint options may be legacy
                        # pass-through args not consumed by the true backend.
                        if not is_alias:
                            rec["ok"] = False
                        rec["issues"].append(f"missing_ckpt:{ck_abs}")

                if smoke and rec["ok"]:
                    try:
                        executor.invoke_a_tool(
                            subtask_name=subtask_key,
                            tool_name=resolved,
                            input_path=tiny_in,
                            output_path=tiny_out,
                            run_gpu_id=tool_run_gpu_id,
                            silent=True,
                        )
                        if not tiny_out.exists():
                            rec["ok"] = False
                            rec["issues"].append("smoke_no_output")
                    except Exception as e:
                        rec["ok"] = False
                        rec["issues"].append(f"smoke_error:{e}")
                report.append(rec)
    finally:
        if old_timeout is None:
            os.environ.pop("SATA_TOOL_CMD_TIMEOUT_SEC", None)
        else:
            os.environ["SATA_TOOL_CMD_TIMEOUT_SEC"] = old_timeout

    ok = all(r.get("ok", False) for r in report)
    try:
        (pre_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        pass
    return ok, report


def _read_binary_mask(mask_path: Path, target_h: int, target_w: int) -> "np.ndarray | None":
    if not _CV2_AVAILABLE:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _compute_bbox(mask_bin: "np.ndarray", pad: int = 24) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    h, w = mask_bin.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)
    y2 = min(h, int(ys.max()) + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _poisson_stitch(
    base_bgr: "np.ndarray",
    restored_roi_bgr: "np.ndarray",
    roi_mask_bin: "np.ndarray",
    bbox: tuple[int, int, int, int],
) -> "np.ndarray":
    x1, y1, x2, y2 = bbox
    patch_h = y2 - y1
    patch_w = x2 - x1

    if restored_roi_bgr.shape[:2] != (patch_h, patch_w):
        restored_roi_bgr = cv2.resize(
            restored_roi_bgr, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC
        )
    if roi_mask_bin.shape[:2] != (patch_h, patch_w):
        roi_mask_bin = cv2.resize(
            roi_mask_bin, (patch_w, patch_h), interpolation=cv2.INTER_NEAREST
        )
    _, roi_mask_bin = cv2.threshold(roi_mask_bin, 127, 255, cv2.THRESH_BINARY)

    # Build full-frame src + mask for seamlessClone.
    src = np.zeros_like(base_bgr, dtype=np.uint8)
    src[y1:y2, x1:x2] = restored_roi_bgr
    full_mask = np.zeros(base_bgr.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = roi_mask_bin

    nonzero = int(np.count_nonzero(full_mask))
    if nonzero < 16:
        # Mask too small/noisy, fallback to simple alpha insertion.
        out = base_bgr.copy()
        alpha = (full_mask.astype(np.float32) / 255.0)[..., None]
        out = (out * (1.0 - alpha) + src * alpha).astype(np.uint8)
        return out

    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    try:
        return cv2.seamlessClone(src, base_bgr, full_mask, center, cv2.NORMAL_CLONE)
    except Exception:
        out = base_bgr.copy()
        alpha = (full_mask.astype(np.float32) / 255.0)[..., None]
        out = (out * (1.0 - alpha) + src * alpha).astype(np.uint8)
        return out


def _ensure_4k_output(
    input_image_path: Path,
    output_image_path: Path,
    target_w: int = 3840,
    target_h: int = 2160,
) -> bool:
    """
    Force final deliverable to UHD 4K canvas (3840x2160).
    Keeps aspect ratio and pads to avoid geometric distortion.
    """
    if not _CV2_AVAILABLE:
        return False

    img = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    h, w = img.shape[:2]
    if (w, h) == (target_w, target_h):
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        if input_image_path.resolve() != output_image_path.resolve():
            shutil.copy(str(input_image_path), str(output_image_path))
        return True

    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_image_path), canvas))


def _run_local_path_model(
    subtask_key: str,
    tool_name: str,
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    run_gpu_id: int | None = None,
) -> bool:
    """
    Local-path routing:
      1) Use mask to crop ROI.
      2) Restore ROI only.
      3) Stitch with Poisson blending.
    """
    if not _CV2_AVAILABLE:
        return False

    base = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if base is None:
        return False

    h, w = base.shape[:2]
    mask_bin = _read_binary_mask(mask_path, h, w)
    if mask_bin is None:
        return False

    bbox = _compute_bbox(mask_bin)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox

    roi_in = base[y1:y2, x1:x2]
    roi_mask = mask_bin[y1:y2, x1:x2]
    roi_input_path = output_path.with_name(f"{output_path.stem}_roi_input.png")
    roi_output_path = output_path.with_name(f"{output_path.stem}_roi_restored.png")

    cv2.imwrite(str(roi_input_path), roi_in)
    executor.invoke_a_tool(
        subtask_name=subtask_key,
        tool_name=tool_name,
        input_path=roi_input_path,
        output_path=roi_output_path,
        run_gpu_id=run_gpu_id,
    )

    roi_restored = cv2.imread(str(roi_output_path), cv2.IMREAD_COLOR)
    if roi_restored is None:
        return False

    blended = _poisson_stitch(base, roi_restored, roi_mask, bbox)
    cv2.imwrite(str(output_path), blended)
    return output_path.exists()


def run_pipeline(
    image_path: str,
    profile_path: str,
    mock_vlm: bool = False,
    force_full_moe: bool = False,
    output_dir: str | None = None,
    tool_run_gpu_id: int | None = None,
    strict_real_execution: bool = False,
    require_true_models: bool = False,
    require_preflight: bool = False,
    preflight_smoke: bool = False,
    preflight_alias_strict: bool = False,
    skip_preflight: bool = False,
) -> str:
    """
    Full three-agent pipeline execution.

    Args:
        image_path:  Path to the raw input satellite image.
        profile_path: Path to the YAML configuration profile.
        mock_vlm:    If True, bypass Agent 1 and inject a test plan.
        force_full_moe: If True, bypass Agent 1 diagnosis and force the full
                     paper-spec MoE schedule (Phase 1A→4).
        output_dir:  Optional directory to write the final output. If None,
                     outputs are written next to the input image.
        tool_run_gpu_id: Optional GPU index to pin tool execution.
        strict_real_execution: Disable all proxy/continuity fallbacks so only
                     real model execution paths are allowed.
        require_true_models: Block placeholder template scripts and allow only
                     true model wrappers.
        require_preflight: Run model preflight checks before restoration.
        preflight_smoke: Execute tiny-image smoke invocation per selected model.
        preflight_alias_strict: Fail preflight when alias wrappers are selected.
        skip_preflight: Skip preflight even if strict mode auto-preflight is enabled.

    Returns:
        Path to the final restored image.
    """
    print("\n" + "=" * 72)
    print("  🛰  SatAgent – MoE Satellite Restoration Pipeline")
    print("=" * 72)
    print(f"  Input  : {image_path}")
    print(f"  Profile: {profile_path}")
    print(f"  MockVLM: {mock_vlm}")
    print(f"  FullMoE: {force_full_moe}")
    print(f"  ToolGPU: {tool_run_gpu_id if tool_run_gpu_id is not None else 'auto'}")
    print(f"  Strict : {strict_real_execution}")
    print(f"  TrueML : {require_true_models}")
    print("=" * 72 + "\n")

    if strict_real_execution:
        os.environ["SATA_ENABLE_BASELINE_PROXY"] = "0"
        os.environ["SATA_TOOL_GRACEFUL_FALLBACK"] = "0"
        os.environ["SATA_ENFORCE_REAL_BACKEND"] = "1"
        os.environ["SATA_FAIL_ON_TIMEOUT"] = "1"
        os.environ["SATA_DISABLE_OUTPUT_RECOVERY_ON_FAILURE"] = "1"
    else:
        os.environ["SATA_ENFORCE_REAL_BACKEND"] = "0"
    if require_true_models:
        os.environ["SATA_REQUIRE_TRUE_MODELS"] = "1"
    auto_preflight = strict_real_execution and os.environ.get("SATA_AUTO_PREFLIGHT", "1") == "1"
    do_preflight = (require_preflight or auto_preflight) and not skip_preflight
    if "SATA_TOOL_CMD_TIMEOUT_SEC" not in os.environ:
        # Guardrail against hanging third-party wrappers.
        os.environ["SATA_TOOL_CMD_TIMEOUT_SEC"] = "180"

    # -----------------------------------------------------------------------
    # 0. Load profile
    # -----------------------------------------------------------------------
    try:
        profile_manager = ProfileManager(profile_path)
    except Exception as e:
        print(f"[!] Profile load failed: {e}")
        return image_path

    qa_cfg = profile_manager.config.qa

    # -----------------------------------------------------------------------
    # AGENT 1 – Perception Engine
    # -----------------------------------------------------------------------
    if mock_vlm or force_full_moe:
        if mock_vlm:
            print("[🧠] AGENT 1: Bypassing VLM (Mock Mode)")
        else:
            print("[🧠] AGENT 1: Forced Full-MoE Mode (paper-spec schedule)")
        execution_plan = _build_full_moe_plan(profile_manager)
        iqa_baseline = ExpertIQAEvaluator.extract_metrics(image_path)
    else:
        print("[🧠] AGENT 1: Perception Engine – VLM Diagnosis & Planning")
        try:
            perceptor = PerceptionAgentOrchestrator(profile_manager)
            # Injecting model health data for the VLM to weigh in its planning
            health_data_path = Path("sata/configs/model_health.json")
            if health_data_path.exists():
                perceptor.load_health_data(str(health_data_path))
            plan_dict = perceptor.execute_pipeline(image_path)
        except Exception as e:
            print(f"  [!] Perception Engine crashed: {e}")
            return image_path

        execution_plan = plan_dict.get("execution_plan", [])
        iqa_baseline = plan_dict.get("expert_iqa_telemetry", {})

        if not execution_plan:
            print("  [✅] No degradations matched the allowed agenda. No restoration needed.")
            return image_path

        print(f"\n  Identified degradations: {plan_dict.get('identified_degradations', [])}")
        print(f"  Execution plan ({len(execution_plan)} steps):")
        for i, step in enumerate(execution_plan, 1):
            stage_task = str(step.get("stage_task", "unknown"))
            mode = str(step.get("execution_mode", "Fidelity"))
            model = str(step.get("assigned_model", "unknown"))
            fallback = str(step.get("fallback_model", "") or "none")
            print(f"    {i}. {stage_task:30s}  [{mode}]  → {model}  (fallback: {fallback})")

    if do_preflight:
        print("\n[🧪] PREFLIGHT: validating selected model paths...")
        ok_pf, pf_report = _run_plan_preflight(
            execution_plan=execution_plan,
            image_path=image_path,
            output_dir=output_dir,
            tool_run_gpu_id=tool_run_gpu_id,
            fail_on_alias=preflight_alias_strict,
            smoke=preflight_smoke,
        )
        bad = [r for r in pf_report if not r.get("ok", False)]
        print(f"  Checked models : {len(pf_report)}")
        print(f"  Failures       : {len(bad)}")
        if bad:
            for rec in bad[:10]:
                issues = "; ".join(rec.get("issues", []))
                print(f"    - {rec.get('subtask')}::{rec.get('model')} -> {issues}")
        if not ok_pf:
            raise RuntimeError("Preflight failed. Resolve issues before full pipeline execution.")

    # -----------------------------------------------------------------------
    # AGENT 3 – QA Engine initialisation (runs alongside Agent 2)
    # -----------------------------------------------------------------------
    qa = QAEngine(
        niqe_rollback_threshold=qa_cfg.niqe_rollback_threshold,
        entropy_rollback_threshold=qa_cfg.entropy_rollback_threshold,
        super_resolution_entropy_allowance=qa_cfg.super_resolution_entropy_allowance,
        attempt_fallback=qa_cfg.attempt_fallback_on_rollback,
        max_retries=qa_cfg.max_rollback_retries,
        force_approve_all_steps=qa_cfg.force_approve_all_steps,
    )

    # Seed the QA cache with the clean baseline state
    try:
        baseline_qa_metrics = qa.extract_metrics(image_path)
        qa.cache_state(
            image_path=Path(image_path).resolve(),
            task="[baseline]",
            model_used="identity",
            metrics=baseline_qa_metrics,
            step_idx=0,
        )
    except Exception:
        baseline_qa_metrics = iqa_baseline  # fallback to Agent 1 metrics

    # -----------------------------------------------------------------------
    # AGENT 2 – Restoration Engine (Dynamic Routing + Execution)
    # -----------------------------------------------------------------------
    print("\n[⚙️]  AGENT 2: Restoration Engine – Dynamic Routing & Execution")
    print("-" * 72)

    current_image = os.path.abspath(image_path)
    base_dir = (
        Path(output_dir).resolve() if output_dir
        else Path(current_image).parent.resolve()
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    runtime_plan = []
    for idx, step in enumerate(execution_plan, start=1):
        runtime_plan.append(
            {
                **step,
                "step_index": idx,
                "status": "pending",
                "qa_reason": "",
            }
        )

    step_idx = 0
    for plan_step in execution_plan:
        step_idx += 1
        runtime_step = runtime_plan[step_idx - 1]
        vlm_task = str(plan_step.get("stage_task", "")).lower().strip()
        primary_model_requested = str(plan_step.get("assigned_model", "")).strip()
        fallback_model_requested = str(plan_step.get("fallback_model", "")).strip() or None
        mode = str(plan_step.get("execution_mode", "Fidelity"))

        subtask_key = TASK_TO_SUBTASK.get(vlm_task)
        if not subtask_key:
            print(f"\n  [Step {step_idx}] ⚠  Unknown task '{vlm_task}' – skipping.")
            runtime_step["status"] = "skipped_unknown_task"
            continue

        primary_model = _resolve_tool_name(subtask_key, primary_model_requested, fallback_to_primary=True)
        fallback_model = _resolve_tool_name(subtask_key, fallback_model_requested, fallback_to_primary=False)
        if primary_model:
            forced = _maybe_force_model(subtask_key, primary_model)
            if forced != primary_model:
                primary_model = forced
                runtime_step["forced_model"] = primary_model
        if not primary_model:
            print(f"\n  [Step {step_idx}] ⚠  No registered models for subtask '{subtask_key}' – skipping.")
            runtime_step["status"] = "skipped_no_registered_tool"
            continue

        # Determine routing path (plan-declared route must match executor physical route).
        planned_routing = str(plan_step.get("routing_path", "")).lower().strip()
        physical_routing = executor.get_routing_mode(subtask_key)
        routing = planned_routing if planned_routing in {"local", "global"} else physical_routing
        if routing != physical_routing:
            routing = physical_routing
            runtime_step["routing_override"] = (
                f"Planner requested '{planned_routing}' but "
                f"subtask '{subtask_key}' enforces '{physical_routing}'."
            )
        runtime_step["routing"] = routing
        runtime_step["resolved_model"] = primary_model
        runtime_step["resolved_fallback"] = fallback_model or ""
        phase_id = str(plan_step.get("phase_id", "")).strip()
        phase_name = str(plan_step.get("phase_name", "")).strip()
        phase_hdr = f"{phase_id} {phase_name}".strip()
        print(f"\n  [Step {step_idx}/{len(execution_plan)}] ─── {vlm_task.upper()} ───")
        if phase_hdr:
            print(f"    Phase   : {phase_hdr}")
        print(f"    Routing : {'LOCAL  (mask-dep.)' if routing == 'local' else 'GLOBAL (full-frame)'}")
        print(f"    Mode    : {mode}")
        print(
            f"    Model   : {primary_model} "
            f"(requested: {primary_model_requested or 'none'}, fallback: {fallback_model or 'none'})"
        )

        # -- Capture baseline telemetry for this step --
        try:
            step_baseline = qa.extract_metrics(current_image)
        except Exception as e:
            print(f"    [!] Baseline IQA failed: {e}. Skipping step.")
            runtime_step["status"] = "skipped_baseline_iqa_failed"
            continue

        # ----------------------------------------------------------------
        # Phase 0 – ROI mask generation (Local Path only)
        # ----------------------------------------------------------------
        mask_path = None
        if routing == "local" and profile_manager.config.restoration.enable_roi_masking:
            mask_tool = SatModelSelector.get_mask_tool(vlm_task) or "rs_sam"
            mask_output = base_dir / f"step{step_idx}_{subtask_key}_mask.png"
            print(f"    Phase 0 : Generating ROI mask via '{mask_tool}'…")
            mask_path = executor.generate_roi_mask(
                input_path=Path(current_image),
                mask_output_path=mask_output,
                mask_tool=mask_tool,
                silent=False,
                run_gpu_id=tool_run_gpu_id,
            )
            if mask_path and mask_path.exists():
                print(f"    Phase 0 : Mask ready → {mask_path}")
            else:
                print("    Phase 0 : ⚠ Mask generation failed – continuing without mask.")
                mask_path = None

        # ----------------------------------------------------------------
        # Agent 2 – Primary model execution
        # ----------------------------------------------------------------
        temp_output = base_dir / f"step{step_idx}_{subtask_key}_out.png"

        def _run_model(model_name: str, in_path: str, out_path: Path) -> bool:
            """Inner helper – run one model, return success flag."""
            try:
                in_path_obj = Path(in_path)
                if routing == "local" and mask_path and mask_path.exists():
                    local_ok = _run_local_path_model(
                        subtask_key=subtask_key,
                        tool_name=model_name,
                        input_path=in_path_obj,
                        output_path=out_path,
                        mask_path=mask_path,
                        run_gpu_id=tool_run_gpu_id,
                    )
                    if local_ok:
                        return True
                    print("      [!] Local route failed, retrying as global full-frame.")

                executor.invoke_a_tool(
                    subtask_name=subtask_key,
                    tool_name=model_name,
                    input_path=in_path_obj,
                    output_path=out_path,
                    run_gpu_id=tool_run_gpu_id,
                )
                return out_path.is_file()
            except Exception as exc:
                print(f"      [!] {model_name} execution failed: {exc}")
                return False

        print(f"    Executing primary model '{primary_model}'…")
        success = _run_model(primary_model, current_image, temp_output)

        if not success:
            print(f"    [!] Primary model produced no output.")
            if fallback_model:
                fallback_out = base_dir / f"step{step_idx}_{subtask_key}_fallback.png"
                print(f"    Trying fallback model '{fallback_model}'…")
                success = _run_model(fallback_model, current_image, fallback_out)
                if success:
                    temp_output = fallback_out
                    primary_model = fallback_model
                    fallback_model = None
                    runtime_step["status"] = "rewritten_primary_to_fallback_after_execution_failure"
                    runtime_step["resolved_model"] = primary_model
                    runtime_step["resolved_fallback"] = ""
                    runtime_step["rewrite_reason"] = "Primary model execution failed."
                    plan_step["assigned_model"] = primary_model
                    plan_step["fallback_model"] = ""
            if not success:
                print("    [❌] All models failed. Rolling back and skipping step.")
                runtime_step["status"] = "skipped_all_models_failed"
                continue

        # ----------------------------------------------------------------
        # Agent 3 – QA evaluation & rollback
        # ----------------------------------------------------------------
        print(f"    Agent 3 : Evaluating output quality…")
        qa_decision = qa.evaluate(
            task=vlm_task,
            subtask_key=subtask_key,
            model_used=primary_model,
            baseline_metrics=step_baseline,
            new_image_path=temp_output,
            step_idx=step_idx,
            fallback_model=fallback_model,
        )

        if qa_decision.approved:
            print(f"    Agent 3 : ✅ Approved – all QA metrics within bounds.")

            # ── Color-fix post-processing (4KAgent wavelet correction) ──
            # For generative / diffusion models, correct potential color drift
            # by replacing low-frequency color from the pre-step image while
            # preserving the high-frequency texture from the restored output.
            generative_models = {
                "diffbir", "osediff", "pisasr", "satdiff", "diffusionsat",
                "deblurdiff", "dehazing_diffgan", "ridcp",
            }
            if _COLOR_FIX_AVAILABLE and primary_model in generative_models:
                try:
                    src_bgr = _cv2_cf.imread(current_image)   # pre-step (colour reference)
                    tgt_bgr = _cv2_cf.imread(str(temp_output)) # restored output
                    if src_bgr is not None and tgt_bgr is not None:
                        src_pil = cv2_to_pil(src_bgr)
                        tgt_pil = cv2_to_pil(tgt_bgr)
                        fixed = wavelet_color_fix(tgt_pil, src_pil)
                        fixed_path = base_dir / f"step{step_idx}_{subtask_key}_colfix.png"
                        _cv2_cf.imwrite(str(fixed_path), pil_to_cv2(fixed))
                        current_image = str(fixed_path)
                        print(f"    Color-fix: Wavelet correction applied → {fixed_path.name}")
                        try:
                            fixed_metrics = qa.extract_metrics(current_image)
                            qa.cache_state(
                                image_path=Path(current_image),
                                task=vlm_task,
                                model_used=primary_model,
                                metrics=fixed_metrics,
                                step_idx=step_idx,
                            )
                        except Exception:
                            pass
                    else:
                        current_image = str(temp_output)
                except Exception as cf_err:
                    print(f"    Color-fix: ⚠ Failed ({cf_err}), using raw output.")
                    current_image = str(temp_output)
            else:
                current_image = str(temp_output)
            runtime_step["status"] = "approved"
            runtime_step["qa_reason"] = qa_decision.reason
        else:
            print(f"    Agent 3 : ❌ ROLLBACK – {qa_decision.reason}")
            runtime_step["status"] = "rollback_triggered"
            runtime_step["qa_reason"] = qa_decision.reason

            if qa_decision.rollback_triggered and qa_decision.fallback_model:
                fallback_out2 = base_dir / \
                    f"step{step_idx}_{subtask_key}_rollback_fallback.png"
                print(f"    Agent 3 : Deploying fallback '{qa_decision.fallback_model}'…")
                fb_success = _run_model(
                    qa_decision.fallback_model, current_image, fallback_out2
                )
                if fb_success:
                    # Re-evaluate the fallback output
                    fb_decision = qa.evaluate(
                        task=vlm_task,
                        subtask_key=subtask_key,
                        model_used=qa_decision.fallback_model,
                        baseline_metrics=step_baseline,
                        new_image_path=fallback_out2,
                        step_idx=step_idx,
                        fallback_model=None,
                    )
                    if fb_decision.approved:
                        print(f"    Agent 3 : ✅ Fallback approved.")
                        runtime_step["status"] = "approved_after_qa_rewrite"
                        runtime_step["resolved_model"] = qa_decision.fallback_model
                        runtime_step["resolved_fallback"] = ""
                        runtime_step["rewrite_reason"] = qa_decision.reason
                        runtime_step["qa_reason"] = fb_decision.reason
                        plan_step["assigned_model"] = qa_decision.fallback_model
                        plan_step["fallback_model"] = ""
                        current_image = str(fallback_out2)
                        continue

            # Final rollback: revert to last good cached state
            last_good = qa.get_last_good_state()
            if last_good:
                print(
                    f"    Agent 3 : Reverting to last good state "
                    f"(after step '{last_good.task}')."
                )
                current_image = str(last_good.image_path)
                runtime_step["reverted_to"] = str(last_good.image_path)
            else:
                print("    Agent 3 : No cached state found – keeping original.")

            if qa_decision.skip_task:
                print("    Agent 3 : Skipping this degradation fix entirely.")
                runtime_step["status"] = "skipped_after_rollback"
            else:
                runtime_step["status"] = "rollback_reverted"

    # Persist runtime rewrite/trace plan
    runtime_payload = {
        "input_image": str(Path(image_path).resolve()),
        "profile_path": str(Path(profile_path).resolve()),
        "final_output": current_image,
        "tool_run_gpu_id": tool_run_gpu_id,
        "strict_real_execution": strict_real_execution,
        "execution_plan": runtime_plan,
    }
    runtime_plan_path = base_dir / "execution_plan_runtime.json"
    try:
        with open(runtime_plan_path, "w") as f:
            json.dump(runtime_payload, f, indent=2)
        print(f"  Runtime plan log saved → {runtime_plan_path}")
    except Exception as e:
        print(f"  [!] Could not write runtime plan log: {e}")

    # Keep legacy output location updated for external tooling compatibility.
    legacy_plan_path = Path(__file__).resolve().parent / "outputs" / "execution_plan.json"
    legacy_plan_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(legacy_plan_path, "w") as f:
            json.dump(runtime_payload, f, indent=2)
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Final 4K deliverable normalization
    # -----------------------------------------------------------------------
    if _CV2_AVAILABLE:
        fourk_path = base_dir / "step_final_4k.png"
        try:
            if _ensure_4k_output(Path(current_image), fourk_path):
                current_image = str(fourk_path)
                print("  4K Layer: Final output normalized to 3840x2160.")
            else:
                print("  4K Layer: ⚠ Could not enforce 4K dimensions.")
        except Exception as e:
            print(f"  4K Layer: ⚠ Failed ({e}).")

    # -----------------------------------------------------------------------
    # Final output
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  🏁  PIPELINE COMPLETE")
    print(f"  Final output: {current_image}")
    print("=" * 72 + "\n")

    # Copy final result to a clean output filename if a dir was specified
    if output_dir:
        final_dest = Path(output_dir) / "satagent_final_output.png"
        try:
            shutil.copy(current_image, final_dest)
            print(f"  Copied final output → {final_dest}")
            return str(final_dest)
        except Exception as e:
            print(f"  [!] Could not copy final output: {e}")

    return current_image


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SatAgent – MoE Satellite Image Restoration Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "--input", dest="image", type=str, required=True,
        help="Path to the raw input satellite image."
    )
    parser.add_argument(
        "--profile", "--profile-name", "--profile_name", dest="profile", type=str,
        default="configs/profiles/Aer4K_P.yaml",
        help="Path to the YAML restoration profile."
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=str, default=None,
        help="Optional directory to write the final restored image."
    )
    parser.add_argument(
        "--mock-vlm", action="store_true",
        help="Bypass Agent 1 VLM and inject a deterministic test plan."
    )
    parser.add_argument(
        "--force-full-moe", action="store_true",
        help="Bypass diagnosis and force full Phase 1A→4 pipeline execution."
    )
    parser.add_argument(
        "--tool-run-gpu-id", "--tool_run_gpu_id", dest="tool_run_gpu_id", type=int, default=None,
        help="Optional GPU index for tool execution (e.g., 1 for A5000)."
    )
    parser.add_argument(
        "--strict-real-execution", action="store_true",
        help="Disable proxy and continuity fallbacks; allow only real model execution."
    )
    parser.add_argument(
        "--require-true-models", action="store_true",
        help="Fail tools that use placeholder template scripts; enforce true model wrappers."
    )
    parser.add_argument(
        "--require-preflight", action="store_true",
        help="Run model preflight checks before restoration."
    )
    parser.add_argument(
        "--preflight-smoke", action="store_true",
        help="During preflight, execute a tiny-image smoke invocation per selected model."
    )
    parser.add_argument(
        "--preflight-alias-strict", action="store_true",
        help="Fail preflight if a selected model is an alias/shim wrapper."
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip preflight checks (overrides strict-mode auto-preflight)."
    )
    args = parser.parse_args()

    result = run_pipeline(
        image_path=args.image,
        profile_path=args.profile,
        mock_vlm=args.mock_vlm,
        force_full_moe=args.force_full_moe,
        output_dir=args.output_dir,
        tool_run_gpu_id=args.tool_run_gpu_id,
        strict_real_execution=args.strict_real_execution,
        require_true_models=args.require_true_models,
        require_preflight=args.require_preflight,
        preflight_smoke=args.preflight_smoke,
        preflight_alias_strict=args.preflight_alias_strict,
        skip_preflight=args.skip_preflight,
    )
    print(f"\nFinal restored image: {result}")
