from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sata.executor import executor


PHASE_BY_SUBTASK = {
    "roi_masking": "0",
    "cloud_removal": "1A",
    "dehazing": "1B",
    "shadow_correction": "1C",
    "denoising": "2A",
    "deblurring": "2B",
    "destriping": "2C",
    "artifact_removal": "2D",
    "pan_sharpen": "3A",
    "temporal_fusion": "3B",
    "super_resolution": "4",
}

PHASE_LABEL = {
    "0": "ROI Masking",
    "1A": "Thick Cloud Removal",
    "1B": "Global Dehazing",
    "1C": "Shadow Correction",
    "2A": "Thermal/ISO Denoising",
    "2B": "Motion Deblurring",
    "2C": "Destriping/Dead-Pixel",
    "2D": "Artifact Removal",
    "3A": "Pan-Sharpening",
    "3B": "Multi-Temporal Fusion",
    "4": "Super-Resolution (4K)",
}


@dataclass
class ToolResult:
    phase: str
    phase_label: str
    subtask: str
    routing: str
    tool_name: str
    script_path: str
    placeholder_wrapper: bool
    missing_checkpoints: List[str]
    ready: bool
    status: str
    error: str
    output_path: str


def _make_test_image(path: Path, size: int = 512) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    xv = np.linspace(0, 255, size, dtype=np.uint8)
    yv = np.linspace(0, 255, size, dtype=np.uint8)
    img[:, :, 0] = np.tile(xv, (size, 1))
    img[:, :, 1] = np.tile(yv[:, None], (1, size))
    img[:, :, 2] = ((img[:, :, 0].astype(np.uint16) + img[:, :, 1].astype(np.uint16)) // 2).astype(np.uint8)
    noise = np.random.randint(0, 25, (size, size, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    cv2.imwrite(str(path), img)
    return path


def _extract_checkpoint_paths(tool: Any) -> List[Path]:
    """
    Best-effort checkpoint path extraction from tool command options.
    """
    candidates: List[Path] = []
    try:
        opts = tool._get_cmd_opts()  # noqa: SLF001
    except Exception:
        opts = []
    if not isinstance(opts, list):
        return []

    key_like = {"--ckpt", "--ckpts", "--checkpoint", "--model_path", "--weight", "--weights", "--ram_ft_path", "--ram_path"}
    path_pat = re.compile(r".*\.(pth|pt|ckpt|safetensors|npz|pkl)$", re.IGNORECASE)
    i = 0
    while i < len(opts):
        tok = str(opts[i])
        next_tok = str(opts[i + 1]) if i + 1 < len(opts) else ""
        if tok in key_like and next_tok and next_tok.lower() != "none":
            candidates.append(Path(next_tok))
            i += 2
            continue
        if path_pat.match(tok) and tok.lower() != "none":
            candidates.append(Path(tok))
        i += 1

    resolved: List[Path] = []
    for p in candidates:
        if p.is_absolute():
            resolved.append(p)
        else:
            wd = Path(getattr(tool, "work_dir", os.getcwd()))
            resolved.append((wd / p).resolve())
    dedup: List[Path] = []
    seen = set()
    for p in resolved:
        key = str(p)
        if key not in seen:
            seen.add(key)
            dedup.append(p)
    return dedup


def _classify_failure(err: str, placeholder: bool, missing_ckpts: List[str], require_true_models: bool) -> str:
    e = (err or "").lower()
    if require_true_models and placeholder:
        return "blocked_placeholder_wrapper"
    if missing_ckpts:
        return "blocked_missing_checkpoint"
    if "timed out after" in e or "timeoutexpired" in e:
        return "blocked_timeout"
    if "no such file or directory" in e:
        return "blocked_missing_file_or_path"
    if "no module named" in e or "modulenotfounderror" in e:
        return "blocked_missing_dependency"
    if "usage:" in e or "unrecognized arguments" in e or "the following arguments are required" in e:
        return "blocked_cli_mismatch"
    if "cuda out of memory" in e:
        return "blocked_vram"
    if "permission denied" in e:
        return "blocked_permission"
    if err:
        return "blocked_runtime_error"
    return "blocked_no_output"


def _write_markdown(out_md: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# SatAgent Full Tool Battery Readiness")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at_utc']}")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- GPU: `{payload['gpu_id']}`")
    lines.append(f"- Input: `{payload['input']}`")
    lines.append(f"- Overall: **{payload['summary']['ready']}/{payload['summary']['total']} ready**")
    lines.append("")
    lines.append("## Phase Summary")
    for p in payload["phase_summary"]:
        lines.append(
            f"- Phase {p['phase']} ({p['label']}): {p['ready']}/{p['total']} ready"
        )
    lines.append("")
    lines.append("## Model Path Results")
    for r in payload["results"]:
        mark = "READY" if r["ready"] else "BLOCKED"
        lines.append(
            f"- [{mark}] Phase {r['phase']} | {r['subtask']} | {r['tool_name']} | "
            f"routing={r['routing']} | status={r['status']}"
        )
        if r["missing_checkpoints"]:
            lines.append(f"  missing_ckpts: {', '.join(r['missing_checkpoints'])}")
        if r["error"]:
            short = r["error"].replace("\n", " ")
            lines.append(f"  error: {short[:400]}")
    out_md.write_text("\n".join(lines) + "\n")


def run_battery(
    output_json: Path,
    output_md: Path,
    workspace: Path,
    gpu_id: int,
    strict_real_execution: bool,
    require_true_models: bool,
    tool_timeout_sec: float | None,
    skip_tools: List[str] | None = None,
    no_exec: bool = False,
) -> Dict[str, Any]:
    workspace = workspace.resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    input_path = _make_test_image((workspace / "input" / "input.png").resolve()).resolve()

    os.environ["SATA_ENABLE_BASELINE_PROXY"] = "0" if strict_real_execution else "1"
    os.environ["SATA_TOOL_GRACEFUL_FALLBACK"] = "0" if strict_real_execution else "1"
    os.environ["SATA_REQUIRE_TRUE_MODELS"] = "1" if require_true_models else "0"
    os.environ["SATA_ENFORCE_REAL_BACKEND"] = "1" if strict_real_execution else "0"
    if tool_timeout_sec is not None and tool_timeout_sec > 0:
        os.environ["SATA_TOOL_CMD_TIMEOUT_SEC"] = str(tool_timeout_sec)

    skip_set = {t.strip().lower() for t in (skip_tools or []) if t.strip()}
    results: List[ToolResult] = []
    for subtask in sorted(executor.subtasks):
        phase = PHASE_BY_SUBTASK.get(subtask, "unknown")
        phase_label = PHASE_LABEL.get(phase, "Unknown")
        routing = "pre-process" if subtask == "roi_masking" else executor.get_routing_mode(subtask)
        for tool in executor.toolbox_router.get(subtask, []):
            if tool.tool_name.lower() in skip_set:
                results.append(
                    ToolResult(
                        phase=phase,
                        phase_label=phase_label,
                        subtask=subtask,
                        routing=routing,
                        tool_name=tool.tool_name,
                        script_path=str(getattr(tool, "script_path", "")),
                        placeholder_wrapper=False,
                        missing_checkpoints=[],
                        ready=False,
                        status="skipped_by_user",
                        error="",
                        output_path="",
                    )
                )
                continue
            run_dir = (workspace / f"{subtask}__{tool.tool_name}").resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            out_path = (run_dir / "output.png").resolve()
            if out_path.exists():
                out_path.unlink()

            script_path = ""
            if getattr(tool, "script_path", None) is not None:
                script_path = str(tool.script_path)

            placeholder = False
            try:
                placeholder = bool(tool._is_placeholder_template_script())  # noqa: SLF001
            except Exception:
                placeholder = False

            ckpts = _extract_checkpoint_paths(tool)
            missing_ckpts = [str(p) for p in ckpts if not p.exists()]

            err = ""
            ready = False
            status = ""
            if no_exec:
                status = _classify_failure(
                    err="",
                    placeholder=placeholder,
                    missing_ckpts=missing_ckpts,
                    require_true_models=require_true_models,
                )
                if status == "blocked_no_output":
                    status = "ready_probe_only"
                    ready = True
            else:
                try:
                    tool(
                        input_path=input_path,
                        output_path=out_path,
                        silent=True,
                        run_gpu_id=gpu_id,
                    )
                    ready = out_path.exists()
                    status = "ready" if ready else "blocked_no_output"
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    status = _classify_failure(
                        err=err,
                        placeholder=placeholder,
                        missing_ckpts=missing_ckpts,
                        require_true_models=require_true_models,
                    )

            if not ready and not status:
                status = _classify_failure(
                    err=err,
                    placeholder=placeholder,
                    missing_ckpts=missing_ckpts,
                    require_true_models=require_true_models,
                )

            results.append(
                ToolResult(
                    phase=phase,
                    phase_label=phase_label,
                    subtask=subtask,
                    routing=routing,
                    tool_name=tool.tool_name,
                    script_path=script_path,
                    placeholder_wrapper=placeholder,
                    missing_checkpoints=missing_ckpts,
                    ready=ready,
                    status=status,
                    error=err,
                    output_path=str(out_path),
                )
            )

    phase_summary: List[Dict[str, Any]] = []
    for phase in sorted({r.phase for r in results}, key=lambda p: (p != "0", p)):
        rows = [r for r in results if r.phase == phase]
        phase_summary.append(
            {
                "phase": phase,
                "label": PHASE_LABEL.get(phase, "Unknown"),
                "total": len(rows),
                "ready": sum(1 for x in rows if x.ready),
            }
        )

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "strict_real_true_models" if (strict_real_execution and require_true_models) else "custom",
        "gpu_id": gpu_id,
        "input": str(input_path.resolve()),
        "summary": {
            "total": len(results),
            "ready": sum(1 for r in results if r.ready),
            "blocked": sum(1 for r in results if not r.ready),
        },
        "phase_summary": phase_summary,
        "results": [r.__dict__ for r in results],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    _write_markdown(output_md, payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser("Run full SatAgent tool battery and emit readiness report.")
    ap.add_argument("--gpu-id", type=int, default=1)
    ap.add_argument("--workspace", type=Path, default=Path("sata/tests/workspace/readiness_battery_true_models"))
    ap.add_argument("--output-json", type=Path, default=Path("sata/tests/report_tool_readiness_true_models.json"))
    ap.add_argument("--output-md", type=Path, default=Path("sata/tests/report_tool_readiness_true_models.md"))
    ap.add_argument("--strict-real-execution", action="store_true")
    ap.add_argument("--require-true-models", action="store_true")
    ap.add_argument("--tool-timeout-sec", type=float, default=20.0)
    ap.add_argument(
        "--skip-tools",
        type=str,
        default="",
        help="Comma-separated tool names to skip (e.g. restormer,hat_psnr).",
    )
    ap.add_argument(
        "--no-exec",
        action="store_true",
        help="Do not execute tools; only probe wrappers/checkpoints.",
    )
    args = ap.parse_args()

    env_skip = os.environ.get("SATA_SKIP_TOOLS", "")
    raw_skip = ",".join([args.skip_tools, env_skip]).strip(",")
    skip_tools = [t for t in raw_skip.split(",") if t.strip()]

    payload = run_battery(
        output_json=args.output_json,
        output_md=args.output_md,
        workspace=args.workspace,
        gpu_id=args.gpu_id,
        strict_real_execution=args.strict_real_execution,
        require_true_models=args.require_true_models,
        tool_timeout_sec=args.tool_timeout_sec,
        skip_tools=skip_tools,
        no_exec=args.no_exec,
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"JSON report: {args.output_json}")
    print(f"MD report  : {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
