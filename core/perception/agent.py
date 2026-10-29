from __future__ import annotations
import gc
import os
import torch
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

from . import prompts
from .expert_iqa import ExpertIQAEvaluator
from .upscale_logic import UpscaleCalculator
from core.profile.manager import ProfileManager
from llm.perception_vlm import SatelliteVisionAgent
from core.profile.selector import SatModelSelector
from utils.logger import setup_logger
from utils.schedule_experience import load_experience


# ---------------------------------------------------------------------------
# Degradation → internal task key mapping (full MoE spec)
# ---------------------------------------------------------------------------
DEGRA_TO_TASK_MAP = {
    # VLM label             → allowed_agenda / plan key
    "cloud cover":               "cloud removal",
    "atmospheric haze":          "dehazing",
    "shadow occlusion":          "shadow correction",
    "sensor noise":              "sensor denoising",
    "orbital motion blur":       "sensor deblurring",
    "stripe artifact":           "destriping",
    "compression artifact":      "compression artifact removal",
    "multispectral mismatch":    "pan-sharpening",
    "low resolution":            "super-resolution"
}

# Common VLM/free-form aliases mapped to canonical degradation labels.
DEGRA_ALIASES = {
    "clouds": "cloud cover",
    "thick clouds": "cloud cover",
    "haze": "atmospheric haze",
    "shadow": "shadow occlusion",
    "noise": "sensor noise",
    "noise_level": "sensor noise",
    "sensor blur": "orbital motion blur",
    "motion blur": "orbital motion blur",
    "striping": "stripe artifact",
    "jpeg artifact": "compression artifact",
    "compression artifacts": "compression artifact",
}

# Physics-safe order used as hard guardrail after VLM planning.
PHYSICS_PRIORITY = {
    "cloud removal": 10,
    "dehazing": 20,
    "shadow correction": 30,
    "destriping": 40,
    "sensor denoising": 50,
    "sensor deblurring": 60,
    "compression artifact removal": 70,
    "pan-sharpening": 80,
    "multi-temporal fusion": 90,
    "super-resolution (4x)": 100,
    "super-resolution (2x)": 110,
    "super-resolution": 120,
}

# Task aliases from VLM/free-text outputs to canonical scheduler keys.
TASK_ALIASES = {
    "thick cloud removal": "cloud removal",
    "global dehazing": "dehazing",
    "thermal/iso denoising": "sensor denoising",
    "thermal denoising": "sensor denoising",
    "iso denoising": "sensor denoising",
    "orbital motion deblurring": "sensor deblurring",
    "motion deblurring": "sensor deblurring",
    "destriping / dead-pixel inpainting": "destriping",
    "destriping / dead-pixel": "destriping",
    "artifact removal": "compression artifact removal",
    "upscaling to 4k": "super-resolution (4x)",
    "super-resolution 4k": "super-resolution (4x)",
}

# Strict plan metadata aligned with the SatAgent MoE spec.
TASK_PHASE_SPECS: Dict[str, Dict[str, Any]] = {
    "cloud removal": {
        "phase_id": "1A",
        "phase_name": "Atmospheric & Environmental Anomalies",
        "routing_path": "local",
        "datasets": ["SEN12MS-CR", "RICE-II"],
    },
    "dehazing": {
        "phase_id": "1B",
        "phase_name": "Atmospheric & Environmental Anomalies",
        "routing_path": "global",
        "datasets": ["RS-Haze", "RRSHID"],
    },
    "shadow correction": {
        "phase_id": "1C",
        "phase_name": "Atmospheric & Environmental Anomalies",
        "routing_path": "local",
        "datasets": ["RSD", "UIIS"],
    },
    "sensor denoising": {
        "phase_id": "2A",
        "phase_name": "Sensor & Hardware Physics",
        "routing_path": "global",
        "datasets": ["xView (noise-injected)", "UC Merced (noise-injected)"],
    },
    "sensor deblurring": {
        "phase_id": "2B",
        "phase_name": "Sensor & Hardware Physics",
        "routing_path": "global",
        "datasets": ["DOTA (motion-blur injected)"],
    },
    "destriping": {
        "phase_id": "2C",
        "phase_name": "Sensor & Hardware Physics",
        "routing_path": "local",
        "datasets": ["Aqua MODIS", "EO-1 Hyperion"],
    },
    "compression artifact removal": {
        "phase_id": "2D",
        "phase_name": "Sensor & Hardware Physics",
        "routing_path": "global",
        "datasets": ["AID (heavy JPEG compression)"],
    },
    "pan-sharpening": {
        "phase_id": "3A",
        "phase_name": "Core Fusion",
        "routing_path": "global",
        "datasets": ["GaoFen-2", "WorldView-3"],
    },
    "fusion": {
        "phase_id": "3A",
        "phase_name": "Core Fusion",
        "routing_path": "global",
        "datasets": ["GaoFen-2", "WorldView-3"],
    },
    "multi-temporal fusion": {
        "phase_id": "3B",
        "phase_name": "Core Fusion",
        "routing_path": "global",
        "datasets": ["SpaceNet", "OSCD"],
    },
    "super-resolution": {
        "phase_id": "4",
        "phase_name": "Super-Resolution (4K Layer)",
        "routing_path": "global",
        "datasets": ["UC Merced Land Use", "DOTA"],
    },
    "super-resolution (2x)": {
        "phase_id": "4",
        "phase_name": "Super-Resolution (4K Layer)",
        "routing_path": "global",
        "datasets": ["UC Merced Land Use", "DOTA"],
    },
    "super-resolution (4x)": {
        "phase_id": "4",
        "phase_name": "Super-Resolution (4K Layer)",
        "routing_path": "global",
        "datasets": ["UC Merced Land Use", "DOTA"],
    },
}


class PerceptionAgentOrchestrator:
    """
    Agent 1 – The Perception Engine (Diagnosis & Planning).

    Workflow:
      Stage 1 – Expert IQA: extract heuristic metrics (no GPU).
      Stage 2 – VLM Diagnosis: identify degradations from image + metrics.
      Stage 3 – Upscale logic: append SR tasks if target resolution requires it.
      Stage 4 – VLM Task Planning: physics-ordered execution plan via JSON.
      Stage 5 – Model assignment: map tasks → model weights per profile mode.
    """

    def __init__(self, profile_manager: ProfileManager):
        self.logger = setup_logger("PerceptionOrchestrator")
        self.profile = profile_manager.config
        self.health_data = {}
        self.vlm = None
        if os.environ.get("SATA_DISABLE_VLM", "0") != "1":
            try:
                self.vlm = SatelliteVisionAgent(
                    model_id=self.profile.perception.vlm_model_id,
                    system_message="You are a strict satellite telemetry diagnostician.",
                )
            except Exception as e:
                self.logger.warning(
                    f"VLM init failed, switching to heuristic planning: {e}"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def load_health_data(self, health_path: str) -> Dict:
        """Loads or initializes model health telemetry to prevent blacklisted model usage."""
        import json
        health_path_obj = Path(health_path)
        if health_path_obj.exists():
            try:
                with open(health_path_obj, 'r') as f:
                    self.health_data = json.load(f)
                self.logger.info(f"Loaded health data from {health_path}")
            except Exception as e:
                self.logger.warning(f"Failed to parse health file: {e}")
                self.health_data = {}
        else:
            self.health_data = {}
        return self.health_data

    def _vlm_call(self, image_path: str, prompt: str, max_tokens: int) -> Dict:
        if self.vlm is None:
            return {}
        try:
            inputs = self.vlm.prepare_inputs(image_path, prompt)
        except Exception as e:
            self.logger.error(f"VLM input preparation failed: {e}")
            return {}
        result = {}
        try:
            output = self.vlm.generate_json(inputs, max_new_tokens=max_tokens)
            if isinstance(output, dict):
                result = output
        except Exception as e:
            self.logger.error(f"VLM call crashed: {e}")
        finally:
            del inputs
            torch.cuda.empty_cache()
        return result

    @staticmethod
    def _canonicalize_degradation(label: str) -> str:
        key = label.lower().strip()
        return DEGRA_ALIASES.get(key, key)

    def _normalize_degradations(self, raw_degradations: List[str]) -> List[str]:
        out: List[str] = []
        for raw in raw_degradations:
            canonical = self._canonicalize_degradation(raw)
            if canonical in DEGRA_TO_TASK_MAP:
                out.append(canonical)
        # Ordered unique
        seen = set()
        unique = []
        for d in out:
            if d not in seen:
                seen.add(d)
                unique.append(d)
        return unique

    @staticmethod
    def _heuristic_degradations(metrics: Dict[str, float]) -> List[str]:
        out: List[str] = []
        if metrics.get("haze_density", 0.0) > 60:
            out.append("atmospheric haze")
        if metrics.get("blur_variance", 9999.0) < 50:
            out.append("orbital motion blur")
        if metrics.get("niqe_score", 0.0) > 8.0 or metrics.get("noise_level", 0.0) > 80:
            out.append("sensor noise")
        if metrics.get("clipiqa_plus", 1.0) >= 0 and metrics.get("clipiqa_plus", 1.0) < 0.3:
            out.append("compression artifact")
        return out

    @staticmethod
    def _is_sr_task(task: str) -> bool:
        return task.startswith("super-resolution")

    def _dedupe_non_sr(self, tasks: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for task in tasks:
            if self._is_sr_task(task):
                out.append(task)
                continue
            if task not in seen:
                seen.add(task)
                out.append(task)
        return out

    @staticmethod
    def _canonicalize_task(task: str) -> str:
        key = task.lower().strip()
        return TASK_ALIASES.get(key, key)

    def _is_allowed_task(self, task: str) -> bool:
        allowed = set(self.profile.restoration.allowed_agenda)
        if task in allowed:
            return True
        # Treat all SR variants as allowed when "super-resolution" is enabled.
        if self._is_sr_task(task) and "super-resolution" in allowed:
            return True
        return False

    def _enforce_physics_order(
        self,
        requested_tasks: List[str],
        candidate_plan: List[str],
    ) -> List[str]:
        requested = list(requested_tasks)
        if not requested:
            return []

        req_counts = Counter(requested)
        merged: List[str] = []

        # Keep candidate order first, but only if task/count belongs to request.
        for task in candidate_plan:
            task = self._canonicalize_task(task)
            if req_counts[task] > 0:
                merged.append(task)
                req_counts[task] -= 1

        # Append any missing requested tasks.
        for task in requested:
            while req_counts[task] > 0:
                merged.append(task)
                req_counts[task] -= 1

        # Hard-guard physics ordering rules.
        ordered = sorted(
            enumerate(merged),
            key=lambda x: (PHYSICS_PRIORITY.get(x[1], 999), x[0]),
        )
        return [task for _, task in ordered]

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def execute_pipeline(self, image_path: str) -> Dict[str, Any]:
        # ---- Stage 1: Expert IQA ----
        self.logger.info("Stage 1: Expert IQA extraction…")
        metrics = ExpertIQAEvaluator.extract_metrics(image_path)

        # ---- Stage 2: VLM degradation diagnosis ----
        self.logger.info("Stage 2: VLM degradation reasoning…")
        iqa_str = ExpertIQAEvaluator.format_for_prompt(metrics)
        diagnosis_prompt = prompts.sat_perception_system_message.format(
            iqa_result=iqa_str
        )
        diagnosis_res = self._vlm_call(
            image_path, diagnosis_prompt,
            max_tokens=self.profile.perception.max_new_tokens
        )

        degradations: List[str] = []
        raw_degradations = diagnosis_res.get("degradations", [])
        if isinstance(raw_degradations, list):
            degradations = self._normalize_degradations(
                [d for d in raw_degradations if isinstance(d, str)]
            )
        if not degradations:
            degradations = self._heuristic_degradations(metrics)

        image_desc: str = diagnosis_res.get(
            "image_description", "Satellite telemetry analysis."
        )

        # ---- Stage 3: Agenda construction ----
        self.logger.info("Stage 3: Agenda construction…")
        agenda: List[str] = []
        
        for d in degradations:
            # 1. SKIP "low resolution" here. 
            # We don't want the VLM's label to add a generic SR task yet.
            if d == "low resolution":
                continue
                
            task = DEGRA_TO_TASK_MAP.get(d)
            if task:
                agenda.append(task)

        # 2. Let the Math-based Scaler handle SR exclusively.
        # This ensures we only add the EXACT scale needed for your 4096px target.
        target_res = self.profile.restoration.target_resolution
        if 4096 in target_res or 3840 in target_res:
            scale = UpscaleCalculator.get_target_factor(
                metrics["width"], metrics["height"]
            )
            if scale > 1:
                # This correctly appends "super-resolution (4x)" based on Eq. 1
                agenda.extend(UpscaleCalculator.append_sr_tasks(scale))

        # 3. Use a strict deduper.
        # Your previous _dedupe_non_sr was likely skipping SR tasks.
        # We'll use a standard ordered dict dedupe here.
        seen = set()
        clean_agenda = []
        for t in agenda:
            if t not in seen:
                clean_agenda.append(t)
                seen.add(t)
        agenda = clean_agenda

        # Optional temporal fusion gate
        if not self.profile.restoration.enable_temporal_fusion:
            agenda = [t for t in agenda if t != "multi-temporal fusion"]

        # Filter to allowed tasks
        agenda = [t for t in agenda if self._is_allowed_task(t)]

        # ---- Stage 4: VLM physics-ordered task planning ----
        # Load empirical scheduling experience (4KAgent memory + satellite rules)
        experience_path = Path(__file__).resolve().parents[2] / "memory" / "schedule_experience.json"
        experience = load_experience(experience_path)

        vlm_ordered_tasks = agenda
        if agenda:
            self.logger.info(
                f"Stage 4: VLM task planning for {len(agenda)} task(s)…"
            )
            plan_prompt = prompts.sat_plan_system_message.format(
                image_description=image_desc,
                degradations=degradations,
                tasks=agenda,
                experience=experience,
            )
            plan_res = self._vlm_call(
                image_path, plan_prompt,
                max_tokens=self.profile.perception.max_new_tokens
            )
            raw_plan = plan_res.get("plan", [])
            if isinstance(raw_plan, list) and raw_plan:
                proposed = [
                    self._canonicalize_task(str(t).strip())
                    for t in raw_plan
                    if isinstance(t, str)
                ]
                vlm_ordered_tasks = self._enforce_physics_order(agenda, proposed)
            else:
                vlm_ordered_tasks = self._enforce_physics_order(agenda, agenda)
        else:
            self.logger.warning("No tasks identified. Skipping VLM task planning.")

        # ---- Stage 5: Model assignment ----
        mode = self.profile.restoration.preference
        final_plan: List[Dict[str, Any]] = []
        for idx, task in enumerate(vlm_ordered_tasks, start=1):
            canonical_task = self._canonicalize_task(task)
            primary_model = SatModelSelector.get_model(canonical_task, mode)
            fallback_model = SatModelSelector.get_fallback_model(
                canonical_task, primary_model, mode
            )
            candidates = SatModelSelector.get_model_candidates(canonical_task)
            task_spec = TASK_PHASE_SPECS.get(
                canonical_task,
                {
                    "phase_id": "unknown",
                    "phase_name": "Unclassified",
                    "routing_path": "global",
                    "datasets": [],
                },
            )
            routing = task_spec["routing_path"]
            mask_tool = SatModelSelector.get_mask_tool(canonical_task) if routing == "local" else None

            final_plan.append({
                "step_id": idx,
                "phase_id": task_spec["phase_id"],
                "phase_name": task_spec["phase_name"],
                "stage_task": canonical_task,
                "routing_path": routing,
                "assigned_model": primary_model,
                "fallback_model": fallback_model or "",
                "model_candidates": candidates,
                "benchmark_datasets": task_spec["datasets"],
                "masking": {
                    "required": routing == "local",
                    "mask_tool": mask_tool or "",
                    "stitching": "poisson" if routing == "local" else "",
                },
                "execution_mode": mode,
            })

        gc.collect()
        torch.cuda.empty_cache()

        return {
            "metadata": {
                "original_resolution": f"{int(metrics['width'])}x{int(metrics['height'])}",
                "global_mode": mode,
                "geographic_caption": image_desc,
                "planner": "Perception Engine (Agent 1)",
                "plan_schema": "satagent_moe_plan_v1",
            },
            "expert_iqa_telemetry": metrics,
            "identified_degradations": degradations,
            "execution_plan": final_plan,
        }
