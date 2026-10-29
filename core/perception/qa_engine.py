from __future__ import annotations
# =============================================================================
# Agent 3 – Quality Assurance Engine (The Rollback Architecture)
# =============================================================================
# Operates concurrently with the Restoration Agent as a critical fail-safe.
# After each task the Restoration Agent completes, the resulting intermediate
# tensor is cached and this QA Agent evaluates the new state using rapid
# No-Reference metrics (NIQE, spatial entropy).
#
# The Rollback Protocol:
#   If the QA Agent detects a severe degradation in structural integrity
#   (e.g., a massive spike in the NIQE score indicating artifacting), a
#   Rollback is triggered:
#     1. The corrupted tensor is discarded.
#     2. The system reverts to the cached previous state.
#     3. The JSON plan is dynamically rewritten (fallback model or skip).
# =============================================================================

import logging
import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import piq
    _PIQ_AVAILABLE = True
except ImportError:
    _PIQ_AVAILABLE = False


@dataclass
class QASnapshot:
    """Immutable snapshot of a pipeline state for rollback purposes."""
    image_path: Path
    task: str
    model_used: str
    metrics: Dict[str, float]
    step_idx: int


@dataclass
class QADecision:
    """Result returned by the QA Engine after evaluating a step output."""
    approved: bool
    reason: str
    rollback_triggered: bool = False
    fallback_model: Optional[str] = None
    skip_task: bool = False


class QAEngine:
    """
    Agent 3 – No-Reference Quality Assurance with Rollback Architecture.

    Evaluation metrics used (all no-reference / rapid):
      - NIQE  : Naturalness Image Quality Evaluator (lower=better)
      - Spatial Entropy : measure of structural information (higher=more detail)
      - Blur Variance   : Laplacian variance (higher=sharper)
      - Noise Level     : high-freq noise (lower=cleaner)
      - Haze Density    : dark-channel prior scattering (lower=clearer)
    """

    def __init__(
        self,
        niqe_rollback_threshold: float = 2.0,
        entropy_rollback_threshold: float = 0.15,
        super_resolution_entropy_allowance: float = 0.75,
        attempt_fallback: bool = True,
        max_retries: int = 1,
        force_approve_all_steps: bool = False,
    ):
        self.logger = logging.getLogger("QAEngine")
        self.niqe_delta_threshold = niqe_rollback_threshold
        self.entropy_delta_threshold = entropy_rollback_threshold
        self.super_resolution_entropy_allowance = super_resolution_entropy_allowance
        self.attempt_fallback = attempt_fallback
        self.max_retries = max_retries
        self.force_approve_all_steps = force_approve_all_steps

        # Cache of approved snapshots (for rollback)
        self._state_cache: list[QASnapshot] = []
        # Per-task retry counters
        self._retry_counts: Dict[str, int] = {}
        self._eps = 1e-6

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_metrics(image_path: str) -> Dict[str, float]:
        """
        Rapid no-reference metric extraction.
        Identical call signature to ExpertIQAEvaluator.extract_metrics so the
        QA engine is a drop-in companion during the feedback loop.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"QA: corrupt or missing tensor: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = (
            torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )

        # NIQE (lower = better natural image statistics)
        niqe_score = 99.99
        if _PIQ_AVAILABLE:
            try:
                niqe_score = round(piq.niqe(tensor_img).item(), 4)
            except Exception:
                pass

        # Spatial entropy (higher = richer structural detail)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / (hist.sum() + 1e-7)
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-7)))

        # Blur variance (Laplacian, higher = sharper)
        blur_variance = round(cv2.Laplacian(gray, cv2.CV_64F).var(), 4)

        # Noise level (high-frequency residual, lower = cleaner)
        noise_level = round(
            float(np.std(cv2.absdiff(gray, cv2.medianBlur(gray, 3)))), 4
        )

        # Haze density (dark-channel, lower = clearer atmosphere)
        haze_density = round(
            float(
                np.mean(
                    cv2.erode(
                        np.min(img, axis=2),
                        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                    )
                )
            ),
            4,
        )

        return {
            "width": float(img.shape[1]),
            "height": float(img.shape[0]),
            "niqe_score": niqe_score,
            "spatial_entropy": round(entropy, 4),
            "blur_variance": blur_variance,
            "noise_level": noise_level,
            "haze_density": haze_density,
        }

    # ------------------------------------------------------------------
    # State caching
    # ------------------------------------------------------------------

    def cache_state(
        self,
        image_path: Path,
        task: str,
        model_used: str,
        metrics: Dict[str, float],
        step_idx: int,
    ) -> None:
        """Cache a successfully-approved pipeline state for potential rollback."""
        self._state_cache.append(
            QASnapshot(
                image_path=image_path,
                task=task,
                model_used=model_used,
                metrics=metrics,
                step_idx=step_idx,
            )
        )

    def get_last_good_state(self) -> Optional[QASnapshot]:
        """Returns the most recently approved snapshot."""
        return self._state_cache[-1] if self._state_cache else None

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        task: str,
        subtask_key: str,
        model_used: str,
        baseline_metrics: Dict[str, float],
        new_image_path: Path,
        step_idx: int,
        fallback_model: Optional[str] = None,
    ) -> QADecision:
        """
        Evaluate the output of a restoration step. Returns a QADecision.

        Args:
            task: Human-readable task name (e.g. "cloud removal").
            subtask_key: Internal executor subtask key (e.g. "cloud_removal").
            model_used: The tool_name of the model that produced the output.
            baseline_metrics: Metrics computed BEFORE the step ran.
            new_image_path: Path to the output image produced by the step.
            step_idx: Current pipeline step index (for retry tracking).
            fallback_model: If provided, use as QADecision.fallback_model on rollback.
        """
        try:
            new_metrics = self.extract_metrics(str(new_image_path))
        except Exception as e:
            return QADecision(
                approved=False,
                reason=f"QA metric extraction failed: {e}",
                rollback_triggered=True,
                fallback_model=fallback_model,
                skip_task=not self.attempt_fallback,
            )

        if self.force_approve_all_steps:
            self.cache_state(new_image_path, task, model_used, new_metrics, step_idx)
            return QADecision(approved=True, reason="Validation mode: force-approved.")

        # ---- Per-subtask physics-aware checks ----
        degradation_detected = False
        reason = ""

        # NIQE universal check: a large spike means hallucination / severe artifacting
        niqe_delta = new_metrics["niqe_score"] - baseline_metrics.get("niqe_score", 0)
        if niqe_delta > self.niqe_delta_threshold:
            degradation_detected = True
            reason = (
                f"NIQE spiked by +{niqe_delta:.3f} "
                f"({baseline_metrics.get('niqe_score',0):.3f} → {new_metrics['niqe_score']:.3f}), "
                f"indicating severe artifacting or hallucianted geography."
            )

        # Spatial entropy: must not collapse (structure loss)
        if not degradation_detected:
            ent_delta = (
                baseline_metrics.get("spatial_entropy", 0)
                - new_metrics["spatial_entropy"]
            )

            # Allow some entropy drop if task-specific metrics improve.
            entropy_allowance = self.entropy_delta_threshold
            base_noise = baseline_metrics.get("noise_level", 0.0)
            base_blur = baseline_metrics.get("blur_variance", 0.0)
            base_haze = baseline_metrics.get("haze_density", 0.0)

            if subtask_key in ("denoising", "artifact_removal"):
                noise_improve = max(0.0, base_noise - new_metrics.get("noise_level", 0.0))
                if base_noise > 0 and (noise_improve / base_noise) > 0.15:
                    entropy_allowance += 0.6
            elif subtask_key == "deblurring":
                blur_gain = new_metrics.get("blur_variance", 0.0) - base_blur
                if base_blur > 0 and (blur_gain / base_blur) > 0.15:
                    entropy_allowance += 0.6
            elif subtask_key in ("dehazing", "cloud_removal"):
                haze_improve = max(0.0, base_haze - new_metrics.get("haze_density", 0.0))
                if base_haze > 0 and (haze_improve / base_haze) > 0.15:
                    entropy_allowance += 0.4
            elif subtask_key == "super_resolution":
                # Bypass strict entropy checks for SR to prevent false rollbacks of upscaled generative output.
                entropy_allowance = float('inf') 

            if ent_delta > entropy_allowance:
                degradation_detected = True
                reason = (
                    f"Spatial entropy collapsed by -{ent_delta:.3f} "
                    f"({baseline_metrics.get('spatial_entropy',0):.3f} → "
                    f"{new_metrics['spatial_entropy']:.3f}), "
                    f"exceeding allowance {entropy_allowance:.3f}."
                )

        # Task-specific physical constraints
        if not degradation_detected:
            if subtask_key == "deblurring":
                if new_metrics["blur_variance"] + self._eps < baseline_metrics.get(
                    "blur_variance", 0
                ):
                    degradation_detected = True
                    reason = (
                        f"Laplacian variance decreased "
                        f"({baseline_metrics.get('blur_variance',0):.2f} → "
                        f"{new_metrics['blur_variance']:.2f}), image got blurrier."
                    )
            elif subtask_key in ("denoising", "artifact_removal"):
                if new_metrics["noise_level"] > baseline_metrics.get("noise_level", 999) + self._eps:
                    degradation_detected = True
                    reason = (
                        f"High-frequency noise increased "
                        f"({baseline_metrics.get('noise_level',0):.2f} → "
                        f"{new_metrics['noise_level']:.2f})."
                    )
            elif subtask_key in ("cloud_removal", "dehazing"):
                base_haze_d = baseline_metrics.get("haze_density", 999)
                new_haze_d = new_metrics["haze_density"]
                haze_increase = new_haze_d - base_haze_d
                # Ignore tiny fluctuations from histogram/statistical jitter.
                # Require both an absolute and relative worsening before rollback.
                abs_tol = 1.5
                rel_tol = 0.05 * max(base_haze_d, 1.0)
                if haze_increase > max(abs_tol, rel_tol):
                    degradation_detected = True
                    reason = (
                        f"Dark-channel scattering increased "
                        f"({base_haze_d:.2f} → {new_haze_d:.2f})."
                    )

        if not degradation_detected:
            # Cache this good state
            self.cache_state(new_image_path, task, model_used, new_metrics, step_idx)
            return QADecision(approved=True, reason="All QA metrics within bounds.")

        # ---- Rollback decision ----
        retry_key = f"{step_idx}:{subtask_key}"
        retries_used = self._retry_counts.get(retry_key, 0)

        if self.attempt_fallback and retries_used < self.max_retries and fallback_model:
            self._retry_counts[retry_key] = retries_used + 1
            return QADecision(
                approved=False,
                reason=reason,
                rollback_triggered=True,
                fallback_model=fallback_model,
                skip_task=False,
            )
        else:
            return QADecision(
                approved=False,
                reason=reason,
                rollback_triggered=True,
                fallback_model=None,
                skip_task=True,
            )
