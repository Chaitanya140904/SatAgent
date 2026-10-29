from __future__ import annotations
"""
SatAgent Expert IQA Evaluator.

Stage 1 of the Perception Engine pipeline.
Computes heuristic + pyiqa-based metrics from the raw satellite image
WITHOUT loading the VLM, so it is fast and CPU-compatible.

Metrics returned:
  - width, height (px)
  - niqe_score    (lower = better)
  - blur_variance (higher = sharper, via Laplacian)
  - noise_level   (higher = noisier, from high-freq residual)
  - haze_density  (higher = hazier, from dark-channel prior)
  - clipiqa_plus  (optional, from pyiqa when SATA_ENABLE_PYIQA=1)
  - topiq_nr      (optional, from pyiqa when SATA_ENABLE_PYIQA=1)
"""
import os
import warnings
from typing import Dict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Low-level metric calculators (OpenCV / NumPy – no GPU needed)
# ---------------------------------------------------------------------------

def _blur_variance(gray: np.ndarray) -> float:
    """Laplacian variance – proxy for sharpness."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(round(lap.var(), 4))


def _noise_level(gray: np.ndarray) -> float:
    """
    High-frequency residual as noise estimator.
    Subtracts a Gaussian-blurred version; variance of residual ≈ noise power.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    return float(round(float(np.var(residual)), 4))


def _haze_density(bgr: np.ndarray) -> float:
    """
    Dark-channel prior haze estimate.
    Returns a value in [0, 100]; higher = more haze.
    """
    patch_size = 15
    dark_channel = np.min(bgr, axis=2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size)
    )
    dark_channel = cv2.erode(dark_channel, kernel)
    density = float(np.mean(dark_channel))
    return float(round(density, 4))


def _niqe_approx(gray: np.ndarray) -> float:
    """
    Lightweight NIQE approximation using local statistics.
    Falls back to this if pyiqa is unavailable.
    """
    mu = float(np.mean(gray))
    sigma = float(np.std(gray))
    kurtosis_approx = float(
        np.mean(((gray.astype(np.float32) - mu) / (sigma + 1e-6)) ** 4)
    )
    # Empirically scaled to roughly match pyiqa NIQE range [1..15]
    niqe_approx = max(1.0, min(15.0, kurtosis_approx * 0.4 + 1.0))
    return float(round(niqe_approx, 4))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ExpertIQAEvaluator:
    """
    Stateless evaluator – all methods are class-level.
    Uses lightweight no-network heuristics by default. Optional pyiqa metrics
    can be enabled with environment variable SATA_ENABLE_PYIQA=1.
    """

    @classmethod
    def extract_metrics(cls, image_path: str) -> Dict[str, float]:
        """
        Extract all IQA metrics for the given satellite image.
        Returns a dict with baseline keys:
          width, height, niqe_score, blur_variance, noise_level, haze_density
        Optional keys when SATA_ENABLE_PYIQA=1:
          clipiqa_plus, topiq_nr
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        metrics: Dict[str, float] = {
            "width":         float(w),
            "height":        float(h),
            "blur_variance": _blur_variance(gray),
            "noise_level":   _noise_level(gray),
            "haze_density":  _haze_density(bgr),
        }

        # Default path: lightweight/no-network metrics only.
        metrics["niqe_score"] = _niqe_approx(gray)

        # Optional advanced IQA (off by default to avoid network/model fetch).
        if os.environ.get("SATA_ENABLE_PYIQA", "0") == "1":
            niqe_score = None
            clipiqa_plus = None
            topiq_nr = None
            try:
                import torch
                import pyiqa
                from PIL import Image
                import torchvision.transforms as T

                device = "cuda" if torch.cuda.is_available() else "cpu"
                img_pil = Image.open(image_path).convert("RGB")
                tensor = T.ToTensor()(img_pil).unsqueeze(0).to(device)

                # Normalise resolution (match 4KAgent logic)
                max_side = max(h, w)
                if max_side <= 240:
                    tensor = torch.nn.functional.interpolate(
                        tensor, scale_factor=2, mode="bicubic", align_corners=False
                    ).clamp(0, 1)
                elif max_side > 4200:
                    tensor = torch.nn.functional.interpolate(
                        tensor, scale_factor=0.5, mode="bicubic", align_corners=False
                    ).clamp(0, 1)

                for metric_name, metric_key in [
                    ("niqe", "niqe"),
                    ("clipiqa+", "clipiqa+"),
                    ("topiq_nr", "topiq_nr"),
                ]:
                    try:
                        model = pyiqa.create_metric(metric_key, device=device)
                        score = round(model(tensor).item(), 4)
                        if metric_name == "niqe":
                            niqe_score = score
                        elif metric_name == "clipiqa+":
                            clipiqa_plus = score
                        elif metric_name == "topiq_nr":
                            topiq_nr = score
                        del model
                    except Exception:
                        pass

                del tensor
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except ImportError:
                warnings.warn(
                    "pyiqa not available. Using heuristic NIQE. "
                    "Install with: pip install pyiqa"
                )

            metrics["niqe_score"] = (
                niqe_score if niqe_score is not None else metrics["niqe_score"]
            )
            metrics["clipiqa_plus"] = clipiqa_plus if clipiqa_plus is not None else -1.0
            metrics["topiq_nr"] = topiq_nr if topiq_nr is not None else -1.0

        return metrics

    @classmethod
    def format_for_prompt(cls, metrics: Dict[str, float]) -> str:
        """Format metrics dict for injection into VLM prompts."""
        lines = []
        if metrics.get("niqe_score", -1) >= 0:
            lines.append(f"NIQE: {metrics['niqe_score']} (lower=better)")
        if metrics.get("clipiqa_plus", -1) >= 0:
            lines.append(f"CLIPIQA+: {metrics['clipiqa_plus']} (higher=better)")
        if metrics.get("topiq_nr", -1) >= 0:
            lines.append(f"TOPIQ_NR: {metrics['topiq_nr']} (higher=better)")
        lines.append(f"Blur variance: {metrics.get('blur_variance', 0):.1f}")
        lines.append(f"Noise level: {metrics.get('noise_level', 0):.1f}")
        lines.append(f"Haze density (DCP): {metrics.get('haze_density', 0):.1f}")
        lines.append(f"Resolution: {int(metrics.get('width',0))}×{int(metrics.get('height',0))}")
        return "\n".join(lines)
