from __future__ import annotations
"""
SatAgent utils – Satellite-domain IQA evaluation using pyiqa.
Adapted from 4KAgent's expert_IQA_eval.py for satellite imagery.

Metrics used:
  - CLIPIQA+  : higher is better
  - TOPIQ_NR  : higher is better
  - MUSIQ     : higher is better
  - NIQE      : lower is better
"""
import gc
from typing import Dict, Optional, Tuple

import torch
from PIL import Image
import torchvision.transforms as transforms

_AVAILABLE_METRICS = {
    "CLIPIQA+": "clipiqa+",
    "TOPIQ_NR":  "topiq_nr",
    "MUSIQ":     "musiq",
    "NIQE":      "niqe",
}

_TARGET_METRICS = {
    "CLIPIQA+": "clipiqa+",
    "MANIQA":   "maniqa",
    "MUSIQ":    "musiq",
    "NIQE":     "niqe",
}

_REFLECTION_WEIGHTS = {
    "CLIPIQA+": 1.0,
    "MANIQA":   1.0,
    "MUSIQ":    0.01,
    "NIQE":     1.0,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    t = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return t


def _normalise_resolution(tensor: torch.Tensor) -> torch.Tensor:
    """Scale tensor so max side is in range [240, 4200]."""
    _, _, h, w = tensor.size()
    max_side = max(h, w)
    if max_side <= 120:
        tensor = torch.nn.functional.interpolate(tensor, scale_factor=4,
                                                 mode='bicubic', align_corners=False).clamp(0, 1)
    elif max_side <= 240:
        tensor = torch.nn.functional.interpolate(tensor, scale_factor=2,
                                                 mode='bicubic', align_corners=False).clamp(0, 1)
    elif max_side > 4200:
        tensor = torch.nn.functional.interpolate(tensor, scale_factor=0.5,
                                                 mode='bicubic', align_corners=False).clamp(0, 1)
    return tensor


def compute_iqa(image_path: str) -> Tuple[str, int, int]:
    """
    Compute perception IQA metrics (CLIPIQA+, TOPIQ_NR, MUSIQ, NIQE).
    Returns a formatted string, image height, image width.
    """
    try:
        import pyiqa
    except ImportError:
        raise ImportError("pyiqa is required. Install with: pip install pyiqa")

    image_tensor = _to_tensor(image_path)
    h, w = image_tensor.size(2), image_tensor.size(3)
    image_tensor = _normalise_resolution(image_tensor)

    results: Dict[str, float] = {}
    for metric_name, metric_key in _AVAILABLE_METRICS.items():
        try:
            iqa_model = pyiqa.create_metric(metric_key, device=device)
            iqa_model.to(device)
            score = iqa_model(image_tensor).item()
            results[metric_name] = round(score, 4)
            del iqa_model
        except Exception as e:
            results[metric_name] = -1.0
        gc.collect()
        torch.cuda.empty_cache()

    del image_tensor
    torch.cuda.empty_cache()

    result_str = "\n".join(f"{m}: {s}" for m, s in results.items())
    return result_str, h, w


def compute_iqa_metric_score(image_path: str) -> float:
    """
    Compute a single weighted quality score for reflection / rollback decisions.
    Returns a float in roughly [0, 1] (higher = better).
    """
    try:
        import pyiqa
    except ImportError:
        return 0.5

    image_tensor = _to_tensor(image_path)
    image_tensor = _normalise_resolution(image_tensor)

    results: Dict[str, float] = {}
    for metric_name, metric_key in _TARGET_METRICS.items():
        try:
            iqa_model = pyiqa.create_metric(metric_key, device=device)
            iqa_model.to(device)
            results[metric_name] = round(iqa_model(image_tensor).item(), 4)
            del iqa_model
        except Exception:
            results[metric_name] = 0.5
        gc.collect()
        torch.cuda.empty_cache()

    del image_tensor
    torch.cuda.empty_cache()

    weighted = sum(
        results[m] * _REFLECTION_WEIGHTS[m]
        if m != "NIQE"
        else (1 - results[m] / 10) * _REFLECTION_WEIGHTS[m]
        for m in results
    )
    return round(weighted / len(results), 4)


def compute_iqa_metric_score_batch(image_paths: list) -> list:
    """
    Compute quality scores for a batch of images, reusing IQA models.
    """
    try:
        import pyiqa
    except ImportError:
        return [0.5] * len(image_paths)

    models = {
        name: pyiqa.create_metric(key, device=device).to(device)
        for name, key in _TARGET_METRICS.items()
    }

    scores = []
    for image_path in image_paths:
        try:
            t = _to_tensor(image_path)
            t = _normalise_resolution(t)

            results = {
                name: round(model(t).item(), 4)
                for name, model in models.items()
            }
            weighted = sum(
                results[m] * _REFLECTION_WEIGHTS[m]
                if m != "NIQE"
                else (1 - results[m] / 10) * _REFLECTION_WEIGHTS[m]
                for m in results
            )
            scores.append(round(weighted / len(results), 4))
            del t
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[IQA] Warning – failed for {image_path}: {e}")
            scores.append(None)

    for model in models.values():
        del model
    gc.collect()
    torch.cuda.empty_cache()
    return scores
