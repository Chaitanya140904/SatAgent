#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _first_image_in_dir(folder: Path) -> Optional[Path]:
    if not folder.is_dir():
        return None
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXT and p.is_file())
    return imgs[0] if imgs else None


def _pick_input(ns: argparse.Namespace) -> Path:
    direct_candidates: Iterable[Optional[str]] = (
        getattr(ns, "input_path", None),
        getattr(ns, "input", None),
        getattr(ns, "ms_path", None),
        getattr(ns, "pan_path", None),
        getattr(ns, "input_image", None),
    )
    for item in direct_candidates:
        if not item:
            continue
        p = Path(item)
        if p.is_file():
            return p
        if p.is_dir():
            hit = _first_image_in_dir(p)
            if hit is not None:
                return hit

    dir_candidates: Iterable[Optional[str]] = (
        getattr(ns, "input_dir", None),
        getattr(ns, "data_dir", None),
        getattr(ns, "folder_lq", None),
    )
    for item in dir_candidates:
        if not item:
            continue
        hit = _first_image_in_dir(Path(item))
        if hit is not None:
            return hit

    raise FileNotFoundError("No usable input image found in provided arguments.")


def _pick_output(ns: argparse.Namespace, inp: Path) -> Path:
    out_file = getattr(ns, "output_path", None)
    if out_file:
        p = Path(out_file)
        if p.suffix.lower() in SUPPORTED_EXT:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        p.mkdir(parents=True, exist_ok=True)
        return p / inp.name

    for maybe_dir in (
        getattr(ns, "output_dir", None),
        getattr(ns, "save_dir", None),
        getattr(ns, "result_dir", None),
        getattr(ns, "output", None),
    ):
        if maybe_dir:
            d = Path(maybe_dir)
            d.mkdir(parents=True, exist_ok=True)
            return d / inp.name

    fallback = Path("/tmp") / f"{inp.stem}_restored.png"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def _local_contrast_enhance(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _cloud_cleanup(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Bright + low-saturation regions are likely thick cloud cover.
    cloud = ((val > 205) & (sat < 70)).astype(np.uint8) * 255
    cloud = cv2.morphologyEx(cloud, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cloud = cv2.dilate(cloud, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.inpaint(img_bgr, cloud, 3, cv2.INPAINT_TELEA)
    return _local_contrast_enhance(cleaned)


def _shadow_lift(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    value = hsv[:, :, 2]
    cutoff = np.percentile(value, 25.0)
    lift_mask = (value < cutoff).astype(np.float32)
    value = np.clip(value + 42.0 * lift_mask, 0, 255)
    hsv[:, :, 2] = value
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _local_contrast_enhance(out)


def _denoise(img_bgr: np.ndarray) -> np.ndarray:
    d1 = cv2.fastNlMeansDenoisingColored(img_bgr, None, 7, 7, 7, 19)
    d2 = cv2.bilateralFilter(d1, d=5, sigmaColor=35, sigmaSpace=35)
    return d2


def _deblur(img_bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 1.1)
    sharp = cv2.addWeighted(img_bgr, 1.75, blur, -0.75, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _destripe(img_bgr: np.ndarray) -> np.ndarray:
    med_h = cv2.medianBlur(img_bgr, 3)
    med_v = cv2.transpose(cv2.medianBlur(cv2.transpose(img_bgr), 3))
    out = cv2.addWeighted(med_h, 0.6, med_v, 0.4, 0)
    return out


def _remove_jpeg_artifacts(img_bgr: np.ndarray) -> np.ndarray:
    base = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=45, sigmaSpace=45)
    return cv2.fastNlMeansDenoisingColored(base, None, 4, 4, 7, 15)


def _pan_sharpen_like(img_bgr: np.ndarray) -> np.ndarray:
    detail = cv2.detailEnhance(img_bgr, sigma_s=12, sigma_r=0.18)
    return np.clip(detail, 0, 255).astype(np.uint8)


def _temporal_like(img_bgr: np.ndarray) -> np.ndarray:
    den = _denoise(img_bgr)
    fused = cv2.addWeighted(den, 1.45, cv2.GaussianBlur(den, (0, 0), 0.9), -0.45, 0)
    return np.clip(fused, 0, 255).astype(np.uint8)


def _super_resolve(img_bgr: np.ndarray, factor: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    f = max(1, int(factor))
    up = cv2.resize(img_bgr, (w * f, h * f), interpolation=cv2.INTER_LANCZOS4)
    return _deblur(up)


def _mask_for_local_tasks(img_bgr: np.ndarray, task: str) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if task == "cloud_removal":
        _, mask = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
    elif task == "shadow_correction":
        _, mask = cv2.threshold(gray, int(np.percentile(gray, 25.0)), 255, cv2.THRESH_BINARY_INV)
    elif task == "destriping":
        grad = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        g = cv2.convertScaleAbs(grad)
        _, mask = cv2.threshold(g, 18, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _blend_local(input_img: np.ndarray, restored_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.float32) / 255.0)[:, :, None]
    return np.clip(restored_img.astype(np.float32) * m + input_img.astype(np.float32) * (1.0 - m), 0, 255).astype(np.uint8)


def apply_task(task: str, img_bgr: np.ndarray, scale: int = 4, local_path: bool = False) -> np.ndarray:
    if task == "roi_masking":
        return cv2.cvtColor(_mask_for_local_tasks(img_bgr, task), cv2.COLOR_GRAY2BGR)

    if task == "cloud_removal":
        enhanced = _cloud_cleanup(img_bgr)
    elif task == "dehazing":
        enhanced = _local_contrast_enhance(img_bgr)
    elif task == "shadow_correction":
        enhanced = _shadow_lift(img_bgr)
    elif task == "denoising":
        enhanced = _denoise(img_bgr)
    elif task == "deblurring":
        enhanced = _deblur(img_bgr)
    elif task == "destriping":
        enhanced = _destripe(img_bgr)
    elif task == "artifact_removal":
        enhanced = _remove_jpeg_artifacts(img_bgr)
    elif task == "pan_sharpen":
        enhanced = _pan_sharpen_like(img_bgr)
    elif task == "temporal_fusion":
        enhanced = _temporal_like(img_bgr)
    elif task == "super_resolution":
        enhanced = _super_resolve(img_bgr, factor=scale)
    else:
        enhanced = img_bgr.copy()

    if local_path and task in {"cloud_removal", "shadow_correction", "destriping"}:
        mask = _mask_for_local_tasks(img_bgr, task)
        return _blend_local(img_bgr, enhanced, mask)

    return enhanced


def execute_from_cli(task: str, model_id: str = "", default_scale: int = 4, local_path: bool = False) -> Path:
    parser = argparse.ArgumentParser(description="SatAgent execution adapter")
    parser.add_argument("--input_path")
    parser.add_argument("--input")
    parser.add_argument("--input_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--folder_lq")
    parser.add_argument("--input_image")
    parser.add_argument("--ms_path")
    parser.add_argument("--pan_path")
    parser.add_argument("--output_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--output")
    parser.add_argument("--save_dir")
    parser.add_argument("--result_dir")
    parser.add_argument("--scale", type=int, default=default_scale)
    parser.add_argument("--upscale", type=float, default=float(default_scale))
    parser.add_argument("-i", dest="short_in")
    parser.add_argument("-o", dest="short_out")
    ns, _extras = parser.parse_known_args()

    if not ns.input_path and ns.short_in:
        ns.input_path = ns.short_in
    if not ns.output_path and ns.short_out:
        ns.output_path = ns.short_out

    src = _pick_input(ns)
    dst = _pick_output(ns, src)

    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read input image: {src}")

    scale = int(ns.scale or ns.upscale or default_scale)
    out = apply_task(task=task, img_bgr=image, scale=scale, local_path=local_path)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), out)
    if not ok:
        raise RuntimeError(f"Unable to write output image: {dst}")

    print(f"[{task}:{model_id}] {src} -> {dst}")
    return dst
