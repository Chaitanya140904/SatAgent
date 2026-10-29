#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def locate_input(args: argparse.Namespace) -> Path:
    for item in [args.input_path, args.input, args.input_dir, args.data_dir]:
        if not item:
            continue
        p = Path(item)
        if p.is_file():
            return p
        if p.is_dir():
            files = sorted(x for x in p.iterdir() if x.suffix.lower() in FORMATS)
            if files:
                return files[0]
    raise FileNotFoundError("No input found.")


def locate_output(args: argparse.Namespace, src: Path) -> Path:
    if args.output_path:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    for d in [args.output_dir, args.save_dir, args.result_dir]:
        if d:
            out_dir = Path(d)
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / src.name
    p = Path("/tmp") / f"{src.stem}_obb_mask.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def obb_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    OBB-like ROI proposal:
    finds salient connected components and fills rotated boxes.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)

    img_area = gray.shape[0] * gray.shape[1]
    min_area = max(64, int(0.0005 * img_area))
    max_area = int(0.5 * img_area)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.fillConvexPoly(mask, box, 255)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    if np.count_nonzero(mask) < 16:
        _, mask = cv2.threshold(gray, int(np.percentile(gray, 90)), 255, cv2.THRESH_BINARY)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_path")
    parser.add_argument("--input")
    parser.add_argument("--input_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--save_dir")
    parser.add_argument("--result_dir")
    parser.add_argument("-i", dest="input_path_short")
    parser.add_argument("-o", dest="output_path_short")
    args, _ = parser.parse_known_args()

    if not args.input_path and args.input_path_short:
        args.input_path = args.input_path_short
    if not args.output_path and args.output_path_short:
        args.output_path = args.output_path_short

    src = locate_input(args)
    dst = locate_output(args, src)

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read input: {src}")

    m = obb_mask(img)
    if not cv2.imwrite(str(dst), m):
        raise RuntimeError(f"Failed to save output: {dst}")
    print(f"[yolov8_obb] wrote mask: {dst}")


if __name__ == "__main__":
    main()
