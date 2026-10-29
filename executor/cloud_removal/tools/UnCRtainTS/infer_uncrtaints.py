#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _locate_spagan(repo_dir: Path) -> Path:
    cands = [
        repo_dir.parent / "SPAGAN" / "SpA-GAN_for_cloud_removal",
        repo_dir / "SpA-GAN_for_cloud_removal",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("SPAGAN repo not found for UnCRtainTS wrapper.")


def _get_input(args: argparse.Namespace) -> Path:
    for val in [args.input_path, args.input, args.input_dir, args.data_dir, args.folder_lq]:
        if not val:
            continue
        p = Path(val)
        if p.is_file():
            return p
        if p.is_dir():
            imgs = sorted(x for x in p.iterdir() if x.suffix.lower() in VALID_EXTS)
            if imgs:
                return imgs[0]
    raise FileNotFoundError("No input image for UnCRtainTS.")


def _get_output(args: argparse.Namespace, inp: Path) -> Path:
    if args.output_path:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    for d in [args.output_dir, args.save_dir, args.result_dir]:
        if d:
            out_dir = Path(d)
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / inp.name
    out = Path("/tmp") / f"{inp.stem}_uncrtaints.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _get_weights(args: argparse.Namespace, spagan_root: Path) -> Path:
    for w in [args.weights, args.pretrained]:
        if w and Path(w).exists():
            return Path(w)
    # Prefer RICE2 for this path.
    p = spagan_root / "pretrained_models" / "RICE2" / "gen_model_epoch_200.pth"
    if p.exists():
        return p
    p = spagan_root / "pretrained_models" / "RICE1" / "gen_model_epoch_200.pth"
    if p.exists():
        return p
    raise FileNotFoundError("No SPAGAN checkpoint available.")


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--input_path")
    ap.add_argument("--input")
    ap.add_argument("--input_dir")
    ap.add_argument("--data_dir")
    ap.add_argument("--folder_lq")
    ap.add_argument("--output_path")
    ap.add_argument("--output_dir")
    ap.add_argument("--save_dir")
    ap.add_argument("--result_dir")
    ap.add_argument("--weights")
    ap.add_argument("--pretrained")
    ap.add_argument("-i", dest="input_short")
    ap.add_argument("-o", dest="output_short")
    args, _ = ap.parse_known_args()

    if not args.input_path and args.input_short:
        args.input_path = args.input_short
    if not args.output_path and args.output_short:
        args.output_path = args.output_short

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    script_dir = Path(__file__).resolve().parent
    spagan_root = _locate_spagan(script_dir)
    if str(spagan_root) not in sys.path:
        sys.path.insert(0, str(spagan_root))

    from models.gen.SPANet import Generator  # pylint: disable=import-outside-toplevel

    inp = _get_input(args)
    out = _get_output(args, inp)
    ckpt = _get_weights(args, spagan_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(gpu_ids=[0] if device.type == "cuda" else [])
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"), strict=True)
    model = model.to(device).eval()

    x = transforms.ToTensor()(Image.open(str(inp)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        _, y = model(x)
        arr = y[0].detach().cpu().numpy()

    arr = np.clip(arr[:3], 0.0, 1.0).transpose(1, 2, 0)
    bgr = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), bgr):
        raise RuntimeError(f"Failed to write output: {out}")
    print(f"[uncrtaints] restored image -> {out}")


if __name__ == "__main__":
    main()
