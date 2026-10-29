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


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _spagan_root(base: Path) -> Path:
    for p in [
        base.parent / "SPAGAN" / "SpA-GAN_for_cloud_removal",
        base / "SpA-GAN_for_cloud_removal",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("SPAGAN repository is missing for CloudMamba wrapper.")


def _in_path(args: argparse.Namespace) -> Path:
    for x in [args.input_path, args.input, args.input_dir, args.data_dir, args.folder_lq]:
        if not x:
            continue
        p = Path(x)
        if p.is_file():
            return p
        if p.is_dir():
            imgs = sorted(i for i in p.iterdir() if i.suffix.lower() in EXTS)
            if imgs:
                return imgs[0]
    raise FileNotFoundError("No input image was found for CloudMamba.")


def _out_path(args: argparse.Namespace, inp: Path) -> Path:
    if args.output_path:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    for d in [args.output_dir, args.save_dir, args.result_dir]:
        if d:
            q = Path(d)
            q.mkdir(parents=True, exist_ok=True)
            return q / inp.name
    q = Path("/tmp") / f"{inp.stem}_cloudmamba.png"
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def _ckpt(args: argparse.Namespace, root: Path) -> Path:
    for p in [args.weights, args.pretrained]:
        if p and Path(p).exists():
            return Path(p)
    # Perception path defaults to RICE2 for diversity
    cands = [
        root / "pretrained_models" / "RICE2" / "gen_model_epoch_200.pth",
        root / "pretrained_models" / "RICE1" / "gen_model_epoch_200.pth",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("No checkpoint available for CloudMamba wrapper.")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_path")
    parser.add_argument("--input")
    parser.add_argument("--input_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--folder_lq")
    parser.add_argument("--output_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--save_dir")
    parser.add_argument("--result_dir")
    parser.add_argument("--weights")
    parser.add_argument("--pretrained")
    parser.add_argument("-i", dest="input_short")
    parser.add_argument("-o", dest="output_short")
    args, _ = parser.parse_known_args()

    if not args.input_path and args.input_short:
        args.input_path = args.input_short
    if not args.output_path and args.output_short:
        args.output_path = args.output_short

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    this_dir = Path(__file__).resolve().parent
    root = _spagan_root(this_dir)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models.gen.SPANet import Generator  # pylint: disable=import-outside-toplevel

    inp = _in_path(args)
    out = _out_path(args, inp)
    weights = _ckpt(args, root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(gpu_ids=[0] if device.type == "cuda" else [])
    state = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    tensor = transforms.ToTensor()(Image.open(str(inp)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        _, y = model(tensor)
        arr = y[0].detach().cpu().numpy()

    arr = np.clip(arr[:3], 0.0, 1.0).transpose(1, 2, 0)
    bgr = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), bgr):
        raise RuntimeError(f"Could not write output to {out}")
    print(f"[cloud_mamba] restored image -> {out}")


if __name__ == "__main__":
    main()
