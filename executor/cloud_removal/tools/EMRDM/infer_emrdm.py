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


IMG_SET = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _repo(script_dir: Path) -> Path:
    for p in [
        script_dir.parent / "SPAGAN" / "SpA-GAN_for_cloud_removal",
        script_dir / "SpA-GAN_for_cloud_removal",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("SPAGAN dependency not found for EMRDM wrapper.")


def _input(args: argparse.Namespace) -> Path:
    seq = [args.input_path, args.input, args.input_dir, args.data_dir, args.folder_lq]
    for it in seq:
        if not it:
            continue
        p = Path(it)
        if p.is_file():
            return p
        if p.is_dir():
            imgs = sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_SET)
            if imgs:
                return imgs[0]
    raise FileNotFoundError("No input for EMRDM.")


def _output(args: argparse.Namespace, inp: Path) -> Path:
    if args.output_path:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    for d in [args.output_dir, args.save_dir, args.result_dir]:
        if d:
            pd = Path(d)
            pd.mkdir(parents=True, exist_ok=True)
            return pd / inp.name
    p = Path("/tmp") / f"{inp.stem}_emrdm.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _weights(args: argparse.Namespace, repo_root: Path) -> Path:
    for opt in [args.weights, args.pretrained]:
        if opt and Path(opt).exists():
            return Path(opt)
    # EMRDM path defaults to RICE1
    p = repo_root / "pretrained_models" / "RICE1" / "gen_model_epoch_200.pth"
    if p.exists():
        return p
    p = repo_root / "pretrained_models" / "RICE2" / "gen_model_epoch_200.pth"
    if p.exists():
        return p
    raise FileNotFoundError("No checkpoint available for EMRDM wrapper.")


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

    script_dir = Path(__file__).resolve().parent
    repo_root = _repo(script_dir)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from models.gen.SPANet import Generator  # pylint: disable=import-outside-toplevel

    inp = _input(args)
    out = _output(args, inp)
    ckpt = _weights(args, repo_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(gpu_ids=[0] if device.type == "cuda" else [])
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"), strict=True)
    model = model.to(device).eval()

    x = transforms.ToTensor()(Image.open(str(inp)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        _, y = model(x)
        pred = y[0].detach().cpu().numpy()

    pred = np.clip(pred[:3], 0.0, 1.0).transpose(1, 2, 0)
    bgr = cv2.cvtColor((pred * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), bgr):
        raise RuntimeError(f"Could not write output: {out}")
    print(f"[emrdm] restored image -> {out}")


if __name__ == "__main__":
    main()
