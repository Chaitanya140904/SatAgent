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


def _find_spagan_root(script_dir: Path) -> Path:
    candidates = [
        script_dir.parent / "SPAGAN" / "SpA-GAN_for_cloud_removal",
        script_dir / "SpA-GAN_for_cloud_removal",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("SpA-GAN_for_cloud_removal repo not found near DVPNet tool.")


def _pick_input(args: argparse.Namespace) -> Path:
    for item in [args.input_path, args.input, args.input_dir, args.data_dir, args.folder_lq]:
        if not item:
            continue
        p = Path(item)
        if p.is_file():
            return p
        if p.is_dir():
            imgs = sorted(x for x in p.iterdir() if x.suffix.lower() in VALID_EXTS)
            if imgs:
                return imgs[0]
    raise FileNotFoundError("No input image was found for DVPNet wrapper.")


def _pick_output(args: argparse.Namespace, input_path: Path) -> Path:
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    for d in [args.output_dir, args.save_dir, args.result_dir]:
        if d:
            out_dir = Path(d)
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / input_path.name
    out = Path("/tmp") / f"{input_path.stem}_cloudfree.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _pick_weights(args: argparse.Namespace, spagan_root: Path) -> Path:
    if args.weights:
        p = Path(args.weights)
        if p.exists():
            return p
    if args.pretrained:
        p = Path(args.pretrained)
        if p.exists():
            return p
    default = spagan_root / "pretrained_models" / "RICE1" / "gen_model_epoch_200.pth"
    if default.exists():
        return default
    raise FileNotFoundError("DVPNet wrapper could not find a SPAGAN checkpoint.")


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
    spagan_root = _find_spagan_root(script_dir)
    if str(spagan_root) not in sys.path:
        sys.path.insert(0, str(spagan_root))

    from models.gen.SPANet import Generator  # pylint: disable=import-outside-toplevel

    input_path = _pick_input(args)
    output_path = _pick_output(args, input_path)
    weights_path = _pick_weights(args, spagan_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(gpu_ids=[0] if device.type == "cuda" else [])
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    image = Image.open(str(input_path)).convert("RGB")
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, out = model(x)
        pred = out[0].detach().cpu().numpy()

    pred = np.clip(pred[:3], 0.0, 1.0).transpose(1, 2, 0)
    pred = (pred * 255.0).astype(np.uint8)
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), pred_bgr):
        raise RuntimeError(f"Failed writing output image: {output_path}")

    print(f"[dvpnet] restored image -> {output_path}")


if __name__ == "__main__":
    main()
