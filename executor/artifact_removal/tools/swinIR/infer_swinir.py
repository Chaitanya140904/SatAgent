import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from collections import OrderedDict

# LOCKDOWN: GPU 1 Only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add SwinIR repo to path to access define_model and setup
# New version: Points specifically to the SwinIR source directory
current_dir = os.path.dirname(os.path.abspath(__file__))
swinir_src = os.path.join(current_dir, 'SwinIR') # Matches the blue highlighted folder in your image

if swinir_src not in sys.path:
    sys.path.insert(0, swinir_src)

from main_test_swinir import define_model, setup

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _pick_input_path(args) -> str:
    for cand in (args.input_path,):
        if cand and Path(cand).is_file():
            return str(Path(cand))
    for d in (args.input_dir, args.folder_lq):
        if not d:
            continue
        p = Path(d)
        if p.is_file():
            return str(p)
        if p.is_dir():
            imgs = sorted([x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS])
            if imgs:
                return str(imgs[0])
    raise FileNotFoundError("No valid input image found for SwinIR.")


def _pick_output_path(args, input_path: str) -> str:
    if args.output_path:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    out_dir = args.output_dir or args.save_dir
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return str(p / Path(input_path).name)
    return str(Path("./output.png").resolve())


def _resolve_weights(args) -> str:
    chosen = args.weights or args.model_path
    if not chosen:
        chosen = "./SwinIR/experiments/pretrained_models/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth"
    p = Path(chosen)
    if p.exists():
        return str(p.resolve())
    if not p.is_absolute():
        p2 = (Path(current_dir) / p).resolve()
        if p2.exists():
            return str(p2)
    # Common wrapper-provided path points to model_zoo; remap to bundled weights.
    fname = p.name
    p3 = (Path(current_dir) / "SwinIR" / "experiments" / "pretrained_models" / fname).resolve()
    if p3.exists():
        return str(p3)
    return str(p.resolve() if p.is_absolute() else (Path(current_dir) / p).resolve())


def run_swinir_car(input_path, output_path, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class SatelliteArgs:
        task = 'jpeg_car' # Matches your 006_CAR weights
        scale = 1
        jpeg = 40
        training_patch_size = 126
        large_model = False
        model_path = weights_path
        folder_lq = None
        folder_gt = None

    args = SatelliteArgs()
    model = define_model(args)
    model.eval()
    model = model.to(device)
    _, _, _, window_size = setup(args)

    # --- 3: COLOR-PRESERVING PRE-PROCESSING ---
    img = cv2.imread(input_path)
    if img is None: raise FileNotFoundError(f"Missing: {input_path}")
    
    # Convert to YCbCr to isolate Luminance (Y)
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_y = img_ycbcr[:, :, 0].astype(np.float32) / 255.
    
    # Prepare tensor [1, 1, H, W]
    img_tensor = torch.from_numpy(img_y).float().unsqueeze(0).unsqueeze(0).to(device)

    # --- 4: INFERENCE ---
    with torch.no_grad():
        _, _, h_old, w_old = img_tensor.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :w_old + w_pad]
        
        output = model(img_tensor)
        output = output[..., :h_old, :w_old]

    # --- 5: RECOMBINE WITH ORIGINAL COLOR ---
    output_y = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_y = (output_y * 255.0).round().astype(np.uint8)
    
    # Put the enhanced Y back into the original YCbCr image
    img_ycbcr[:, :, 0] = output_y
    output_bgr = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2BGR)
    
    cv2.imwrite(output_path, output_bgr)
    print(f"[SUCCESS] Saved color-preserved image to: {output_path}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--folder_lq', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    args, _ = parser.parse_known_args()

    inp = _pick_input_path(args)
    out = _pick_output_path(args, inp)
    wts = _resolve_weights(args)
    run_swinir_car(inp, out, wts)
