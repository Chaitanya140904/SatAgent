import os
import sys
import argparse
import torch
import gc
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from safetensors.torch import load_file

import sys, os, types

current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
basicsr_path = os.path.join(current_dir, "basicsr")
models_path = os.path.join(basicsr_path, "models")
archs_path = os.path.join(models_path, "archs")

# Create fake package structure
basicsr = types.ModuleType("basicsr")
models = types.ModuleType("basicsr.models")
archs = types.ModuleType("basicsr.models.archs")

# Set __path__ so Python treats them as packages
basicsr.__path__ = [basicsr_path]
models.__path__ = [models_path]
archs.__path__ = [archs_path]

# Register in sys.modules
sys.modules["basicsr"] = basicsr
sys.modules["basicsr.models"] = models
sys.modules["basicsr.models.archs"] = archs

# Now add archs path for direct import
sys.path.insert(0, archs_path)

from NAFNet_arch import NAFNet


def _pick_device(min_free_mb: int = 1536) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        free_b, _total_b = torch.cuda.mem_get_info()
        free_mb = free_b / (1024 * 1024)
        if free_mb < min_free_mb:
            print(f"[i] GPU free memory low ({free_mb:.0f} MiB). Falling back to CPU.")
            return torch.device("cpu")
    except Exception:
        pass
    return torch.device("cuda")

def parse_args():
    parser = argparse.ArgumentParser(description="NAFNet Satellite Denoising")
    parser.add_argument('--input_path', type=str, required=True, help="Path to noisy image")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save result")
    # Setting the default weights path
    parser.add_argument('--weights', type=str, default="./nafnet_denoise.pth", help="Path to weights")
    return parser.parse_args()

def run_denoiser():
    args = parse_args()
    device = _pick_device()
    use_half = device.type == "cuda"

    # --- 1: LOAD AND PRE-PROCESS IMAGE ---
    print(f"[+] Loading noisy image: {args.input_path}")
    img_pil = Image.open(args.input_path).convert('RGB')
    w_orig, h_orig = img_pil.size
    
    # NAFNet requires dimensions divisible by 32
    pad_factor = 32
    h_pad = (pad_factor - h_orig % pad_factor) % pad_factor
    w_pad = (pad_factor - w_orig % pad_factor) % pad_factor
    
    # Transform: ToTensor handles [0, 1] range
    img_tensor = transforms.ToTensor()(img_pil)
    # Apply padding: (left, top, right, bottom)
    img_tensor = F.pad(img_tensor, (0, 0, w_pad, h_pad), padding_mode='reflect')
    input_tensor = img_tensor.unsqueeze(0).to(device)
    if use_half:
        input_tensor = input_tensor.half()

    # --- 2: BUILD ARCHITECTURE ---
    print(f"[+] Initializing NAFNet architecture on {device}...")
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]).to(device)
    if use_half:
        model = model.half()

    # --- 3: LOAD WEIGHTS ---
    print(f"[+] Loading weights from {args.weights}...")
    if args.weights.endswith('.safetensors'):
        weights = load_file(args.weights, device=device)
    else:
        weights = torch.load(args.weights, map_location=device)
        if 'params' in weights: weights = weights['params']
        elif 'state_dict' in weights: weights = weights['state_dict']
            
    model.load_state_dict(weights)
    model.eval()

    # --- 4: INFERENCE ---
    print(f"[*] Denoising satellite patch (Padded to: {w_orig+w_pad}x{h_orig+h_pad})...")
    with torch.no_grad():
        output = model(input_tensor)
        
        # Post-process: Clamp and move to CPU
        output = torch.clamp(output, 0, 1).squeeze(0).cpu().numpy()
        
        # CROP: Remove the padding
        output = output[:, :h_orig, :w_orig]
        
        # Convert to HWC [0, 255] for saving
        out_img = (output.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(args.output_path, out_bgr)
    
    print(f"[SUCCESS] Saved output to: {args.output_path}")

    # --- 5: TOTAL PURGE ---
    del model, input_tensor, output, weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    run_denoiser()
