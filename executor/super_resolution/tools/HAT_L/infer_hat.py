import os
import sys
import argparse
import types
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

# Respect externally selected GPU; only default to 0 if unset.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Compatibility shim:
# Some basicsr versions import torchvision.transforms.functional_tensor,
# which was removed/relocated in newer torchvision.
try:
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        _ft_mod = types.ModuleType("torchvision.transforms.functional_tensor")
        if hasattr(F, "rgb_to_grayscale"):
            _ft_mod.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = _ft_mod
except Exception:
    pass

# Add HAT repo to path
current_dir = os.path.dirname(os.path.abspath(__file__))
hat_dir = os.path.join(current_dir, "HAT")
if hat_dir not in sys.path:
    sys.path.insert(0, hat_dir)

from hat.archs.hat_arch import HAT


def _pick_device(min_free_mb: int = 2048) -> torch.device:
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
    parser = argparse.ArgumentParser(description='Auto-Mapping HAT Satellite SR')
    parser.add_argument('--input_path', type=str, required=True, help='Path to LR image')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save SR result')
    # The Perception Agent only needs to provide the factor (2, 3, or 4)
    parser.add_argument('--upscale', type=int, default=4, choices=[2, 3, 4], help='Upscale factor')
    # Optional: allow overriding the model style
    parser.add_argument('--mode', type=str, default='fidelity', choices=['fidelity', 'perception'], 
                        help='fidelity (ImageNet) or perception (GAN)')
    return parser.parse_args()

def get_auto_configs(upscale, mode):
    """Maps the upscale factor to the specific weights and architecture."""
    base_weight_dir = os.path.join(current_dir, "HAT/experiments/pretrained_models")
    
    # Mapping logic based on your downloaded files
    if mode == 'perception' and upscale == 4:
        weight_file = "Real_HAT_GAN_SRx4.pth"
    else:
        # Default to the ImageNet-pretrain weights for x2, x3, x4
        weight_file = f"HAT-L_SRx{upscale}_ImageNet-pretrain.pth"
    
    weights_path = os.path.join(base_weight_dir, weight_file)
    
    # Standard HAT-L Configuration
    config = {
        "upscale": upscale, "in_chans": 3, "img_size": 64, "window_size": 16,
        "compress_ratio": 3, "squeeze_factor": 30, "conv_scale": 0.01, "overlap_ratio": 0.5,
        "img_range": 1., "depths": [6, 6, 6, 6, 6, 6], "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6], "mlp_ratio": 2, "upsampler": 'pixelshuffle', "resi_connection": '1conv'
    }
    
    return config, weights_path

def run_sr():
    args = parse_args()
    device = _pick_device()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    # 1. AUTO-RESOLVE WEIGHTS AND CONFIG
    config, weights_path = get_auto_configs(args.upscale, args.mode)
    
    if not os.path.exists(weights_path):
        print(f"[!] FATAL: Weights not found at {weights_path}")
        sys.exit(1)

    print(f"[+] Initializing HAT x{args.upscale} ({args.mode}) on {device}...")
    model = HAT(
        upscale=args.upscale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.,
        # THE FIX: Increase depths to [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6] (12 groups)
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], 
        embed_dim=180,
        # THE FIX: Match num_heads to the 12 groups
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    # 2. LOAD WEIGHTS
    try:
        loadnet = torch.load(weights_path, map_location=device)
        keyname = 'params_ema' if 'params_ema' in loadnet else ('params' if 'params' in loadnet else None)
        state_dict = loadnet[keyname] if keyname else loadnet
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[!] Load Error: {e}")
        sys.exit(1)
        
    model.eval()

    # 3. PRE-PROCESS
    img_pil = Image.open(args.input_path).convert('RGB')
    w_orig, h_orig = img_pil.size
    
    pad_factor = 16
    h_pad = (pad_factor - h_orig % pad_factor) % pad_factor
    w_pad = (pad_factor - w_orig % pad_factor) % pad_factor
    
    img_tensor = F.to_tensor(img_pil)
    img_tensor = F.pad(img_tensor, (0, 0, w_pad, h_pad), padding_mode='reflect')
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 4. INFERENCE
    print(f"[*] Processing {w_orig}x{h_orig} -> {w_orig*args.upscale}x{h_orig*args.upscale}...")
    with torch.no_grad():
        output = model(img_tensor)
        output = output.clamp(0, 1).squeeze(0).cpu().numpy()
        output = output[:, :h_orig * args.upscale, :w_orig * args.upscale]
        
        out_img = (output.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(args.output_path, out_bgr)
        if not ok:
            print(f"[!] Failed to write output image: {args.output_path}")
            sys.exit(1)

    print(f"[SUCCESS] Saved SR result to: {args.output_path}")

if __name__ == "__main__":
    run_sr()
