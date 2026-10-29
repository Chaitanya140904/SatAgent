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

# LOCKDOWN: GPU 1 Only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Setup architectural imports
current_dir = os.path.dirname(os.path.abspath(__file__))
arch_path = os.path.join(current_dir, "architectures", "SCUNet")
if arch_path not in sys.path:
    sys.path.insert(0, arch_path)

# Import the SCUNet architecture
from models.network_scunet import SCUNet as net


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
    parser = argparse.ArgumentParser(description="SCUNet Satellite Denoising")
    parser.add_argument('--input_path', type=str, required=True, help="Path to noisy image")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save result")
    # Setting the default weights path to the .pth file as per your setup
    parser.add_argument('--weights', type=str, default="./scunet_color.pth", help="Path to weights")
    return parser.parse_args()

def run_scunet():
    args = parse_args()
    device = _pick_device()
    use_half = device.type == "cuda"

    # --- 1: LOAD AND PRE-PROCESS IMAGE ---
    print(f"[+] Loading noisy image: {args.input_path}")
    img_pil = Image.open(args.input_path).convert('RGB')
    w_orig, h_orig = img_pil.size
    
    # SCUNet requires dimensions divisible by 8
    pad_factor = 8
    h_pad = (pad_factor - h_orig % pad_factor) % pad_factor
    w_pad = (pad_factor - w_orig % pad_factor) % pad_factor
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img_pil)
    # Apply padding: (left, top, right, bottom)
    img_tensor = F.pad(img_tensor, (0, 0, w_pad, h_pad), padding_mode='reflect')
    input_tensor = img_tensor.unsqueeze(0).to(device)
    if use_half:
        input_tensor = input_tensor.half()

    # --- 2: BUILD ARCHITECTURE ---
    print(f"[+] Initializing SCUNet architecture on {device}...")
    # Using the config provided in your setup: [4, 4, 4, 4, 4, 4, 4] with dim=64
    model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64).to(device)
    if use_half:
        model = model.half()

    # --- 3: LOAD WEIGHTS (Handling SafeTensors conversion) ---
    safe_path = args.weights.replace('.pth', '.safetensors')
    
    if not os.path.exists(safe_path):
        print("[!] SafeTensors not found. Loading from .pth...")
        state_dict = torch.load(args.weights, map_location='cpu')
        if 'params' in state_dict: state_dict = state_dict['params']
        elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        # Clean module prefix
        state_dict = {k.replace('module.', ''): v.contiguous() for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        print(f"[+] Streaming weights from {safe_path}...")
        weights = load_file(safe_path, device=str(device))
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
    del model, input_tensor, output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    run_scunet()
