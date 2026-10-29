import os
import sys
import gc
import argparse
import cv2
import numpy as np
import torch

# LOCKDOWN: GPU 1 Only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# BasicSR Imports
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse

def parse_args():
    parser = argparse.ArgumentParser(description='NAFNet Denoising/Deblurring Expert')
    parser.add_argument('--input_path', type=str, required=True, help='Path to degraded image')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save restored result')
    parser.add_argument('--task', type=str, choices=['denoise', 'deblur'], default='deblur', help='Task type')
    # Stereo SR omitted for standard single-patch satellite processing, but can be added if needed
    return parser.parse_args()

def imread(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"[!] Could not read image at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img2tensor(img):
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=False, float32=True)

def run_nafnet():
    args = parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. SELECT CONFIGURATION BASED ON TASK
    print(f"\n[+] Initializing NAFNet ({args.task.upper()}) on GPU 1...")
    if args.task == 'denoise':
        opt_path = os.path.join(current_dir, "options", "test", "SIDD", "NAFNet-width64.yml")
    else: # deblur
        opt_path = os.path.join(current_dir, "options", "test", "GoPro", "NAFNet-width64.yml")

    if not os.path.exists(opt_path):
        print(f"[!] FATAL: Configuration not found at {opt_path}")
        sys.exit(1)

    # 2. BUILD ARCHITECTURE
    opt = parse(opt_path, is_train=False)
    opt["dist"] = False
    
    # --- THE FIX: DYNAMIC PATH OVERRIDE ---
    # Tell basicsr to ignore the YAML path and use the local directory
    weight_name = "NAFNet-GoPro-width64.pth" if args.task == 'deblur' else "NAFNet-SIDD-width64.pth"
    local_weight_path = os.path.join(current_dir, weight_name)
    
    if not os.path.exists(local_weight_path):
        print(f"[!] FATAL: Weights not found at {local_weight_path}")
        sys.exit(1)
        
    # Override the path dictionary
    opt['path']['pretrain_network_g'] = local_weight_path
    opt['path']['strict_load_g'] = True
    # --------------------------------------

    model = create_model(opt)

    # 3. LOAD & PRE-PROCESS IMAGE
    print(f"[*] Processing {args.task} on patch: {args.input_path}")
    img_input = imread(args.input_path)
    inp_tensor = img2tensor(img_input).cuda()

    # Save Input Image
    input_save_path = os.path.join(current_dir, "input_debug.png")
    imwrite(img_input, input_save_path)
    print(f"[✓] Input image saved for debugging at: {input_save_path}")

    # 4. INFERENCE
    with torch.no_grad():
        model.feed_data(data={"lq": inp_tensor.unsqueeze(dim=0)})
        
        if model.opt["val"].get("grids", False):
            model.grids()

        model.test()

        if model.opt["val"].get("grids", False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals["result"]])

    # 5. SAVE
    imwrite(sr_img, args.output_path)
    print(f"[SUCCESS] {args.task.capitalize()}d image saved to: {args.output_path}")

    # 6. TOTAL PURGE
    print("[-] Executing VRAM purge...")
    del model, img_input, inp_tensor, visuals, sr_img
    torch.cuda.empty_cache()
    gc.collect()
    print("[SUCCESS] GPU 1 is empty and ready for the next Expert.")

if __name__ == "__main__":
    run_nafnet()