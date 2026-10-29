import os
import argparse
import torch
import gc
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from diffusers import UNet2DModel, DDPMScheduler

# LOCKDOWN: GPU 1 Only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="CloudDiff Satellite Restoration")
    parser.add_argument('--input_path', type=str, required=True, help="Path to cloudy image")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save result")
    parser.add_argument('--pretrained_path', type=str, required=True, help="Path to UNet weights")
    # For restoration, we don't start from 100% noise. 0.5-0.7 is standard.
    parser.add_argument('--strength', type=float, default=0.6, help="Restoration strength (0 to 1)")
    return parser.parse_args()

def run_restoration():
    args = parse_args()
    device = "cuda"

    # --- 1: LOAD THE IMAGE ---
    print(f"[+] Loading cloudy image: {args.input_path}")
    init_image = Image.open(args.input_path).convert("RGB").resize((256, 256))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_tensor = transform(init_image).unsqueeze(0).to(device).half()

    # --- 2: BUILD ARCHITECTURE ---
    print("[+] Initializing CloudDiff UNet architecture...")
    model = UNet2DModel(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    ).to(device).half()

    # Load your actual trained weights
    if os.path.exists(args.pretrained_path):
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
    
    model.eval()

    # --- 3: SETUP SCHEDULER & NOISE ---
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(20) # Fast inference for the agent

    # Calculate starting step based on strength
    start_step = int(len(scheduler.timesteps) * (1 - args.strength))
    timesteps = scheduler.timesteps[start_step:]
    
    # Add noise to the cloudy image to create the "starting point" for diffusion
    noise = torch.randn_like(input_tensor)
    noisy_image = scheduler.add_noise(input_tensor, noise, timesteps[0])

    # --- 4: DIFFUSION LOOP ---
    print(f"[*] Diffusing clouds from image (Strength: {args.strength})...")
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Model predicts the cloud/noise component
            residual = model(noisy_image, t).sample
            
            # Scheduler steps back toward a clean image
            noisy_image = scheduler.step(residual, t, noisy_image).prev_sample

    # --- 5: POST-PROCESS & SAVE ---
    # Revert normalization and convert to BGR for CV2
    output = (noisy_image / 2 + 0.5).clamp(0, 1)
    output_np = (output.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(args.output_path, output_bgr)
    print(f"[SUCCESS] Saved output to: {args.output_path}")

    # --- 6: VRAM PURGE ---
    del model, scheduler, noisy_image, input_tensor
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    run_restoration()