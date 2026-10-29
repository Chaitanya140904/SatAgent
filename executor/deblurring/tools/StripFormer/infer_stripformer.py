import os
import argparse
import torch
import cv2
import yaml
import numpy as np
from pathlib import Path
import torchvision.utils as vutils
from torch.autograd import Variable

# LOCKDOWN: GPU 1 Only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import Generator from the Stripformer architecture
from models.networks import get_generator

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_io(args):
    in_path = args.input_path
    if not in_path and args.input_dir:
        p = Path(args.input_dir)
        if p.is_dir():
            imgs = sorted([x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS])
            if imgs:
                in_path = str(imgs[0])
    if not in_path:
        raise ValueError("Missing input image. Provide --input_path or --input_dir.")

    out_path = args.output_path
    if not out_path:
        out_dir = args.output_dir or "."
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir_path / Path(in_path).name)
    return in_path, out_path


def parse_args():
    parser = argparse.ArgumentParser(description='Stripformer Satellite Deblurring')
    # Default to the GoPro config typically used for general deblurring
    parser.add_argument('--config', type=str, default='config/config_Stripformer_gopro.yaml', help='Path to config file')
    parser.add_argument('--input_path', type=str, default=None, help='Path to blurry satellite image')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory of blurry images (first image is used)')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save restored image')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output image')
    parser.add_argument('--weights', type=str, default="./Stripformer_gopro.pth", help='Path to trained weights .pth')
    args, _ = parser.parse_known_args()
    return args

def run_deblur():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path, output_path = _resolve_io(args)

    with open(args.config) as cfg:
        config = yaml.safe_load(cfg)
    
    print(f'===> Initializing Stripformer on GPU 1...')
    model = get_generator(config['model'])
    
    state_dict = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f'===> Processing: {input_path}')
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # --- NEW: Calculate padding to make dimensions multiples of 8 ---
    pad_factor = 8
    h_pad = (pad_factor - h_orig % pad_factor) % pad_factor
    w_pad = (pad_factor - w_orig % pad_factor) % pad_factor
    
    import torchvision.transforms.functional as F
    
    # Transform and pad
    img_tensor = torch.from_numpy(np.transpose(img_rgb / 255.0, (2, 0, 1)).astype('float32')) - 0.5
    # Apply padding: (left, top, right, bottom)
    img_tensor = F.pad(img_tensor, (0, 0, w_pad, h_pad), padding_mode='reflect')
    img_tensor = Variable(img_tensor.unsqueeze(0)).to(device)

    print(f'[*] Deblurring (Padded to: {w_orig+w_pad}x{h_orig+h_pad})...')

    with torch.no_grad():
        result_image = model(img_tensor)
        
        # Post-process
        result_image = result_image + 0.5
        result_image = torch.clamp(result_image, 0, 1)

        # --- NEW: Crop back to original dimensions ---
        # result_image is [Batch, C, H, W]
        result_image = result_image[:, :, :h_orig, :w_orig]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        vutils.save_image(result_image, output_path)
        print(f'[SUCCESS] Saved output to: {output_path}')

if __name__ == '__main__':
    run_deblur()
