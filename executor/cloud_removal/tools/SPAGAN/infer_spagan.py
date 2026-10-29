import argparse
import logging
import os
import sys
import traceback
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# Force GPU 1 as the only visible device BEFORE torch initializes further
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add the SpAGAN subdirectory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
spagan_repo_path = os.path.join(current_dir, "SpA-GAN_for_cloud_removal")
if spagan_repo_path not in sys.path:
    sys.path.append(spagan_repo_path)

# Import Generator directly
from models.gen.SPANet import Generator

class AttrDict(dict):
    """A simple helper to allow dot notation access to dictionary keys."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    # gpu_ids is kept for Generator initialization compatibility
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0]) 
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        config = AttrDict(data)
    return args, config

def predict(config, args):
    print('===> Loading model')
    # Use [0] because CUDA_VISIBLE_DEVICES=1 makes GPU 1 look like index 0
    gen = Generator(gpu_ids=[0]) 
    
    param = torch.load(args.pretrained, map_location='cpu')
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda()

    gen.eval()

    print(f"===> Processing image: {args.input_path}")
    img = Image.open(args.input_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    x = transform(img).unsqueeze(0)
    
    if args.cuda:
        x = x.cuda()

    with torch.no_grad():
        att, out = gen(x)
        out_ = out.cpu().numpy()[0]
        out_rgb = np.clip(out_[:3], 0, 1) * 255.0
        out_rgb = out_rgb.transpose(1, 2, 0).astype(np.uint8)

        # Save as BGR for OpenCV
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output_path, out_bgr)
        print(f"===> Saved output image to: {args.output_path}")
    
if __name__ == '__main__':
    try:
        args, config = parse_args_and_config()
        # predict now handles its own device placement
        predict(config, args)
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        traceback.print_exc()
        sys.exit(1)