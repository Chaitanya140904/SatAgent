from __future__ import annotations
"""
Color correction utilities adapted from 4KAgent.
Used by Agent 2 after model execution to correct color drift introduced
by generative models (diffusion / GAN).

Two modes:
  - AdaIN  : Adaptive Instance Normalization – fast, coarse correction.
  - Wavelet: Wavelet frequency decomposition – preserves fine texture.
"""
import cv2
import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import ToTensor, ToPILImage


# ---------------------------------------------------------------------------
# Tensor-level helpers
# ---------------------------------------------------------------------------

def _calc_mean_std(feat: Tensor, eps: float = 1e-5):
    b, c = feat.size()[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std


def _adaptive_instance_normalization(content: Tensor, style: Tensor) -> Tensor:
    size = content.size()
    style_mean, style_std = _calc_mean_std(style)
    content_mean, content_std = _calc_mean_std(content)
    normalised = (content - content_mean.expand(size)) / content_std.expand(size)
    return normalised * style_std.expand(size) + style_mean.expand(size)


def _wavelet_blur(image: Tensor, radius: int) -> Tensor:
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.250, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    return F.conv2d(image, kernel, groups=3, dilation=radius)


def _wavelet_decompose(image: Tensor, levels: int = 5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = _wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def _wavelet_reconstruction(content: Tensor, style: Tensor) -> Tensor:
    content_high, _ = _wavelet_decompose(content)
    _, style_low = _wavelet_decompose(style)
    return content_high + style_low


# ---------------------------------------------------------------------------
# Public API (PIL Image → PIL Image)
# ---------------------------------------------------------------------------

def adain_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """
    Apply AdaIN color correction: match target's color distribution to source.
    Fast but can over-smoothen fine color gradients.
    """
    to_tensor = ToTensor()
    to_img = ToPILImage()
    t_t = to_tensor(target).unsqueeze(0)
    s_t = to_tensor(source).unsqueeze(0)
    result = _adaptive_instance_normalization(t_t, s_t)
    return to_img(result.squeeze(0).clamp_(0.0, 1.0))


def wavelet_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """
    Apply wavelet color correction: replaces low-frequency color from source
    while preserving high-frequency texture from target.
    Preferred for satellite imagery – preserves spectral structure better.
    """
    to_tensor = ToTensor()
    to_img = ToPILImage()
    t_t = to_tensor(target).unsqueeze(0)
    s_t = to_tensor(source).unsqueeze(0)
    result = _wavelet_reconstruction(t_t, s_t)
    return to_img(result.squeeze(0).clamp_(0.0, 1.0))


# ---------------------------------------------------------------------------
# OpenCV ↔ PIL converters
# ---------------------------------------------------------------------------

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    if cv2_img is None:
        raise ValueError("Input image is None.")
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    if pil_img is None:
        raise ValueError("Input image is None.")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
