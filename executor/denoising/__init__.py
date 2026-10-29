from __future__ import annotations
"""
Denoising Toolbox – Phase 2A of the SatAgent MoE pipeline.

Routing: GLOBAL (full-frame)
Dataset: xView, UC Merced (+ Poisson noise injection)

Models integrated (from 4KAgent + satellite-specific):
  Fidelity  : restormer, mprnet, xrestormer, swinir_15, swinir_50, nafnet, satmae, nafnet_rs
  Perception : diffusionsat, maxim

Note: Restormer is available but should be used extremely rarely due to
occasional GPU driver hangs on some systems. Prefer SatMAE/NAFNet-RS first.
"""
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import (
    XRestormer, NAFNet, SwinIR, Restormer, MPRNet, MAXIM
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
subtask = "denoising"


class SatMAE(Tool):
    """
    SatMAE – Masked Autoencoder pre-trained on multispectral satellite data.
    Fine-tuned for denoising; best SAM on Landsat/Sentinel data.
    """
    def __init__(self):
        super().__init__(
            tool_name="satmae", subtask=subtask,
            work_dir="SatMAE", script_rel_path="infer_satmae_denoise.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/SatMAE/satmae_denoise.pth"),
        ]


class NAFNet_RS(Tool):
    """
    NAFNet fine-tuned on remote-sensing noise datasets (xView + UC Merced Poisson).
    """
    def __init__(self):
        super().__init__(
            tool_name="nafnet_rs", subtask=subtask,
            work_dir="NAFNet", script_rel_path="infer_nafnet_rs_denoise.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/NAFNet_RS/nafnet_rs_denoise.pth"),
        ]


class DiffusionSat(Tool):
    """
    DiffusionSat – Satellite-conditioned diffusion model.
    Evaluated for perceptual quality (NIQE/LPIPS).
    """
    def __init__(self):
        super().__init__(
            tool_name="diffusionsat", subtask=subtask,
            work_dir="DiffusionSat", script_rel_path="infer_diffusionsat_denoise.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/DiffusionSat/diffusionsat_denoise.pth"),
        ]


# ── Toolbox assembly (Fidelity first) ────────────────────────────────────────
denoising_toolbox = [
    # Fidelity tier (PSNR/SSIM/SAM)
    SatMAE(),
    NAFNet_RS(),
    Restormer(subtask=subtask),
    MPRNet(subtask=subtask),
    XRestormer(subtask=subtask),
    SwinIR(subtask=subtask, pretrained_on="15"),
    SwinIR(subtask=subtask, pretrained_on="50"),
    NAFNet(subtask=subtask),
    MAXIM(subtask=subtask),
    # Perception tier (NIQE/LPIPS)
    DiffusionSat(),
]
