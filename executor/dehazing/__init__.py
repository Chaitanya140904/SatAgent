from __future__ import annotations
"""
Dehazing Toolbox – Phase 1B of the SatAgent MoE pipeline.

Routing: GLOBAL (full-frame – haze is a sensor-wide atmospheric phenomenon)
Dataset: RS-Haze, RRSHID

Models integrated (from 4KAgent dehazing + satellite-specific):
  Fidelity  : rsdehamba, rshazenet, dehazeformer, xrestormer, maxim
  Perception : dehazing_diffgan, ridcp, diffplugin
"""
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import XRestormer, MAXIM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
subtask = "dehazing"


class RSDehamba(Tool):
    """
    RSDehamba – Remote-sensing dehazing with Mamba state-space model.
    Evaluated on RS-Haze; best PSNR/SSIM on satellite haze datasets.
    """
    def __init__(self):
        super().__init__(
            tool_name="rsdehamba", subtask=subtask,
            work_dir="RSDehamba", script_rel_path="infer_rsdehamba.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/RSDehamba/rsdehamba.pth"),
        ]


class RSHazeNet(Tool):
    """
    RSHazeNet – Attention-based dehazing for remote-sensing imagery.
    """
    def __init__(self):
        super().__init__(
            tool_name="rshazenet", subtask=subtask,
            work_dir="RSHazeNet", script_rel_path="infer_rshazenet.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/RSHazeNet/rshazenet.pth"),
        ]


class DehazeFormer(Tool):
    """
    [TIP 2023] DehazeFormer – Vision Transformer for Single Image Dehazing.
    General-purpose; strong baseline from 4KAgent.
    """
    def __init__(self):
        super().__init__(
            tool_name="dehazeformer", subtask=subtask,
            work_dir="DehazeFormer", script_rel_path="inference.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--data_dir", str(self.input_path.parent),
            "--result_dir", str(self.output_path.parent),
            "--save_dir", str(PROJECT_ROOT / "pretrained_ckpts/DehazeFormer"),
            "--tile_size", "1024",
            "--tile_overlap", "64",
        ]


class DehazingDiffGAN(Tool):
    """
    Dehazing-DiffGAN – Diffusion + GAN hybrid for perceptual dehazing.
    Evaluated on RRSHID for NIQE / FID on satellite-domain haze.
    """
    def __init__(self):
        super().__init__(
            tool_name="dehazing_diffgan", subtask=subtask,
            work_dir="DehazingDiffGAN", script_rel_path="infer_dehazing_diffgan.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/DehazingDiffGAN/diffgan_dehaze.pth"),
        ]


class RIDCP(Tool):
    """
    [CVPR 2023] RIDCP – Revitalizing Real Image Dehazing via High-Quality Codebook Priors.
    From 4KAgent's dehazing toolbox; strong perceptual quality.
    """
    def __init__(self):
        super().__init__(
            tool_name="ridcp", subtask=subtask,
            work_dir="RIDCP_dehazing", script_rel_path="infer_ridcp_4kagent.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "-i", str(self.input_path.parent),
            "-o", str(self.output_path.parent),
            "--weight", str(PROJECT_ROOT / "pretrained_ckpts/RIDCP_dehazing/pretrained_RIDCP.pth"),
            "--matching_weight_path",
            str(PROJECT_ROOT / "pretrained_ckpts/RIDCP_dehazing/weight_for_matching_dehazing_Flickr.pth"),
            "--use_weight",
            "--alpha", "-21.25",
        ]


# ── Toolbox assembly ──────────────────────────────────────────────────────────
dehazing_toolbox = [
    # Fidelity tier (PSNR/SSIM)
    RSDehamba(),
    RSHazeNet(),
    DehazeFormer(),
    XRestormer(subtask=subtask),
    MAXIM(subtask=subtask),
    # Perception tier (NIQE/LPIPS)
    DehazingDiffGAN(),
    RIDCP(),
]
