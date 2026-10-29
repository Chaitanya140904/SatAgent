from __future__ import annotations
"""
Deblurring Toolbox – Phase 2B of the SatAgent MoE pipeline.

Routing: GLOBAL (full-frame)
Dataset: DOTA (+ linear motion blur injection)

Models integrated (from 4KAgent motion_deblurring + satellite-specific):
  Fidelity  : restormer, mprnet, xrestormer, nafnet, evssm, pirn, stripformer
  Perception : maxim, blur_mamba, deblurdiff

Note: Restormer is wired but should be used extremely rarely in practice due to
occasional GPU driver hangs on some systems. Prefer PiRN/StripFormer/BlurMamba.
"""
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import XRestormer, NAFNet, Restormer, MPRNet, MAXIM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
subtask = "deblurring"


class EVSSM(Tool):
    """
    EVSSM – State-Space Vision Model for motion deblurring.
    Strong performance on orbital-blur (linear kernel) datasets.
    """
    def __init__(self):
        super().__init__(
            tool_name="EVSSM", subtask=subtask,
            work_dir="EVSSM", script_rel_path="infer_evssm_4kagent.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--save_image_dir", str(self.output_path.parent),
            "--ckpts", str(PROJECT_ROOT / "pretrained_ckpts/EVSSM/net_g_GoPro.pth"),
        ]


class PiRN(Tool):
    """
    PiRN – Physics-informed Restoration Network for orbital motion deblurring.
    Evaluated on DOTA with linear motion kernels.
    """
    def __init__(self):
        super().__init__(
            tool_name="pirn", subtask=subtask,
            work_dir="PiRN", script_rel_path="infer_pirn.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/PiRN/pirn_orbital.pth"),
        ]


class StripFormer(Tool):
    """
    StripFormer – Strip-attention Transformer for motion deblurring.
    State-of-the-art on GoPro/HIDE; adapted for satellite linear blur.
    """
    def __init__(self):
        super().__init__(
            tool_name="stripformer", subtask=subtask,
            work_dir="StripFormer", script_rel_path="infer_stripformer.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/StripFormer/stripformer_gopro.pth"),
        ]


class BlurMamba(Tool):
    """
    BlurMamba – State-Space Model deblurring with perceptual loss.
    Optimised for LPIPS on satellite-domain blur.
    """
    def __init__(self):
        super().__init__(
            tool_name="blur_mamba", subtask=subtask,
            work_dir="BlurMamba", script_rel_path="infer_blurmamba.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/BlurMamba/blurmamba_sat.pth"),
        ]


class DeblurDiff(Tool):
    """
    DeblurDiff – Diffusion-based deblurring for perceptual fidelity.
    """
    def __init__(self):
        super().__init__(
            tool_name="deblurdiff", subtask=subtask,
            work_dir="DeblurDiff", script_rel_path="infer_deblurdiff.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/DeblurDiff/deblurdiff.pth"),
        ]


# ── Toolbox assembly ──────────────────────────────────────────────────────────
deblurring_toolbox = [
    # Fidelity tier
    PiRN(),
    StripFormer(),
    Restormer(subtask=subtask),
    MPRNet(subtask=subtask),
    XRestormer(subtask=subtask),
    NAFNet(subtask=subtask),
    EVSSM(),
    MAXIM(subtask=subtask),
    # Perception tier
    BlurMamba(),
    DeblurDiff(),
]
