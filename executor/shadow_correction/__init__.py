from __future__ import annotations
from pathlib import Path
from ..tool import Tool

__all__ = ['shadow_correction_toolbox']
project_root = Path(__file__).resolve().parents[2]


class ShadowFormer(Tool):
    """
    ShadowFormer – Transformer-based shadow removal.
    Routing: Local Path (mask-dependent). Datasets: RSD, UIIS.
    Mode: Fidelity (PSNR/SSIM optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="shadowformer",
            subtask="shadow_correction",
            work_dir="ShadowFormer",
            script_rel_path="infer_shadowformer.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--model_path", "none"
        ]


class EnlightenGAN_RS(Tool):
    """
    EnlightenGAN-RS – GAN-based shadow/dark-region correction.
    Routing: Local Path (mask-dependent). Datasets: RSD, UIIS.
    Mode: Perception (NIQE/LPIPS optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="enlighten_gan_rs",
            subtask="shadow_correction",
            work_dir="EnlightenGAN_RS",
            script_rel_path="infer_enlightengan_rs.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path)
        ]


subtask = 'shadow_correction'
shadow_correction_toolbox = [
    ShadowFormer(),
    EnlightenGAN_RS(),
]
