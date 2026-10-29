from __future__ import annotations
from pathlib import Path
from ..tool import Tool

__all__ = ['pan_sharpen_toolbox']
project_root = Path(__file__).resolve().parents[2]


class PanMamba(Tool):
    """
    PanMamba – Mamba-architecture pan-sharpening for GF2/WV3.
    Routing: Global Path (full-frame). Datasets: GaoFen-2, WorldView-3.
    Mode: Fidelity (PSNR/SSIM/SAM optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="pan_mamba",
            subtask="pan_sharpen",
            work_dir="PanMamba",
            script_rel_path="infer_panmamba.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


class ADWM(Tool):
    """
    ADWM – Adaptive Detail Weight Modulation pan-sharpening.
    Routing: Global Path (full-frame). Datasets: GaoFen-2, WorldView-3.
    Mode: Fidelity (PSNR/SSIM/SAM optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="adwm",
            subtask="pan_sharpen",
            work_dir="ADWM",
            script_rel_path="infer_adwm.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


class MGDN(Tool):
    """
    MGDN – Multi-scale Guided Detail Network for perceptual pan-sharpening.
    Routing: Global Path (full-frame). Datasets: GaoFen-2, WorldView-3.
    Mode: Perception (NIQE/LPIPS/FID optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="mgdn",
            subtask="pan_sharpen",
            work_dir="MGDN",
            script_rel_path="infer_mgdn.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path)
        ]


class PreMix(Tool):
    """
    PreMix – Frequency mixing pan-sharpening (perceptual).
    Routing: Global Path (full-frame). Datasets: GaoFen-2, WorldView-3.
    Mode: Perception (NIQE/LPIPS/FID optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="premix",
            subtask="pan_sharpen",
            work_dir="PreMix",
            script_rel_path="infer_premix.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path)
        ]


subtask = 'pan_sharpen'
pan_sharpen_toolbox = [
    PanMamba(),
    ADWM(),
    MGDN(),
    PreMix(),
]
