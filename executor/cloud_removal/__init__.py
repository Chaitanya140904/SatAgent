from __future__ import annotations
from pathlib import Path
from ..tool import Tool

__all__ = ['cloud_removal_toolbox']
project_root = Path(__file__).resolve().parents[2]


class DVPNet(Tool):
    """
    DVPNet – Deep Video Prior for thick cloud removal.
    Routing: Local Path (mask-dependent). Datasets: SEN12MS-CR, RICE-II.
    Mode: Fidelity (PSNR/SSIM/SAM optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="dvpnet",
            subtask="cloud_removal",
            work_dir="DVPNet",
            script_rel_path="infer_dvpnet.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


class UnCRtainTS(Tool):
    """
    UnCRtainTS – Uncertainty-aware Cloud Removal with Temporal Series.
    Routing: Local Path (mask-dependent). Datasets: SEN12MS-CR, RICE-II.
    Mode: Fidelity (PSNR/SSIM/SAM optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="uncrtaints",
            subtask="cloud_removal",
            work_dir="UnCRtainTS",
            script_rel_path="infer_uncrtaints.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


class EMRDM(Tool):
    """
    EMRDM – Efficient Multi-Resolution Diffusion Model for cloud removal.
    Routing: Local Path (mask-dependent). Datasets: SEN12MS-CR, RICE-II.
    Mode: Perception (NIQE/LPIPS/FID optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="emrdm",
            subtask="cloud_removal",
            work_dir="EMRDM",
            script_rel_path="infer_emrdm.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path)
        ]


class CloudMamba(Tool):
    """
    CloudMamba – State-Space-Model cloud removal (Mamba architecture).
    Routing: Local Path (mask-dependent). Datasets: SEN12MS-CR, RICE-II.
    Mode: Perception (NIQE/LPIPS/FID optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="cloud_mamba",
            subtask="cloud_removal",
            work_dir="CloudMamba",
            script_rel_path="infer_cloudmamba.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


subtask = 'cloud_removal'
cloud_removal_toolbox = [
    DVPNet(),
    UnCRtainTS(),
    EMRDM(),
    CloudMamba(),
]