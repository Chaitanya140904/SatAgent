from __future__ import annotations
from pathlib import Path
from ..tool import Tool

__all__ = ['destriping_toolbox']
project_root = Path(__file__).resolve().parents[2]


class QGCN(Tool):
    """
    QGCN – Quasi-Graph Convolutional Network for stripe removal.
    Routing: Local Path (mask-dependent). Datasets: Aqua MODIS, EO-1 Hyperion.
    Mode: Fidelity (PSNR/SSIM optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="qgcn",
            subtask="destriping",
            work_dir="QGCN",
            script_rel_path="infer_qgcn.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


class SSTNet(Tool):
    """
    SST-Net – Spatial-Spectral Transformer for destriping.
    Routing: Local Path (mask-dependent). Datasets: Aqua MODIS, EO-1 Hyperion.
    Mode: Fidelity (PSNR/SSIM optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="sst_net",
            subtask="destriping",
            work_dir="SSTNet",
            script_rel_path="infer_sstnet.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


class RepaintRS(Tool):
    """
    Repaint-RS – Diffusion-based dead-pixel inpainting.
    Routing: Local Path (mask-dependent). Datasets: Aqua MODIS, EO-1 Hyperion.
    Mode: Perception (NIQE/LPIPS optimised).
    """
    def __init__(self):
        super().__init__(
            tool_name="repaint_rs",
            subtask="destriping",
            work_dir="RepaintRS",
            script_rel_path="infer_repaint_rs.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--mask_path", "none"
        ]


subtask = 'destriping'
destriping_toolbox = [
    QGCN(),
    SSTNet(),
    RepaintRS(),
]
