from __future__ import annotations
"""
Artifact Removal Toolbox – Phase 2D of the SatAgent MoE pipeline.

Routing: GLOBAL (full-frame)
Dataset: AID (with heavy JPEG compression q=[5…40])

Models integrated (from 4KAgent JPEG CAR + satellite-specific):
  Fidelity  : fbcnn_blind, fbcnn_5, fbcnn_90, swinir_40, nafnet_jpeg
  Perception : swinir_car
"""
from pathlib import Path

from ..tool import Tool
from ..multitask_tools import SwinIR, NAFNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
subtask = "artifact_removal"


class FBCNN(Tool):
    """
    [ICCV 2021] FBCNN – Flexible Blind JPEG Artifacts Removal.
    Multiple quality-factor checkpoints; 'blind' auto-detects QF.
    Adapted from 4KAgent's FBCNN implementation.
    """
    def __init__(self, qf: str = "blind"):
        super().__init__(
            tool_name=f"fbcnn_{qf}",
            subtask=subtask,
            work_dir="FBCNN",
            script_rel_path="infer_fbcnn_4kagent.py",
        )
        self.qf = qf

    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--weight_dir", "FBCNN/model_zoo",
            "--output_dir", str(self.output_path.parent),
            "--qf", str(self.qf),
        ]


class NAFNet_JPEG(Tool):
    """
    NAFNet fine-tuned for JPEG artifact removal on remote-sensing imagery (AID).
    """
    def __init__(self):
        super().__init__(
            tool_name="nafnet_jpeg", subtask=subtask,
            work_dir="NAFNet", script_rel_path="infer_nafnet_jpeg.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/NAFNet_JPEG/nafnet_jpeg_aid.pth"),
        ]


class SwinIR_CAR(Tool):
    """
    SwinIR trained for Color JPEG CAR (q=40).
    Slightly over-smoothened but excellent perceptual quality.
    """
    def __init__(self):
        super().__init__(
            tool_name="swinir_car", subtask=subtask,
            work_dir="SwinIR", script_rel_path="infer_swinir_4kagent.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--task", "color_jpeg_car",
            "--model_path", "SwinIR/model_zoo/swinir/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
            "--folder_lq", str(self.input_path.parent),
            "--save_dir", str(self.output_path.parent),
            "--tile", "896",
            "--tile_overlap", "56",
        ]


# ── Toolbox assembly ──────────────────────────────────────────────────────────
artifact_removal_toolbox = [
    # Fidelity tier
    FBCNN(qf="blind"),
    FBCNN(qf=5),
    FBCNN(qf=90),
    NAFNet_JPEG(),
    SwinIR(subtask=subtask, pretrained_on="40"),
    # Perception tier
    SwinIR_CAR(),
]
