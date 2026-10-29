from __future__ import annotations
from pathlib import Path
from ..tool import Tool

__all__ = ['roi_masking_toolbox']
project_root = Path(__file__).resolve().parents[2]


class RSSAM(Tool):
    """
    RS-SAM – Segment-Anything Model adapted for remote sensing ROI masking.
    Phase 0 Pre-Processor. Datasets: 38-Cloud, Cloud-Net Dataset.
    Produces binary masks for Local Path routing.
    """
    def __init__(self):
        super().__init__(
            tool_name="rs_sam",
            subtask="roi_masking",
            work_dir="RS_SAM",
            script_rel_path="infer_rs_sam.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--checkpoint", "none"
        ]


class CloudNetPlus(Tool):
    """
    Cloud-Net+ – Deep cloud/shadow segmentation for mask generation.
    Phase 0 Pre-Processor. Datasets: 38-Cloud, Cloud-Net Dataset.
    Produces binary masks for Local Path routing.
    """
    def __init__(self):
        super().__init__(
            tool_name="cloud_net_plus",
            subtask="roi_masking",
            work_dir="CloudNetPlus",
            script_rel_path="infer_cloudnetplus.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


class YOLOv8OBB(Tool):
    """
    YOLOv8-OBB – Oriented Bounding Box detector for arbitrary anomaly ROI masking.
    Phase 0 Pre-Processor.
    Produces bounding-box masks for Local Path routing.
    """
    def __init__(self):
        super().__init__(
            tool_name="yolov8_obb",
            subtask="roi_masking",
            work_dir="YOLOv8_OBB",
            script_rel_path="infer_yolov8_obb.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


subtask = 'roi_masking'
roi_masking_toolbox = [
    RSSAM(),
    CloudNetPlus(),
    YOLOv8OBB(),
]
