from __future__ import annotations
from pathlib import Path
from ..tool import Tool

__all__ = ['temporal_fusion_toolbox']
project_root = Path(__file__).resolve().parents[2]


class TransFusion(Tool):
    """
    TransFusion – Transformer-based multi-temporal satellite image fusion.
    Routing: Global Path (full-frame). Datasets: SpaceNet, OSCD.
    Mode: Both Fidelity and Perception (same model, different eval metrics).
    """
    def __init__(self):
        super().__init__(
            tool_name="transfusion",
            subtask="temporal_fusion",
            work_dir="TransFusion",
            script_rel_path="infer_transfusion.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


class TimeSformerRS(Tool):
    """
    TimeSformer-RS – Video-Transformer adapted for multi-temporal RS fusion.
    Routing: Global Path (full-frame). Datasets: SpaceNet, OSCD.
    Mode: Both Fidelity and Perception (same model, different eval metrics).
    """
    def __init__(self):
        super().__init__(
            tool_name="timesformer_rs",
            subtask="temporal_fusion",
            work_dir="TimeSformerRS",
            script_rel_path="infer_timesformer_rs.py"
        )

    def _get_cmd_opts(self) -> list[str]:
        return [
            "--input_path", str(self.input_path),
            "--output_path", str(self.output_path),
            "--weights", "none"
        ]


subtask = 'temporal_fusion'
temporal_fusion_toolbox = [
    TransFusion(),
    TimeSformerRS(),
]
