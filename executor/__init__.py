from __future__ import annotations
import os
from pathlib import Path
import shutil
import time
import cv2
from typing import Set, List, Dict, Optional, Tuple

from .artifact_removal import artifact_removal_toolbox
from .cloud_removal import cloud_removal_toolbox
from .deblurring import deblurring_toolbox
from .dehazing import dehazing_toolbox
from .denoising import denoising_toolbox
from .destriping import destriping_toolbox
from .pan_sharpen import pan_sharpen_toolbox
from .roi_masking import roi_masking_toolbox
from .shadow_correction import shadow_correction_toolbox
from .super_resolution import sr_toolbox
from .temporal_fusion import temporal_fusion_toolbox

from .tool import Tool

__all__ = ['executor']

# ---------------------------------------------------------------------------
# Routing constants
# ---------------------------------------------------------------------------
# Tasks that operate on a locally-masked ROI (Local Path / mask-dependent).
LOCAL_PATH_TASKS = {
    "cloud_removal",      # Phase 1A
    "shadow_correction",  # Phase 1C
    "destriping",         # Phase 2C
}

# Tasks that process the full sensor frame (Global Path).
GLOBAL_PATH_TASKS = {
    "dehazing",           # Phase 1B
    "denoising",          # Phase 2A
    "deblurring",         # Phase 2B
    "artifact_removal",   # Phase 2D
    "pan_sharpen",        # Phase 3A
    "temporal_fusion",    # Phase 3B
    "super_resolution",   # Phase 4
}

# Phase 0 pre-processor (runs before everything else on Local-Path tasks).
MASKING_TASK = "roi_masking"


class Executor:
    def __init__(self):
        self.toolbox_router: Dict[str, List[Tool]] = {}
        self._executed_subtask_cnt: int = 0

    def register_subtask(self, subtask_name: str, subtask_toolbox: List[Tool]) -> None:
        self.toolbox_router[subtask_name] = subtask_toolbox

    @property
    def subtasks(self) -> Set[str]:
        return set(self.toolbox_router.keys())

    @property
    def executed_subtask_cnt(self) -> int:
        return self._executed_subtask_cnt

    # ------------------------------------------------------------------
    # Core invocation helpers
    # ------------------------------------------------------------------

    def invoke_a_tool(
        self,
        subtask_name: str,
        tool_name: str,
        input_path: Path,
        output_path: Path,
        silent: bool = False,
        run_gpu_id: Optional[int] = None,
    ) -> None:
        """Direct invocation of a named tool within a subtask toolbox."""
        toolbox = self.toolbox_router.get(subtask_name, [])
        for tool in toolbox:
            if tool.tool_name == tool_name:
                tool(input_path, output_path, silent=silent, run_gpu_id=run_gpu_id)
                return
        raise KeyError(f"Tool '{tool_name}' not found in subtask '{subtask_name}'.")

    def get_routing_mode(self, subtask: str) -> str:
        """Returns 'local' or 'global' routing classification for a subtask."""
        if subtask in LOCAL_PATH_TASKS:
            return "local"
        return "global"

    # ------------------------------------------------------------------
    # Phase 0 – ROI Mask Generation (Local-Path pre-processor)
    # ------------------------------------------------------------------

    def generate_roi_mask(
        self,
        input_path: Path,
        mask_output_path: Path,
        mask_tool: str = "rs_sam",
        silent: bool = False,
        run_gpu_id: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Run the Phase-0 masking agent to produce a binary ROI mask.
        Returns path to the mask file, or None on failure.
        """
        try:
            toolbox = self.toolbox_router.get(MASKING_TASK, [])
            for tool in toolbox:
                if tool.tool_name == mask_tool:
                    tool(
                        input_path,
                        mask_output_path,
                        silent=silent,
                        run_gpu_id=run_gpu_id,
                    )
                    if mask_output_path.exists():
                        return mask_output_path
            return None
        except Exception as e:
            if not silent:
                print(f"      [!] ROI mask generation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Execute subtask (full pipeline step)
    # ------------------------------------------------------------------

    def execute_subtask(self, subtask: str, input_path: Path) -> Path:
        """
        Invokes the *first* tool in a subtask toolbox and returns output path.
        Follows the MoE routing protocol:
          - Local-Path tasks: generates ROI mask first, then runs the tool.
          - Global-Path tasks: runs the tool on the full frame directly.

        Directory layout produced:
        ```
        {parent}/
          subtask{N}-{subtask}/
            tool1-{name}/
              output.png
        ```
        """
        self._executed_subtask_cnt += 1
        subtask_dir = input_path.parents[1] / \
            f"subtask{self._executed_subtask_cnt}-{subtask.replace(' ', '_')}"
        subtask_dir.mkdir(parents=True, exist_ok=True)

        toolbox = self.toolbox_router[subtask]
        routing = self.get_routing_mode(subtask)

        # Phase 0 – generate mask for local-path tasks
        mask_path: Optional[Path] = None
        if routing == "local":
            mask_dir = subtask_dir / "0-mask"
            mask_dir.mkdir(exist_ok=True)
            mask_output = mask_dir / "mask.png"
            mask_path = self.generate_roi_mask(
                input_path=input_path,
                mask_output_path=mask_output,
                silent=True,
            )

        output_path: Optional[Path] = None
        tool_idx = 0
        for tool in toolbox:
            tool_idx += 1
            tool_dir = subtask_dir / f"tool{tool_idx}-{tool.tool_name}"
            output_dir = tool_dir / "0-img"
            output_dir.mkdir(parents=True, exist_ok=True)
            final_out = output_dir / input_path.name
            tool(input_path=input_path, output_path=final_out)
            outputs = list(output_dir.glob("*"))
            if outputs:
                output_path = outputs[0]
            break  # Only run the primary (first) tool in pipeline mode.

        return output_path or input_path

    # ------------------------------------------------------------------
    # Testing helpers
    # ------------------------------------------------------------------

    def test_toolbox(
        self,
        input_path: Path,
        output_dir: Path,
        subtask: str,
    ) -> None:
        toolbox = self.toolbox_router[subtask]
        tmp_output_dir = output_dir / "tmp"
        tmp_output_dir.mkdir(exist_ok=True)
        for tool in toolbox:
            out_file = tmp_output_dir / input_path.name
            tool(input_path, out_file)
            outputs = list(tmp_output_dir.glob("*"))
            if len(outputs) == 1:
                outputs[0].replace(output_dir / f"{tool.tool_name}.png")
            else:
                for output in outputs:
                    output.replace(output_dir / f"{tool.tool_name}_{output.stem[7:]}.png")
        tmp_output_dir.rmdir()

    def test_all_tools(self, input_dir: Path, output_dir: Path) -> None:
        def _check_shape(input_shape, output_shape):
            if input_shape == output_shape:
                return True
            if (input_shape[0] * 4 == output_shape[0]
                    and input_shape[1] * 4 == output_shape[1]):
                return True
            return False

        start_time = time.time()
        imgs = sorted(list(input_dir.glob("*")))
        tool_cnt = 0
        misaligned_tools: list[tuple[str, str]] = []

        for i, img_path in enumerate(imgs):
            input_shape = cv2.imread(str(img_path)).shape
            subtask_name = img_path.stem
            subtask_dir = output_dir / f"{i+1}-{subtask_name.replace(' ', '_')}"
            new_input_dir = subtask_dir / "0-input"
            new_input_path = new_input_dir / "input.png"
            new_input_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, new_input_path)
            toolbox = self.toolbox_router[subtask_name]
            tool_idx = 0

            for tool in toolbox:
                tool_cnt += 1
                tool_idx += 1
                tool_dir = subtask_dir / f"{tool_idx}_{tool.tool_name}"
                tool_dir.mkdir(exist_ok=True)
                output_file = tool_dir / "output.png"
                tool(new_input_path, output_path=output_file)
                image_files = [
                    f for f in tool_dir.glob("*")
                    if f.suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
                ]
                if image_files:
                    output_shape = cv2.imread(str(image_files[0])).shape
                    if not _check_shape(input_shape, output_shape):
                        misaligned_tools.append((subtask_name, tool.tool_name))
                    image_files[0].replace(
                        subtask_dir / f"{tool_idx}_{tool.tool_name}.png"
                    )
                tool_dir.rmdir()

            new_input_path.replace(subtask_dir / "input.png")
            new_input_dir.rmdir()

        end_time = time.time()
        consumed_time = end_time - start_time
        print(f"Time elapsed: {consumed_time:.2f}s")
        print(f"Tool count: {tool_cnt}")
        if tool_cnt:
            print(f"Average time per tool: {consumed_time / tool_cnt:.2f}s")
        print(f"Tools that cannot keep the image size: {misaligned_tools}")


# ---------------------------------------------------------------------------
# Singleton – all 11 subtasks registered according to MoE spec
# ---------------------------------------------------------------------------
executor = Executor()

# Phase 0 – Pre-processor (masking)
executor.register_subtask("roi_masking",        roi_masking_toolbox)

# Phase 1 – Atmospheric & Environmental
executor.register_subtask("cloud_removal",      cloud_removal_toolbox)
executor.register_subtask("dehazing",           dehazing_toolbox)
executor.register_subtask("shadow_correction",  shadow_correction_toolbox)

# Phase 2 – Sensor & Hardware Physics
executor.register_subtask("denoising",          denoising_toolbox)
executor.register_subtask("deblurring",         deblurring_toolbox)
executor.register_subtask("destriping",         destriping_toolbox)
executor.register_subtask("artifact_removal",   artifact_removal_toolbox)

# Phase 3 – Core Fusion
executor.register_subtask("pan_sharpen",        pan_sharpen_toolbox)
executor.register_subtask("temporal_fusion",    temporal_fusion_toolbox)

# Phase 4 – 4K Super-Resolution
executor.register_subtask("super_resolution",   sr_toolbox)
