from __future__ import annotations
"""
SatAgent Multitask Tools – Satellite-domain reusable model wrappers.

Adapted from 4KAgent/executor/multitask_tools.py.
All tools follow the sata Tool contract:
  __call__(input_path: Path, output_path: Path, silent: bool, run_gpu_id: int)

Key additions vs 4KAgent:
  - work_dir now resolves from THIS file (sata/executor/) not CWD
  - python → python3 in all subprocess calls
  - Satellite-specific tile sizes (larger to preserve spatial context)
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import yaml

from .tool import Tool

# Absolute path to the sata project root (parent of executor/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# BasicSR-based model base (HAT, NAFNet, SwinFIR …)
# ---------------------------------------------------------------------------

class BasicSRModel(Tool):
    """
    Wrapper for models using the BasicSR training framework.
    Expects a YAML config under:
      executor/{subtask}/configs/{tool_name}.yml
    And an inference script:
      executor/{subtask}/tools/{work_dir}/{tool_name}/inference.py
    """

    def __init__(
        self,
        tool_name: str,
        subtask: str,
        work_dir: str,
        script_rel_path: Optional[str] = None,
    ):
        super().__init__(
            tool_name=tool_name,
            subtask=subtask,
            work_dir=work_dir,
            script_rel_path=(
                Path(tool_name) / "inference.py"
                if script_rel_path is None
                else script_rel_path
            ),
        )
        self.work_dir_name = work_dir

    def _preprocess(self) -> None:
        cfg_path = (
            PROJECT_ROOT / "executor" / self.subtask / "configs" / f"{self.tool_name}.yml"
        )
        if not cfg_path.exists():
            return  # Skip config preprocessing if config file doesn't exist

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["datasets"]["test_1"]["dataroot_lq"] = str(self.input_path.parent)
        cfg["path"]["results"] = str(self.output_path.parent)

        self._update_pretrained_ckpt(cfg)

        self.new_cfg_dir = self.output_path.parent / "cfg"
        self.new_cfg_dir.mkdir(exist_ok=True)
        self.new_cfg_path = self.new_cfg_dir / "cfg.yml"
        with open(self.new_cfg_path, "w") as f:
            yaml.dump(cfg, f)

    def _update_pretrained_ckpt(self, cfg: dict) -> None:
        """Override in subclasses to update checkpoint path."""
        pass

    def _get_cmd_opts(self) -> list:
        if hasattr(self, "new_cfg_path"):
            return ["-opt", str(self.new_cfg_path)]
        return []

    def _postprocess(self) -> None:
        if hasattr(self, "new_cfg_dir") and self.new_cfg_dir.exists():
            shutil.rmtree(self.new_cfg_dir)

        # Flatten any nested output
        outputs = list(self.output_path.parent.rglob("*.png"))
        if len(outputs) == 1 and outputs[0] != self.output_path:
            shutil.move(str(outputs[0]), str(self.output_path))
            # Clean up empty subdirs
            for p in self.output_path.parent.glob("*"):
                if p.is_dir():
                    shutil.rmtree(p)


# ---------------------------------------------------------------------------
# XRestormer (multi-task: SR, denoising, dehazing, deblurring, deraining)
# ---------------------------------------------------------------------------

class XRestormer(BasicSRModel):
    """
    [ECCV 2024] A Comparative Study of Image Restoration Networks for General
    Backbone Network Design. Handles SR, denoising, dehazing, motion deblurring,
    deraining with a single Transformer backbone.
    """

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="xrestormer",
            subtask=subtask,
            work_dir="X-Restormer",
        )

    def _update_pretrained_ckpt(self, cfg: dict) -> None:
        if "path" in cfg and "pretrain_network_g" in cfg["path"]:
            ckpt_name = cfg["path"]["pretrain_network_g"]
            cfg["path"]["pretrain_network_g"] = str(
                PROJECT_ROOT / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}"
            )


# ---------------------------------------------------------------------------
# NAFNet (multi-task: SR, denoising, deblurring)
# ---------------------------------------------------------------------------

class NAFNet(BasicSRModel):
    """[ECCV 2022] Nonlinear Activation Free Network for Image Restoration."""

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="nafnet",
            subtask=subtask,
            work_dir="NAFNet",
        )

    def _update_pretrained_ckpt(self, cfg: dict) -> None:
        if "path" in cfg and "pretrain_network_g" in cfg["path"]:
            ckpt_name = cfg["path"]["pretrain_network_g"]
            cfg["path"]["pretrain_network_g"] = str(
                PROJECT_ROOT / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}"
            )


class NAFNet_2x(BasicSRModel):
    """NAFNet 2× SR variant."""

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="nafnet_2x",
            subtask=subtask,
            work_dir="NAFNet",
            script_rel_path=str(Path("nafnet") / "inference.py"),
        )

    def _update_pretrained_ckpt(self, cfg: dict) -> None:
        if "path" in cfg and "pretrain_network_g" in cfg["path"]:
            ckpt_name = cfg["path"]["pretrain_network_g"]
            cfg["path"]["pretrain_network_g"] = str(
                PROJECT_ROOT / f"pretrained_ckpts/{self.work_dir_name}/{ckpt_name}"
            )


# ---------------------------------------------------------------------------
# SwinIR (multi-task: SR, denoising, JPEG artifact removal)
# ---------------------------------------------------------------------------

class SwinIR(Tool):
    """
    [ICCVW 2021] SwinIR – Image Restoration Using Swin Transformer.
    Multi-task: real SR (4×), denoising (σ=15/50), JPEG CAR (q=40).
    """

    _OPT_DICT = {
        "super_resolution": (
            "real_sr",
            {
                "gan": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
                "psnr": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth",
            },
        ),
        "denoising": (
            "color_dn",
            {
                "15": "005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
                "50": "005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth",
            },
        ),
        "artifact_removal": (
            "color_jpeg_car",
            {"40": "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth"},
        ),
    }

    def __init__(self, subtask: str, pretrained_on: str):
        super().__init__(
            tool_name=f"swinir_{pretrained_on}",
            subtask=subtask,
            work_dir="SwinIR",
            script_rel_path="infer_swinir_4kagent.py",
        )
        self.opt_task, opt_model_names = self._OPT_DICT[subtask]
        self.model_name = opt_model_names[pretrained_on]

    def _get_cmd_opts(self) -> list:
        opts = [
            "--task", self.opt_task,
            "--model_path", str(PROJECT_ROOT / f"pretrained_ckpts/SwinIR/{self.model_name}"),
            "--folder_lq", str(self.input_path.parent),
            "--save_dir", str(self.output_path.parent),
        ]
        if self.subtask == "super_resolution":
            opts += ["--scale", "4", "--tile", "512", "--tile_overlap", "32", "--large_model"]
        elif self.subtask == "artifact_removal":
            opts += ["--tile", "896", "--tile_overlap", "56"]
        else:
            opts += ["--tile", "1024", "--tile_overlap", "64"]
        return opts


# ---------------------------------------------------------------------------
# Restormer (denoising, motion deblurring, defocus deblurring, deraining)
# ---------------------------------------------------------------------------

class Restormer(Tool):
    """
    [CVPR 2022] Restormer – Efficient Transformer for High-Resolution
    Image Restoration. Best in class for satellite denoising / deblurring.
    """

    _OPT_DICT = {
        "denoising": "Real_Denoising",
        "deblurring": "Motion_Deblurring",
        "deraining": "Deraining",
    }

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="restormer",
            subtask=subtask,
            work_dir="Restormer",
            script_rel_path="infer_restormer_4kagent.py",
        )
        self.opt_task = self._OPT_DICT.get(subtask, "Real_Denoising")

    def _get_cmd_opts(self) -> list:
        return [
            "--task", self.opt_task,
            "--input_dir", str(self.input_path.parent),
            "--result_dir", str(self.output_path.parent),
            "--tile", "1024",
            "--tile_overlap", "64",
            "--ckpt", str(
                PROJECT_ROOT / f"pretrained_ckpts/Restormer/{self.opt_task.lower()}.pth"
            ),
        ]

    def _postprocess(self) -> None:
        outputs = list(self.output_path.parent.glob("*/*"))
        if outputs:
            outputs[0].replace(self.output_path)
            outputs[0].parent.rmdir()


# ---------------------------------------------------------------------------
# MPRNet (denoising, motion deblurring, deraining)
# ---------------------------------------------------------------------------

class MPRNet(Tool):
    """[CVPR 2021] Multi-Stage Progressive Image Restoration."""

    _OPT_DICT = {
        "denoising": "Denoising",
        "deblurring": "Deblurring",
        "deraining": "Deraining",
    }

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="mprnet",
            subtask=subtask,
            work_dir="MPRNet",
            script_rel_path="infer_mprnet_4kagent.py",
        )
        self.opt_task = self._OPT_DICT.get(subtask, "Denoising")

    def _get_cmd_opts(self) -> list:
        return [
            "--task", self.opt_task,
            "--input_dir", str(self.input_path.parent),
            "--result_dir", str(self.output_path.parent),
            "--model_arch_file",
            str(PROJECT_ROOT / f"executor/denoising/tools/MPRNet/{self.opt_task}/MPRNet.py"),
            "--ckpt",
            str(PROJECT_ROOT / f"pretrained_ckpts/MPRNet/model_{self.opt_task.lower()}.pth"),
            "--tile", "1024",
            "--tile_overlap", "64",
        ]


# ---------------------------------------------------------------------------
# MAXIM (denoising, dehazing, deblurring, deraining)
# ---------------------------------------------------------------------------

class MAXIM(Tool):
    """[CVPR 2022] Multi-Axis MLP for Image Processing."""

    _OPT_DICT = {
        "dehazing": ("Dehazing", "maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz"),
        "denoising": ("Denoising", "maxim_ckpt_Denoising_SIDD_checkpoint.npz"),
        "deblurring": ("Deblurring", "maxim_ckpt_Deblurring_RealBlur_R_checkpoint.npz"),
        "deraining": ("Deraining", "maxim_ckpt_Deraining_Rain13k_checkpoint.npz"),
    }

    def __init__(self, subtask: str):
        super().__init__(
            tool_name="maxim",
            subtask=subtask,
            work_dir="maxim",
            script_rel_path=str(Path("maxim") / "infer_maxim_4kagent.py"),
        )
        self.opt_task, self.opt_ckpt_name = self._OPT_DICT.get(
            subtask, ("Denoising", "maxim_ckpt_Denoising_SIDD_checkpoint.npz")
        )

    def _preprocess(self) -> None:
        """MAXIM requires input/{img} structure."""
        img_name = self.input_path.name
        rqd_input_dir = self.input_path.parent / "input"
        self.rqd_input_dir = rqd_input_dir
        rqd_input_dir.mkdir(exist_ok=True)
        shutil.copy(self.input_path, rqd_input_dir / img_name)

    def _get_cmd_opts(self) -> list:
        return [
            "--task", self.opt_task,
            "--ckpt_path", str(PROJECT_ROOT / f"pretrained_ckpts/MAXIM/{self.opt_ckpt_name}"),
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--has_target=False",
        ]

    def _postprocess(self) -> None:
        if hasattr(self, "rqd_input_dir") and self.rqd_input_dir.exists():
            shutil.rmtree(self.rqd_input_dir)
