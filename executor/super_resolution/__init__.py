from __future__ import annotations
"""
Super Resolution Toolbox – Phase 4 of the SatAgent MoE pipeline.

Routing: GLOBAL (full-frame)
Dataset: UC Merced Land Use, DOTA (downsampled GT)

Models integrated (from 4KAgent SR toolbox + satellite-specific):
  Fidelity  : hat_psnr, swinfir, rep_mamba, prithvi, nafnet_4x, swinir_psnr, hma
  Perception : diffbir, osediff, pisasr, satdiff, transenet

Scale support: 4× (primary), 2× (intermediate), 16× (extreme upscale)
"""
from pathlib import Path
from shutil import rmtree

from ..tool import Tool
from ..multitask_tools import BasicSRModel, XRestormer, NAFNet, NAFNet_2x, SwinIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
subtask = "super_resolution"


# ── Fidelity path – HAT (CVPR 2023) ─────────────────────────────────────────

class HATPSNR(BasicSRModel):
    """
    [CVPR 2023] HAT – Activating More Pixels in Image Super-Resolution Transformer.
    PSNR-optimised weights. Strong fidelity baseline for satellite imagery.
    """
    def __init__(self):
        super().__init__(
            tool_name="hat_psnr", subtask=subtask,
            work_dir="HAT", script_rel_path=str(Path("hat") / "infer_hat_4kagent.py")
        )
    def _update_pretrained_ckpt(self, cfg: dict) -> None:
        if "path" in cfg and "pretrain_network_g" in cfg["path"]:
            cfg["path"]["pretrain_network_g"] = str(
                PROJECT_ROOT / f"pretrained_ckpts/{self.work_dir_name}/{cfg['path']['pretrain_network_g']}"
            )


class HATGAN(BasicSRModel):
    """HAT GAN-variant for perceptual quality."""
    def __init__(self):
        super().__init__(
            tool_name="hat_gan", subtask=subtask,
            work_dir="HAT", script_rel_path=str(Path("hat") / "infer_hat_4kagent.py")
        )
    def _update_pretrained_ckpt(self, cfg: dict) -> None:
        if "path" in cfg and "pretrain_network_g" in cfg["path"]:
            cfg["path"]["pretrain_network_g"] = str(
                PROJECT_ROOT / f"pretrained_ckpts/{self.work_dir_name}/{cfg['path']['pretrain_network_g']}"
            )


class HATPSNR_2x(BasicSRModel):
    """HAT 2× fidelity (intermediate upscale step)."""
    def __init__(self):
        super().__init__(
            tool_name="hat_psnr_2x", subtask=subtask,
            work_dir="HAT", script_rel_path=str(Path("hat") / "infer_hat_4kagent.py")
        )
    def _update_pretrained_ckpt(self, cfg: dict) -> None:
        if "path" in cfg and "pretrain_network_g" in cfg["path"]:
            cfg["path"]["pretrain_network_g"] = str(
                PROJECT_ROOT / f"pretrained_ckpts/{self.work_dir_name}/{cfg['path']['pretrain_network_g']}"
            )


# ── Fidelity path – SwinFIR (NeurIPS 2024) ───────────────────────────────────

class SwinFIR(Tool):
    """[NeurIPS 2024] SwinFIR – Frequency-domain Swin Transformer for SR."""
    def __init__(self):
        super().__init__(
            tool_name="swinfir", subtask=subtask,
            work_dir="SwinFIR", script_rel_path="infer_swinfir_4kagent.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input", str(self.input_path.parent),
            "--output", str(self.output_path.parent),
            "--model_path", str(PROJECT_ROOT / "pretrained_ckpts/SwinFIR/SwinFIR_SRx4.pth"),
            "--scale", "4",
        ]


class SwinFIR_2x(Tool):
    """SwinFIR 2× variant."""
    def __init__(self):
        super().__init__(
            tool_name="swinfir_2x", subtask=subtask,
            work_dir="SwinFIR", script_rel_path="infer_swinfir_4kagent.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input", str(self.input_path.parent),
            "--output", str(self.output_path.parent),
            "--model_path", str(PROJECT_ROOT / "pretrained_ckpts/SwinFIR/SwinFIR_SRx2.pth"),
            "--scale", "2",
        ]


# ── Fidelity path – HMA (CVPRW 2024) ─────────────────────────────────────────

class HMA(BasicSRModel):
    """[CVPRW 2024] HMANet – Hybrid Multi-Axis Aggregation for Image SR."""
    def __init__(self):
        super().__init__(
            tool_name="hma", subtask=subtask,
            work_dir="HMA", script_rel_path=str(Path("hma") / "infer_hma_4kagent.py")
        )
    def _update_pretrained_ckpt(self, cfg: dict) -> None:
        if "path" in cfg and "pretrain_network_g" in cfg["path"]:
            cfg["path"]["pretrain_network_g"] = str(
                PROJECT_ROOT / f"pretrained_ckpts/{self.work_dir_name}/{cfg['path']['pretrain_network_g']}"
            )


# ── Satellite-specific Fidelity – Rep-Mamba & Prithvi ────────────────────────

class RepMamba(Tool):
    """
    Rep-Mamba – State-Space Model SR optimised for satellite PSNR/SSIM.
    Evaluated on downsampled UC Merced / DOTA.
    """
    def __init__(self):
        super().__init__(
            tool_name="rep_mamba", subtask=subtask,
            work_dir="RepMamba", script_rel_path="infer_repmamba_sat.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/RepMamba/rep_mamba_sat_4x.pth"),
            "--scale", "4",
            "--tile", "512",
        ]


class Prithvi(Tool):
    """
    IBM/NASA Prithvi – Remote-sensing foundational model fine-tuned for SR.
    Best spectral fidelity (SAM) on multispectral satellite data.
    """
    def __init__(self):
        super().__init__(
            tool_name="prithvi", subtask=subtask,
            work_dir="Prithvi", script_rel_path="infer_prithvi_sr.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/Prithvi/prithvi_sr_4x.pth"),
            "--scale", "4",
        ]


# ── Perception path – DiffBIR (ECCV 2024) ────────────────────────────────────

class DiffBIR(Tool):
    """
    [ECCV 2024] DiffBIR – Blind Image Restoration with Generative Diffusion Prior.
    Top perceptual quality; uses tiled inference for 4K compatibility.
    """
    def __init__(self):
        super().__init__(
            tool_name="diffbir", subtask=subtask,
            work_dir="DiffBIR", script_rel_path="infer_diffbir_4kagent.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--task", "sr",
            "--input", str(self.input_path.parent),
            "--pos_prompt", "high-resolution satellite image, clear terrain details",
            "--neg_prompt", "low quality, blurry, noise, artifacts, haze",
            "--sampler", "spaced",
            "--version", "v2",
            "--steps", "50",
            "--upscale", "4",
            "--cfg_scale", "4",
            "--captioner", "none",
            "--output", str(self.output_path.parent),
            "--device", "cuda",
            "--cleaner_tiled",
            "--vae_encoder_tiled",
            "--vae_decoder_tiled",
            "--cldm_tiled",
        ]


class DiffBIR_2x(Tool):
    """DiffBIR 2× variant."""
    def __init__(self):
        super().__init__(
            tool_name="diffbir_2x", subtask=subtask,
            work_dir="DiffBIR", script_rel_path="infer_diffbir_4kagent.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--task", "sr",
            "--input", str(self.input_path.parent),
            "--pos_prompt", "high-resolution satellite image",
            "--neg_prompt", "low quality, blurry, noise",
            "--sampler", "spaced", "--version", "v2",
            "--steps", "50", "--upscale", "2", "--cfg_scale", "4",
            "--captioner", "none",
            "--output", str(self.output_path.parent), "--device", "cuda",
            "--cleaner_tiled", "--vae_encoder_tiled",
            "--vae_decoder_tiled", "--cldm_tiled",
        ]


# ── Perception path – OSEDiff (NeurIPS 2024) ─────────────────────────────────

class OSEDiff(Tool):
    """[NeurIPS 2024] OSEDiff – One-Step Effective Diffusion SR."""
    def __init__(self):
        super().__init__(
            tool_name="osediff", subtask=subtask,
            work_dir="OSEDiff", script_rel_path="infer_osediff_4kagent.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--i", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--osediff_path", str(PROJECT_ROOT / "executor/super_resolution/tools/OSEDiff/preset/models/osediff.pkl"),
            "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-1-base",
            "--ram_ft_path", str(PROJECT_ROOT / "pretrained_ckpts/OSEDiff/DAPE/DAPE.pth"),
            "--ram_path", str(PROJECT_ROOT / "pretrained_ckpts/OSEDiff/RAM/ram_swin_large_14m.pth"),
            "--vae_decoder_tiled_size", "170",
        ]


# ── Perception path – PiSA-SR (NeurIPS 2024) ─────────────────────────────────

class PISASR(Tool):
    """[NeurIPS 2024] PiSA-SR – Perceptual Image Super-Alignment SR."""
    def __init__(self):
        super().__init__(
            tool_name="pisasr", subtask=subtask,
            work_dir="PiSA-SR", script_rel_path="infer_pisasr_4kagent.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_image", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--pretrained_path", str(PROJECT_ROOT / "pretrained_ckpts/PiSA-SR/pisa_sr.pkl"),
            "--pretrained_model_path", "stabilityai/stable-diffusion-2-1-base",
            "--vae_decoder_tiled_size", "170",
            "--default",
        ]


# ── Perception path – Satellite-specific: SatDiff & TransENet ────────────────

class SatDiff(Tool):
    """
    SatDiff – Satellite-conditioned diffusion SR.
    Evaluated on DOTA / xView; optimised for NIQE + FID.
    """
    def __init__(self):
        super().__init__(
            tool_name="satdiff", subtask=subtask,
            work_dir="SatDiff", script_rel_path="infer_satdiff.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/SatDiff/satdiff_4x.pth"),
            "--scale", "4",
        ]


class TransENet(Tool):
    """
    TransENet – Transformer-Enhanced SR for remote-sensing images.
    """
    def __init__(self):
        super().__init__(
            tool_name="transenet", subtask=subtask,
            work_dir="TransENet", script_rel_path="infer_transenet.py"
        )
    def _get_cmd_opts(self) -> list:
        return [
            "--input_dir", str(self.input_path.parent),
            "--output_dir", str(self.output_path.parent),
            "--ckpt", str(PROJECT_ROOT / "pretrained_ckpts/TransENet/transenet_4x.pth"),
            "--scale", "4",
        ]


# ── Toolbox assemblies ────────────────────────────────────────────────────────

# 4× toolbox (primary – Fidelity first, Perception at end)
sr_toolbox = [
    # Fidelity tier
    HATPSNR(),
    SwinFIR(),
    HMA(),
    RepMamba(),
    Prithvi(),
    XRestormer(subtask=subtask),
    SwinIR(subtask=subtask, pretrained_on="psnr"),
    NAFNet(subtask=subtask),
    # Perception tier
    DiffBIR(),
    OSEDiff(),
    PISASR(),
    HATGAN(),
    SwinIR(subtask=subtask, pretrained_on="gan"),
    SatDiff(),
    TransENet(),
]

# 2× toolbox (intermediate upscale step for 8× / 16× targets)
sr_toolbox_2x = [
    HATPSNR_2x(),
    SwinFIR_2x(),
    NAFNet_2x(subtask=subtask),
    DiffBIR_2x(),
    OSEDiff(),
    SwinIR(subtask=subtask, pretrained_on="psnr"),
]
