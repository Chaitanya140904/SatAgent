# # from __future__ import annotations
# # """
# # SatModelSelector – Maps identified degradation tasks to specific model weights
# # based on the user's Profile Mode (Fidelity | Perception | Hybrid).

# # Fidelity models: optimised for PSNR / SSIM / SAM (mathematical truth).
# # Perception models: optimised for NIQE / LPIPS / FID (visual realism).
# # Hybrid: falls back to the Fidelity model unless overridden.

# # Phase & Task mapping follows the full MoE spec:
# #   Phase 0  – ROI Masking        (pre-processor, no model preference)
# #   Phase 1A – Thick Cloud Removal
# #   Phase 1B – Global Dehazing
# #   Phase 1C – Shadow Correction
# #   Phase 2A – Thermal/ISO Denoising
# #   Phase 2B – Orbital Motion Deblurring
# #   Phase 2C – Destriping / Dead-Pixel Inpainting
# #   Phase 2D – Compression Artifact Removal
# #   Phase 3A – Pan-Sharpening
# #   Phase 3B – Multi-Temporal Fusion
# #   Phase 4  – Super-Resolution (4K Layer)
# # """

# # from typing import Optional, Dict, List


# # class SatModelSelector:
# #     """Maps identified tasks to specific model weights based on the user's Profile Mode."""

# #     # -----------------------------------------------------------------------
# #     # Tool-name strings here must match the `tool_name` attribute of the
# #     # corresponding Tool subclasses registered in executor/__init__.py.
# #     # -----------------------------------------------------------------------
# #     MODEL_TABLE: Dict[str, Dict[str, List[str]]] = {
# #         # Phase 1A – Thick Cloud Removal (Local Path / mask-dependent)
# #         "cloud removal": {
# #             "Fidelity":    ["dvpnet", "uncrtaints"],
# #             "Perception":  ["emrdm", "cloud_mamba"],
# #             "Hybrid":      ["dvpnet", "emrdm"],
# #         },
# #         # Phase 1B – Global Dehazing (Global Path)
# #         "dehazing": {
# #             "Fidelity":    ["rsdehamba"],
# #             "Perception":  ["dehazing_diffgan", "ridcp"],
# #             "Hybrid":      ["rsdehamba", "dehazing_diffgan"],
# #         },
# #         # Phase 1C – Shadow Correction (Local Path / mask-dependent)
# #         "shadow correction": {
# #             "Fidelity":    ["shadowformer"],
# #             "Perception":  ["enlighten_gan_rs"],
# #             "Hybrid":      ["shadowformer", "enlighten_gan_rs"],
# #         },
# #         # Phase 2A – Thermal/ISO Denoising (Global Path)
# #         "sensor denoising": {
# #             "Fidelity":    ["satmae", "nafnet_rs"],
# #             "Perception":  ["diffusionsat"],
# #             "Hybrid":      ["nafnet_rs", "diffusionsat"],
# #         },
# #         # Phase 2B – Orbital Motion Deblurring (Global Path)
# #         "sensor deblurring": {
# #             "Fidelity":    ["pirn", "stripformer"],
# #             "Perception":  ["blur_mamba", "deblurdiff"],
# #             "Hybrid":      ["stripformer", "blur_mamba"],
# #         },
# #         # Phase 2C – Destriping / Dead-Pixel Inpainting (Local Path)
# #         "destriping": {
# #             "Fidelity":    ["qgcn", "sst_net"],
# #             "Perception":  ["repaint_rs"],
# #             "Hybrid":      ["sst_net", "repaint_rs"],
# #         },
# #         # Phase 2D – Compression Artifact Removal (Global Path)
# #         "compression artifact removal": {
# #             "Fidelity":    ["fbcnn_blind", "nafnet_jpeg"],
# #             "Perception":  ["swinir_car"],
# #             "Hybrid":      ["nafnet_jpeg", "swinir_car"],
# #         },
# #         # Phase 3A – Pan-Sharpening (Global Path)
# #         "pan-sharpening": {
# #             "Fidelity":    ["pan_mamba", "adwm"],
# #             "Perception":  ["mgdn", "premix"],
# #             "Hybrid":      ["adwm", "mgdn"],
# #         },
# #         # Alias for VLM plan outputs that say "fusion"
# #         "fusion": {
# #             "Fidelity":    ["pan_mamba", "adwm"],
# #             "Perception":  ["mgdn", "premix"],
# #             "Hybrid":      ["adwm", "mgdn"],
# #         },
# #         # Phase 3B – Multi-Temporal Fusion (Global Path)
# #         "multi-temporal fusion": {
# #             "Fidelity":    ["transfusion", "timesformer_rs"],
# #             "Perception":  ["timesformer_rs", "transfusion"],
# #             "Hybrid":      ["transfusion", "timesformer_rs"],
# #         },
# #         # Phase 4 – Super-Resolution 4K Layer (Global Path)
# #         "super-resolution": {
# #             "Fidelity":    ["hat_psnr", "rep_mamba", "prithvi"],
# #             "Perception":  ["satdiff", "transenet"],
# #             "Hybrid":      ["hat_psnr", "satdiff"],
# #         },
# #         # VLM alias forms
# #         "super-resolution (2x)": {
# #             "Fidelity":    ["hat_psnr"],
# #             "Perception":  ["diffbir"],
# #             "Hybrid":      ["swinfir", "hat_psnr"],
# #         },
# #         "super-resolution (4x)": {
# #             "Fidelity":    ["hat_psnr", "rep_mamba", "prithvi"],
# #             "Perception":  ["satdiff", "transenet"],
# #             "Hybrid":      ["hat_psnr", "satdiff"],
# #         },
# #     }

# #     # Fallback masking tool used by Phase 0 based on the primary task type
# #     MASK_TOOL_TABLE: dict[str, str] = {
# #         "cloud removal":    "cloud_net_plus",
# #         "shadow correction": "rs_sam",
# #         "destriping":       "yolov8_obb",
# #     }

# #     @classmethod
# #     def get_model(cls, task: str, mode: str) -> str:
# #         """
# #         Returns the executor tool_name for the given task and mode.
# #         Falls back to 'UnknownModel' if the task is not in the table.
# #         """
# #         task_lower = task.lower().strip()
# #         task_models = cls.MODEL_TABLE.get(task_lower, {})
# #         candidates = task_models.get(mode) or task_models.get("Fidelity") or []
# #         return candidates[0] if candidates else "UnknownModel"

# #     @classmethod
# #     def get_model_candidates(cls, task: str) -> Dict[str, List[str]]:
# #         """
# #         Returns all configured candidate models for each execution mode.
# #         """
# #         task_lower = task.lower().strip()
# #         task_models = cls.MODEL_TABLE.get(task_lower, {})
# #         return {
# #             "Fidelity": list(task_models.get("Fidelity", [])),
# #             "Perception": list(task_models.get("Perception", [])),
# #             "Hybrid": list(task_models.get("Hybrid", [])),
# #         }

# #     @classmethod
# #     def get_mask_tool(cls, task: str) -> Optional[str]:
# #         """
# #         Returns the Phase-0 ROI masking tool name for Local-Path tasks,
# #         or None if the task follows the Global Path.
# #         """
# #         return cls.MASK_TOOL_TABLE.get(task.lower().strip(), None)

# #     @classmethod
# #     def get_fallback_model(cls, task: str, failed_model: str, mode: str) -> Optional[str]:
# #         """
# #         QA Rollback: returns an alternate model for the same task after a failure.
# #         Fallbacks are restricted to the active execution mode to avoid
# #         profile-crossing behavior during strict runs.
# #         """
# #         task_lower = task.lower().strip()
# #         task_models = cls.MODEL_TABLE.get(task_lower, {})
# #         mode_models = list(task_models.get(mode, []))
# #         if not mode_models:
# #             mode_models = list(task_models.get("Fidelity", []))
# #         for model in mode_models:
# #             if model != failed_model:
# #                 return model
# #         return None
# from __future__ import annotations
# """
# SatModelSelector – Maps identified degradation tasks to specific model weights
# based on the user's Profile Mode (Fidelity | Perception | Hybrid).

# Fidelity models: optimised for PSNR / SSIM / SAM (mathematical truth).
# Perception models: optimised for NIQE / LPIPS / FID (visual realism).
# Hybrid: falls back to the Fidelity model unless overridden.

# Phase & Task mapping follows the full MoE spec:
#   Phase 0  – ROI Masking        (pre-processor, no model preference)
#   Phase 1A – Thick Cloud Removal
#   Phase 1B – Global Dehazing
#   Phase 1C – Shadow Correction
#   Phase 2A – Thermal/ISO Denoising
#   Phase 2B – Orbital Motion Deblurring
#   Phase 2C – Destriping / Dead-Pixel Inpainting
#   Phase 2D – Compression Artifact Removal
#   Phase 3A – Pan-Sharpening
#   Phase 3B – Multi-Temporal Fusion
#   Phase 4  – Super-Resolution (4K Layer)
# """

# from typing import Optional, Dict, List


# class SatModelSelector:
#     """Maps identified tasks to specific model weights based on the user's Profile Mode."""

#     # -----------------------------------------------------------------------
#     # Tool-name strings here must match the `tool_name` attribute of the
#     # corresponding Tool subclasses registered in executor/__init__.py.
#     # -----------------------------------------------------------------------
#     MODEL_TABLE: Dict[str, Dict[str, List[str]]] = {
#         # Phase 1A – Thick Cloud Removal (Local Path / mask-dependent)
#         "cloud removal": {
#             "Fidelity":    ["dvpnet", "uncrtaints"],
#             "Perception":  ["emrdm", "cloud_mamba"],
#             "Hybrid":      ["dvpnet", "emrdm"],
#         },
#         # Phase 1B – Global Dehazing (Global Path)
#         "dehazing": {
#             "Fidelity":    ["rsdehamba"],
#             "Perception":  ["dehazing_diffgan", "ridcp"],
#             "Hybrid":      ["rsdehamba", "dehazing_diffgan"],
#         },
#         # Phase 1C – Shadow Correction (Local Path / mask-dependent)
#         "shadow correction": {
#             "Fidelity":    ["shadowformer"],
#             "Perception":  ["enlighten_gan_rs"],
#             "Hybrid":      ["shadowformer", "enlighten_gan_rs"],
#         },
#         # Phase 2A – Thermal/ISO Denoising (Global Path)
#         "sensor denoising": {
#             "Fidelity":    ["satmae", "nafnet_rs"],
#             "Perception":  ["satmae"],
#             "Hybrid":      ["nafnet_rs", "satmae"],
#         },
#         # Phase 2B – Orbital Motion Deblurring (Global Path)
#         "sensor deblurring": {
#             "Fidelity":    ["mprnet", "restormer"],
#             "Perception":  ["restormer", "mprnet"],
#             "Hybrid":      ["restormer", "mprnet"],
#         },
#         # Phase 2C – Destriping / Dead-Pixel Inpainting (Local Path)
#         "destriping": {
#             "Fidelity":    ["qgcn", "sst_net"],
#             "Perception":  ["repaint_rs"],
#             "Hybrid":      ["sst_net", "repaint_rs"],
#         },
#         # Phase 2D – Compression Artifact Removal (Global Path)
#         "compression artifact removal": {
#             "Fidelity":    ["nafnet_jpeg"],
#             "Perception":  ["swinir_car"],
#             "Hybrid":      ["nafnet_jpeg", "swinir_car"],
#         },
#         # Phase 3A – Pan-Sharpening (Global Path)
#         "pan-sharpening": {
#             "Fidelity":    ["pan_mamba", "adwm"],
#             "Perception":  ["mgdn", "premix"],
#             "Hybrid":      ["adwm", "mgdn"],
#         },
#         # Alias for VLM plan outputs that say "fusion"
#         "fusion": {
#             "Fidelity":    ["pan_mamba", "adwm"],
#             "Perception":  ["mgdn", "premix"],
#             "Hybrid":      ["adwm", "mgdn"],
#         },
#         # Phase 3B – Multi-Temporal Fusion (Global Path)
#         "multi-temporal fusion": {
#             "Fidelity":    ["transfusion", "timesformer_rs"],
#             "Perception":  ["timesformer_rs", "transfusion"],
#             "Hybrid":      ["transfusion", "timesformer_rs"],
#         },
#         # Phase 4 – Super-Resolution 4K Layer (Global Path)
#         "super-resolution": {
#             "Fidelity":    ["swinir_psnr"],
#             "Perception":  ["swinir_gan"],
#             "Hybrid":      ["swinir_psnr"],
#         },
#         # VLM alias forms
#         "super-resolution (2x)": {
#             "Fidelity":    ["swinir_psnr"],
#             "Perception":  ["swinir_gan"],
#             "Hybrid":      ["swinir_psnr", "swinir_gan"],
#         },
#         "super-resolution (4x)": {
#             "Fidelity":    ["swinir_psnr"],
#             "Perception":  ["swinir_gan"],
#             "Hybrid":      ["swinir_psnr"],
#         },
#     }

#     # Fallback masking tool used by Phase 0 based on the primary task type
#     MASK_TOOL_TABLE: dict[str, str] = {
#         "cloud removal":    "cloud_net_plus",
#         "shadow correction": "rs_sam",
#         "destriping":       "yolov8_obb",
#     }

#     @classmethod
#     def get_model(cls, task: str, mode: str) -> str:
#         """
#         Returns the executor tool_name for the given task and mode.
#         Falls back to 'UnknownModel' if the task is not in the table.
#         """
#         task_lower = task.lower().strip()
#         task_models = cls.MODEL_TABLE.get(task_lower, {})
#         candidates = task_models.get(mode) or task_models.get("Fidelity") or []
#         return candidates[0] if candidates else "UnknownModel"

#     @classmethod
#     def get_model_candidates(cls, task: str) -> Dict[str, List[str]]:
#         """
#         Returns all configured candidate models for each execution mode.
#         """
#         task_lower = task.lower().strip()
#         task_models = cls.MODEL_TABLE.get(task_lower, {})
#         return {
#             "Fidelity": list(task_models.get("Fidelity", [])),
#             "Perception": list(task_models.get("Perception", [])),
#             "Hybrid": list(task_models.get("Hybrid", [])),
#         }

#     @classmethod
#     def get_mask_tool(cls, task: str) -> Optional[str]:
#         """
#         Returns the Phase-0 ROI masking tool name for Local-Path tasks,
#         or None if the task follows the Global Path.
#         """
#         return cls.MASK_TOOL_TABLE.get(task.lower().strip(), None)

#     @classmethod
#     def get_fallback_model(cls, task: str, failed_model: str, mode: str) -> Optional[str]:
#         """
#         QA Rollback: returns an alternate model for the same task after a failure.
#         Fallbacks are restricted to the active execution mode to avoid
#         profile-crossing behavior during strict runs.
#         """
#         task_lower = task.lower().strip()
#         task_models = cls.MODEL_TABLE.get(task_lower, {})
#         mode_models = list(task_models.get(mode, []))
#         if not mode_models:
#             mode_models = list(task_models.get("Fidelity", []))
#         for model in mode_models:
#             if model != failed_model:
#                 return model
#         return None
from __future__ import annotations
"""
SatModelSelector – Maps identified degradation tasks to specific model weights
based on the user's Profile Mode (Fidelity | Perception | Hybrid).

REWRITTEN: 
1. Promoted RIDCP to primary to prevent DehazeFormer memory hangs.
2. Aligned with local .pth availability (SwinIR, NAFNet, Restormer, EMRDM).
3. Ensured 4K Super-Resolution mapping for Eq. 1 execution.
"""

from typing import Optional, Dict, List


class SatModelSelector:
    """Maps identified tasks to specific model weights based on the user's Profile Mode."""

    MODEL_TABLE: Dict[str, Dict[str, List[str]]] = {
        # Phase 1A – Thick Cloud Removal (Local Path / mask-dependent)
        "cloud removal": {
            "Fidelity":    ["dvpnet"],
            "Perception":  ["emrdm", "cloud_mamba"],
            "Hybrid":      ["emrdm"],
        },
        # Phase 1B – Global Dehazing (Global Path)
        "dehazing": {
            "Fidelity":    ["ridcp"],
            "Perception":  ["ridcp", "dehazing_diffgan"], # RIDCP is now the primary expert
            "Hybrid":      ["ridcp"],
        },
        # Phase 1C – Shadow Correction (Local Path / mask-dependent)
        "shadow correction": {
            "Fidelity":    ["shadowformer"],
            "Perception":  ["enlighten_gan_rs"],
            "Hybrid":      ["shadowformer"],
        },
        # Phase 2A – Thermal/ISO Denoising (Global Path)
        "sensor denoising": {
            "Fidelity":    ["satmae", "nafnet_rs"],
            "Perception":  ["satmae", "mprnet"],
            "Hybrid":      ["nafnet_rs"],
        },
        # Phase 2B – Orbital Motion Deblurring (Global Path)
        "sensor deblurring": {
            "Fidelity":    ["mprnet", "restormer"],
            "Perception":  ["restormer", "mprnet"], # Local .pth priority
            "Hybrid":      ["restormer"],
        },
        # Phase 2C – Destriping / Dead-Pixel Inpainting (Local Path)
        "destriping": {
            "Fidelity":    ["sst_net"],
            "Perception":  ["repaint_rs"],
            "Hybrid":      ["sst_net"],
        },
        # Phase 2D – Compression Artifact Removal (Global Path)
        "compression artifact removal": {
            "Fidelity":    ["nafnet_jpeg"],
            "Perception":  ["swinir_car"],
            "Hybrid":      ["nafnet_jpeg"],
        },
        # Phase 3A – Pan-Sharpening (Global Path)
        "pan-sharpening": {
            "Fidelity":    ["pan_mamba"],
            "Perception":  ["mgdn"],
            "Hybrid":      ["adwm"],
        },
        # Phase 4 – Super-Resolution 4K Layer (Global Path)
        "super-resolution": {
            "Fidelity":    ["swinir_psnr"],
            "Perception":  ["swinir_gan", "transenet"], # Locks SwinIR-GAN for 4KAgent
            "Hybrid":      ["swinir_gan"],
        },
        # VLM alias forms for dynamic scaling triggers
        "super-resolution (2x)": {
            "Fidelity":    ["swinir_psnr"],
            "Perception":  ["swinir_gan"],
            "Hybrid":      ["swinir_gan"],
        },
        "super-resolution (4x)": {
            "Fidelity":    ["swinir_psnr"],
            "Perception":  ["swinir_gan"],
            "Hybrid":      ["swinir_gan"],
        },
    }

    MASK_TOOL_TABLE: dict[str, str] = {
        "cloud removal":    "cloud_net_plus",
        "shadow correction": "rs_sam",
        "destriping":       "yolov8_obb",
    }

    @classmethod
    def get_model(cls, task: str, mode: str) -> str:
        task_lower = task.lower().strip()
        task_models = cls.MODEL_TABLE.get(task_lower, {})
        candidates = task_models.get(mode) or task_models.get("Fidelity") or []
        return candidates[0] if candidates else "UnknownModel"

    @classmethod
    def get_model_candidates(cls, task: str) -> Dict[str, List[str]]:
        task_lower = task.lower().strip()
        task_models = cls.MODEL_TABLE.get(task_lower, {})
        return {
            "Fidelity": list(task_models.get("Fidelity", [])),
            "Perception": list(task_models.get("Perception", [])),
            "Hybrid": list(task_models.get("Hybrid", [])),
        }

    @classmethod
    def get_mask_tool(cls, task: str) -> Optional[str]:
        return cls.MASK_TOOL_TABLE.get(task.lower().strip(), None)

    @classmethod
    def get_fallback_model(cls, task: str, failed_model: str, mode: str) -> Optional[str]:
        task_lower = task.lower().strip()
        task_models = cls.MODEL_TABLE.get(task_lower, {})
        mode_models = list(task_models.get(mode, []))
        if not mode_models:
            mode_models = list(task_models.get("Fidelity", []))
        for model in mode_models:
            if model != failed_model:
                return model
        return None