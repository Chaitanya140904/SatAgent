from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


class PerceptionConfig(BaseModel):
    vlm_model_id: str = Field(default="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit")
    max_new_tokens: int = Field(default=1024)


class RestorationConfig(BaseModel):
    preference: str = Field(..., pattern="^(Fidelity|Perception|Hybrid)$")
    target_resolution: List[int] = Field(default=[4096, 4096])
    allowed_agenda: List[str]
    # Optional: enable Phase-0 ROI masking for Local-Path tasks (default: True)
    enable_roi_masking: bool = Field(default=True)
    # Optional: enable Phase-3B multi-temporal fusion (requires multi-date input)
    enable_temporal_fusion: bool = Field(default=False)


class QAConfig(BaseModel):
    """Configuration for Agent 3: QA Engine / Rollback Architecture."""
    # Maximum NIQE degradation allowed before triggering rollback (lower = stricter)
    niqe_rollback_threshold: float = Field(default=2.0)
    # Minimum spatial entropy delta before triggering rollback
    entropy_rollback_threshold: float = Field(default=0.15)
    # Dedicated entropy allowance for super-resolution, where denoising-like
    # structural regularization can legitimately reduce histogram entropy.
    super_resolution_entropy_allowance: float = Field(default=0.75)
    # Whether to attempt fallback model on rollback (vs. skip the step entirely)
    attempt_fallback_on_rollback: bool = Field(default=True)
    # Maximum number of rollback retries per task
    max_rollback_retries: int = Field(default=1)
    # Validation/demo override: force-approve every step after metric extraction.
    force_approve_all_steps: bool = Field(default=False)


class ProfileSchema(BaseModel):
    profile_name: str
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    restoration: RestorationConfig
    qa: QAConfig = Field(default_factory=QAConfig)
