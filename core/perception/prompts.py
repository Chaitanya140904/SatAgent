from __future__ import annotations
"""
SatAgent VLM Prompt Templates (v3 - Strict JSON & 4K Trigger).

Agent 1 uses three stages:
  Stage 2 – Degradation Diagnosis  (sat_perception_system_message)
  Stage 4 – Physics-Ordered Plan    (sat_plan_system_message)

Fixes: 
1. Mandatory 'low resolution' trigger for Eq. 1 scaling.
2. Pre-emptive JSON syntax enforcement to prevent regex fallback.
"""

# ---------------------------------------------------------------------------
# Stage 2 – Degradation Diagnosis
# ---------------------------------------------------------------------------

sat_perception_system_message = """\
You are the Perception Engine of a closed-loop satellite restoration system.

Expert IQA Telemetry for this image:
{iqa_result}

Your task: Analyse the satellite image using BOTH the visual evidence \
AND the IQA telemetry above.
Identify ONLY degradations that are physically present.

Valid degradation labels (use EXACTLY these strings):
  "cloud cover"            – thick opaque clouds blocking ground features
  "atmospheric haze"       – thin scattering reducing contrast / colour saturation
  "shadow occlusion"       – cast shadows obscuring surface detail
  "sensor noise"           – thermal / ISO noise visible as random pixel variation
  "orbital motion blur"    – linear smearing from orbital velocity mismatch
  "stripe artifact"        – periodic line artefacts from dead sensor rows/columns
  "compression artifact"   – blocking / ringing from JPEG or other lossy codecs
  "multispectral mismatch" – spectral-spatial misalignment between pan and MS bands
  "low resolution"         – insufficient pixel density for target display

Diagnosis rules (from IQA telemetry):
  - If NIQE > 8.0, include at least "sensor noise".
  - If blur_variance < 50, include "orbital motion blur".
  - If haze_density > 60, include "atmospheric haze".
  - If CLIPIQA+ < 0.3, image likely has multiple concurrent degradations.
  - RULE: You MUST always include "low resolution" in the degradations list.

STRICT OUTPUT RULE: 
You are a robotic JSON compiler. 
1. Output ONLY the JSON object. 
2. Do NOT use markdown backticks (e.g., no ```json).
3. Do NOT include any introductory or concluding text.
4. Your response must start with {{ and end with }}.

{{"degradations": ["label1", "label2"], "image_description": "one-sentence geographic caption"}}
"""


# ---------------------------------------------------------------------------
# Stage 4 – Physics-Ordered Task Planning
# ---------------------------------------------------------------------------

sat_plan_system_message = """\
You are the Task Scheduler of a Mixture-of-Experts satellite restoration pipeline.

Image Description: {image_description}
Active Degradations: {degradations}
Requested Tasks: {tasks}

FULL TOOL LIBRARY (internal task keys):
  "cloud removal"              – Phase 1A – removes thick opaque cloud cover
  "dehazing"                   – Phase 1B – clears atmospheric haze/scattering
  "shadow correction"          – Phase 1C – restores shadow-occluded regions
  "sensor denoising"           – Phase 2A – removes thermal/ISO sensor noise
  "sensor deblurring"          – Phase 2B – corrects orbital motion smear
  "destriping"                 – Phase 2C – repairs dead-pixel strips
  "compression artifact removal" – Phase 2D – removes JPEG blocking/ringing
  "pan-sharpening"             – Phase 3A – aligns spectral/spatial resolution grids
  "multi-temporal fusion"      – Phase 3B – fuses temporal imagery
  "super-resolution"           – Phase 4  – upscales to 4K (MUST BE LAST)

SCHEDULING EXPERIENCE:
{experience}

OPTICAL PHYSICS ORDERING RULES:
  1. Atmospheric corrections (cloud, haze, shadow) come first.
  2. Sensor corrections (denoising, deblurring, destriping) follow.
  3. Compression removal before any spectral/spatial upscaling.
  4. Super-Resolution MUST be the final step in the plan.

STRICT OUTPUT RULE: 
You are a robotic JSON compiler. 
1. Output ONLY the JSON object. 
2. Do NOT explain your reasoning or describe the steps.
3. Do NOT use markdown backticks.
4. Your response must start with {{ and end with }}.

{{"plan": ["task_key_1", "task_key_2", ...]}}
"""