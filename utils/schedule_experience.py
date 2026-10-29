from __future__ import annotations
"""
Schedule experience memory module.
Stores empirically-derived rules on optimal task ordering
from 4KAgent's experimental evidence, adapted for satellite-domain tasks.
"""
import json
from pathlib import Path
from typing import Optional

# Distilled ordering rules from 4KAgent's memory/schedule_experience.json
# + satellite-domain extensions from the SatAgent MoE spec.
DISTILLED_EXPERIENCE = """
Based on collected experience, the following ordering rules apply:

1. **Denoising before Brightening**: Denoise before brightening (32% vs 35% fail rate).
2. **Defocus Deblurring before Dehazing**: Deblur first (18% vs 20% fail rate).
3. **Artifact Removal before Deblurring**: Remove JPEG artifacts before deblurring (20% vs 28%).
4. **Motion Deblurring before Super-Resolution**: Deblur before SR (16% vs 19% fail rate).
5. **Deraining before Dehazing**: Derain first (21% vs 24% fail rate).
6. **Deraining before Super-Resolution**: Derain first (14% vs 32% fail rate).

Satellite-domain physics rules (higher priority):
7. **Atmospheric clearance first**: Cloud removal, dehazing, and shadow correction must precede
   all sensor corrections. Photons traverse the atmosphere before reaching the sensor.
8. **Sensor corrections before fusion**: Denoising, deblurring, destriping must complete
   before pan-sharpening or temporal fusion.
9. **Compression artifact removal before spectral fusion**: Propagating codec artifacts
   into pan-sharpened products is irreversible.
10. **Super-Resolution is ALWAYS last**: Upscaling amplifies every prior artifact.
    Never run SR before any restoration step.
"""


def load_experience(path: Optional[Path] = None) -> str:
    """
    Load scheduling experience.
    If a path is given and exists, loads from JSON (4KAgent format).
    Otherwise returns the built-in distilled satellite experience.
    """
    if path is not None and Path(path).exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("distilled", DISTILLED_EXPERIENCE)
        except Exception:
            pass
    return DISTILLED_EXPERIENCE
