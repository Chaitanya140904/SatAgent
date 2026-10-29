from typing import List

class UpscaleCalculator:
    """Calculates optimal SR scaling factor to reach 4K boundaries."""
    
    @staticmethod
    def get_target_factor(width: float, height: float, target_dim: int = 4000) -> int:
        max_edge = max(width, height)
        if max_edge >= target_dim:
            return 1
            
        valid_scales = [s for s in [2, 4, 8, 16] if max_edge * s >= target_dim]
        return min(valid_scales) if valid_scales else 16

    @staticmethod
    def append_sr_tasks(factor: int) -> List[str]:
        factor_map = {
            2: ["super-resolution (2x)"],
            4: ["super-resolution (4x)"],
            8: ["super-resolution (4x)", "super-resolution (2x)"],
            16: ["super-resolution (4x)", "super-resolution (4x)"],
        }
        return factor_map.get(factor, [])
