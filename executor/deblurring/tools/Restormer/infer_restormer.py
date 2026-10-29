#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _load_runtime_proxy():
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "sata" / "executor" / "runtime"
        if candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            from native_proxy import parse_common_args, run_backend_with_opencv_fallback
            return parse_common_args, run_backend_with_opencv_fallback
    raise RuntimeError("Could not locate sata/executor/runtime/native_proxy.py")


def _find_sata_executor_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "sata" / "executor"
        if cand.exists():
            return cand
    raise RuntimeError("Could not locate /sata/executor root")


def main() -> None:
    parse_common_args, run_backend_with_opencv_fallback = _load_runtime_proxy()
    _ns, _extras, src, dst, scale = parse_common_args(default_scale=1)

    ex_root = _find_sata_executor_root()
    backend = (ex_root / "deblurring/tools/Restormer/infer_restormer.py").resolve()

    # ---------------------------------------------------------
    # THE FIX: Inject memory-paging (tiling) arguments directly 
    # into the Restormer execution command to prevent VRAM hangs.
    # ---------------------------------------------------------
    extra_args = [
        "--tile", "256", 
        "--tile_overlap", "16"
    ]

    extra_args = [str(scale) if x == "{scale}" else x for x in extra_args]

    run_backend_with_opencv_fallback(
        backend_script=backend,
        src=src,
        dst=dst,
        task="deblurring",
        scale=scale,
        variant="xrestormer",
        extra_args=extra_args,
        cwd=backend.parent,
    )


if __name__ == "__main__":
    main()