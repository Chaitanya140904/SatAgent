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
            from native_proxy import parse_common_args, run_backend
            return parse_common_args, run_backend
    raise RuntimeError("Could not locate sata/executor/runtime/native_proxy.py")


def main() -> None:
    parse_common_args, run_backend = _load_runtime_proxy()
    _ns, _extras, src, dst, scale = parse_common_args(default_scale=1)

    here = Path(__file__).resolve()
    backend = None
    for parent in here.parents:
        cand = parent / "sata" / "executor" / "runtime" / "opencv_native_backend.py"
        if cand.exists():
            backend = cand
            break
    if backend is None:
        raise RuntimeError("Could not locate sata/executor/runtime/opencv_native_backend.py")

    extra_args = [
        "--task", "pan_sharpen",
        "--scale", str(scale),
        "--variant", "pan_mamba",
    ]
    run_backend(
        backend_script=backend,
        src=src,
        dst=dst,
        extra_args=extra_args,
        cwd=backend.parent,
    )


if __name__ == "__main__":
    main()
