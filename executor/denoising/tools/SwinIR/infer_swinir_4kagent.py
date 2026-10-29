#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

def _load_runtime_proxy():
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "sata" / "executor" / "runtime"
        if candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            from native_proxy import parse_common_args
            return parse_common_args
    raise RuntimeError("Could not locate sata/executor/runtime/native_proxy.py")

def main() -> None:
    parse_common_args = _load_runtime_proxy()
    # Extract the standard SatAgent inputs
    _ns, _extras, src, dst, scale = parse_common_args(default_scale=4)

    backend_env = os.environ.get("SATA_SWINIR_SR_BACKEND", "").strip()
    if not backend_env:
        raise RuntimeError("SATA_SWINIR_SR_BACKEND not set in environment.")
    backend = Path(backend_env).expanduser().resolve()

    # The verified path to your large GAN model
    model_path = Path("/home/project/satagent/sata/pretrained_ckpts/SwinIR/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")

    print(f"[*] Running SwinIR SR on {src.name}")
    
    # SwinIR requires a folder input, so we trick it using a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "in"
        tmp_in.mkdir()
        shutil.copy(src, tmp_in / src.name)

        cmd = [
            "python3", str(backend),
            "--task", "real_sr",
            "--scale", str(scale),
            "--large_model",
            "--model_path", str(model_path),
            "--folder_lq", str(tmp_in),
            # Tiling prevents the GPU from running out of memory on large satellite images
            # "--tile", "512",
            # "--tile_overlap", "32"
            "--tile", "256",
            "--tile_overlap", "16"
        ]

        print(f"[*] Executing CMD: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=backend.parent, check=True)

        # SwinIR drops outputs into its own 'results' folder. Let's hunt down the newest one.
        results_dir = backend.parent / "results"
        
        # SwinIR usually appends "_SwinIR" to the filename
        candidates = list(results_dir.rglob("*.png"))
        if not candidates:
            raise RuntimeError("SwinIR completed but no output PNG was found in the results directory.")

        # Grab the most recently created file in the results dir
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        newest_out = candidates[0]

        # Deliver it to SatAgent's requested destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(newest_out, dst)
        print(f"[*] Successfully saved deep-learning upscale to {dst}")

if __name__ == "__main__":
    main()