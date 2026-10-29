#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Optional
import os

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _first_image_in_dir(folder: Path) -> Optional[Path]:
    if not folder.is_dir():
        return None
    imgs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    return imgs[0] if imgs else None


def _pick_input(ns: argparse.Namespace) -> Path:
    direct_candidates: Iterable[Optional[str]] = (
        getattr(ns, "input_path", None),
        getattr(ns, "input", None),
        getattr(ns, "ms_path", None),
        getattr(ns, "pan_path", None),
        getattr(ns, "input_image", None),
    )
    for item in direct_candidates:
        if not item:
            continue
        p = Path(item)
        if p.is_file():
            return p
        if p.is_dir():
            hit = _first_image_in_dir(p)
            if hit is not None:
                return hit

    for item in (
        getattr(ns, "input_dir", None),
        getattr(ns, "data_dir", None),
        getattr(ns, "folder_lq", None),
    ):
        if not item:
            continue
        hit = _first_image_in_dir(Path(item))
        if hit is not None:
            return hit

    raise FileNotFoundError("No usable input image found.")


def _pick_output(ns: argparse.Namespace, inp: Path) -> Path:
    out_file = getattr(ns, "output_path", None)
    if out_file:
        p = Path(out_file)
        if p.suffix.lower() in SUPPORTED_EXT:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        p.mkdir(parents=True, exist_ok=True)
        return p / inp.name

    for maybe_dir in (
        getattr(ns, "output_dir", None),
        getattr(ns, "save_dir", None),
        getattr(ns, "result_dir", None),
        getattr(ns, "output", None),
    ):
        if maybe_dir:
            d = Path(maybe_dir)
            d.mkdir(parents=True, exist_ok=True)
            return d / inp.name

    fallback = Path("/tmp") / f"{inp.stem}_native_out.png"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def parse_common_args(default_scale: int = 4):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_path")
    parser.add_argument("--input")
    parser.add_argument("--input_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--folder_lq")
    parser.add_argument("--input_image")
    parser.add_argument("--ms_path")
    parser.add_argument("--pan_path")

    parser.add_argument("--output_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--output")
    parser.add_argument("--save_dir")
    parser.add_argument("--result_dir")

    parser.add_argument("--scale", type=int, default=default_scale)
    parser.add_argument("--upscale", type=float, default=float(default_scale))

    parser.add_argument("-i", dest="short_in")
    parser.add_argument("-o", dest="short_out")

    ns, extras = parser.parse_known_args()

    if not ns.input_path and ns.short_in:
        ns.input_path = ns.short_in
    if not ns.output_path and ns.short_out:
        ns.output_path = ns.short_out

    src = _pick_input(ns)
    dst = _pick_output(ns, src)

    scale = int(ns.scale or ns.upscale or default_scale)
    return ns, extras, src, dst, scale


def run_backend(
    *,
    backend_script: Path,
    src: Path,
    dst: Path,
    extra_args: list[str] | None = None,
    cwd: Path | None = None,
) -> None:
    if not backend_script.exists():
        raise FileNotFoundError(f"Backend script not found: {backend_script}")

    timeout_sec = 0
    try:
        timeout_sec = int(os.environ.get("SATA_NATIVE_BACKEND_TIMEOUT_SEC", "0"))
    except Exception:
        timeout_sec = 0

    cmd = [
        "python3",
        str(backend_script),
        "--input_path",
        str(src),
        "--output_path",
        str(dst),
    ]
    if extra_args:
        cmd.extend(extra_args)

    work_dir = str(cwd if cwd is not None else backend_script.parent)
    proc = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=timeout_sec if timeout_sec > 0 else None,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Native backend failed ({backend_script.name})\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    if not dst.exists():
        raise RuntimeError(f"Backend finished but output was not created: {dst}")


def run_backend_with_opencv_fallback(
    *,
    backend_script: Path,
    src: Path,
    dst: Path,
    task: str,
    scale: int = 4,
    variant: str = "",
    extra_args: list[str] | None = None,
    cwd: Path | None = None,
) -> None:
    """
    Try the real backend first. If it fails, fallback to the local OpenCV backend
    unless SATA_ENFORCE_REAL_BACKEND=1 is set.
    """
    try:
        run_backend(
            backend_script=backend_script,
            src=src,
            dst=dst,
            extra_args=extra_args or [],
            cwd=cwd,
        )
        print(f"[native-real] {backend_script.name} succeeded")
        return
    except Exception as e:
        if os.environ.get("SATA_ENFORCE_REAL_BACKEND", "0") == "1":
            raise
        print(f"[native-real] {backend_script.name} failed; fallback -> opencv backend. Reason: {e}")

    here = Path(__file__).resolve().parent
    opencv_backend = (here / "opencv_native_backend.py").resolve()
    run_backend(
        backend_script=opencv_backend,
        src=src,
        dst=dst,
        extra_args=[
            "--task", task,
            "--scale", str(scale),
            "--variant", variant,
        ],
        cwd=opencv_backend.parent,
    )
