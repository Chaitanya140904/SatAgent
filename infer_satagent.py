from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from main import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SatAgent Inference (single image or batch directory)"
    )

    io = parser.add_argument_group("Inputs/Outputs")
    io.add_argument(
        "--image",
        "--input",
        dest="image",
        type=str,
        default=None,
        help="Single input image path.",
    )
    io.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        type=str,
        default=None,
        help="Directory containing input images for batch mode.",
    )
    io.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="./outputs/LQ_results",
        help="Output root directory.",
    )

    cfg = parser.add_argument_group("Pipeline")
    cfg.add_argument(
        "--profile",
        "--profile-name",
        "--profile_name",
        dest="profile",
        type=str,
        default="configs/profiles/Aer4K_F.yaml",
        help="Profile YAML path for SatAgent.",
    )
    cfg.add_argument(
        "--tool-run-gpu-id",
        "--tool_run_gpu_id",
        dest="tool_run_gpu_id",
        type=int,
        default=0,
        help="GPU ID to run toolbox models.",
    )
    cfg.add_argument("--mock-vlm", action="store_true")
    cfg.add_argument("--force-full-moe", action="store_true")
    cfg.add_argument("--strict-real-execution", action="store_true")
    cfg.add_argument("--require-true-models", action="store_true")
    cfg.add_argument("--require-preflight", action="store_true")
    cfg.add_argument("--preflight-smoke", action="store_true")
    cfg.add_argument("--preflight-alias-strict", action="store_true")
    cfg.add_argument("--skip-preflight", action="store_true")
    cfg.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Batch mode: continue processing other images when one fails.",
    )
    cfg.add_argument(
        "--dry-run-parse",
        action="store_true",
        help="Validate/print normalized arguments and exit without running models.",
    )

    args = parser.parse_args()
    if args.image is None and args.input_dir is None:
        parser.error("Provide either --image (single) or --input-dir (batch).")
    if args.image is not None and args.input_dir is not None:
        parser.error("Use only one of --image or --input-dir.")
    return args


def _iter_images(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in input_dir.glob("*") if p.suffix.lower() in exts])


def _run_one(image_path: Path, output_dir: Path, args: argparse.Namespace) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_pipeline(
        image_path=str(image_path),
        profile_path=args.profile,
        mock_vlm=args.mock_vlm,
        force_full_moe=args.force_full_moe,
        output_dir=str(output_dir),
        tool_run_gpu_id=args.tool_run_gpu_id,
        strict_real_execution=args.strict_real_execution,
        require_true_models=args.require_true_models,
        require_preflight=args.require_preflight,
        preflight_smoke=args.preflight_smoke,
        preflight_alias_strict=args.preflight_alias_strict,
        skip_preflight=args.skip_preflight,
    )


def main() -> None:
    args = parse_args()
    if args.dry_run_parse:
        print(
            {
                "image": args.image,
                "input_dir": args.input_dir,
                "output_dir": args.output_dir,
                "profile": args.profile,
                "tool_run_gpu_id": args.tool_run_gpu_id,
                "mock_vlm": args.mock_vlm,
                "force_full_moe": args.force_full_moe,
                "strict_real_execution": args.strict_real_execution,
                "require_true_models": args.require_true_models,
                "require_preflight": args.require_preflight,
                "preflight_smoke": args.preflight_smoke,
                "preflight_alias_strict": args.preflight_alias_strict,
                "skip_preflight": args.skip_preflight,
                "continue_on_error": args.continue_on_error,
            }
        )
        return
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        result = _run_one(image_path, out_root, args)
        print(f"Final restored image: {result}")
        return

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    images = _iter_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    for image_path in images:
        image_output_dir = out_root / image_path.stem
        result_png = image_output_dir / "satagent_final_output.png"
        if result_png.exists():
            print(f"[Skip] Already processed: {image_path.name}")
            continue
        print(f"[Processing] {image_path.name}")
        try:
            result = _run_one(image_path, image_output_dir, args)
            print(f"[Done] {image_path.name} -> {result}")
        except Exception as e:
            print(f"[Fail] {image_path.name}: {e}")
            if not args.continue_on_error:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
