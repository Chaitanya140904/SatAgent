from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from sata.main import run_pipeline


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            out.append(p)
    out.sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser("Run SATA pipeline over dataset images.")
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--profile", default="sata/configs/profiles/Aer4K_F.yaml")
    ap.add_argument("--gpu-id", type=int, default=1)
    ap.add_argument("--disable-vlm", action="store_true")
    ap.add_argument("--strict-real-execution", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.disable_vlm:
        os.environ["SATA_DISABLE_VLM"] = "1"

    images = list_images(input_root)
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_jsonl = output_root / f"progress_{ts}.jsonl"
    summary_json = output_root / "summary_latest.json"

    succeeded = 0
    failed = 0
    skipped = 0

    for idx, img in enumerate(images, start=1):
        rel = img.relative_to(input_root)
        out_dir = output_root / rel.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        final_out = out_dir / "satagent_final_output.png"

        if final_out.exists():
            skipped += 1
            rec = {
                "index": idx,
                "input": str(img),
                "output_dir": str(out_dir),
                "final_output": str(final_out),
                "status": "skipped_exists",
                "error": "",
            }
        else:
            status = "ok"
            err = ""
            final = ""
            try:
                final = run_pipeline(
                    image_path=str(img),
                    profile_path=str(args.profile),
                    mock_vlm=False,
                    output_dir=str(out_dir),
                    tool_run_gpu_id=args.gpu_id,
                    strict_real_execution=args.strict_real_execution,
                )
                succeeded += 1
            except Exception as e:
                failed += 1
                status = "failed"
                err = f"{type(e).__name__}: {e}"
            rec = {
                "index": idx,
                "input": str(img),
                "output_dir": str(out_dir),
                "final_output": str(final),
                "status": status,
                "error": err,
            }

        with progress_jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        print(
            f"[{idx}/{len(images)}] {rec['status']} :: {img}",
            flush=True,
        )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "profile": str(args.profile),
        "gpu_id": args.gpu_id,
        "disable_vlm": bool(args.disable_vlm),
        "strict_real_execution": bool(args.strict_real_execution),
        "total": len(images),
        "succeeded": succeeded,
        "failed": failed,
        "skipped_exists": skipped,
        "progress_file": str(progress_jsonl),
        "finished_at": datetime.now().isoformat(),
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

