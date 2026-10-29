from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sata.main import run_pipeline
from sata.scripts.run_full_tool_battery import run_battery
from sata.scripts.report_backend_coverage import generate as generate_backend_coverage


def _write_markdown(path: Path, payload: dict) -> None:
    lines: list[str] = []
    lines.append("# SatAgent Paper-Ready Run")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at_utc']}")
    lines.append(f"- Image: `{payload['image']}`")
    lines.append(f"- Profile: `{payload['profile']}`")
    lines.append(f"- Final output: `{payload['final_output']}`")
    lines.append(f"- Strict real execution: `{payload['strict_real_execution']}`")
    lines.append(f"- Require true models: `{payload['require_true_models']}`")
    lines.append("")
    lines.append("## Reports")
    lines.append(f"- Backend coverage JSON: `{payload['reports']['backend_json']}`")
    lines.append(f"- Backend coverage MD: `{payload['reports']['backend_md']}`")
    lines.append(f"- Tool readiness JSON: `{payload['reports']['battery_json']}`")
    lines.append(f"- Tool readiness MD: `{payload['reports']['battery_md']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- Tool readiness: {payload['battery_summary']['ready']}/{payload['battery_summary']['total']} ready"
    )
    lines.append(
        f"- Backend kinds: native_script={payload['backend_summary']['native_script']}, "
        f"proxy_real_first={payload['backend_summary']['native_proxy_real_first']}, "
        f"proxy_opencv_only={payload['backend_summary']['native_proxy_opencv_only']}, "
        f"adapter={payload['backend_summary']['adapter']}"
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        "Run full SatAgent pipeline + readiness reports for paper/demo packaging."
    )
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--profile", type=Path, default=Path("sata/configs/profiles/Aer4K_F.yaml"))
    ap.add_argument("--output-root", type=Path, default=Path("sata/outputs/paper_ready"))
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--mock-vlm", action="store_true")
    ap.add_argument("--force-full-moe", action="store_true")
    ap.add_argument("--strict-real-execution", action="store_true")
    ap.add_argument("--require-true-models", action="store_true")
    ap.add_argument("--tool-timeout-sec", type=float, default=180.0)
    ap.add_argument("--skip-battery", action="store_true")
    ap.add_argument("--skip-backend-report", action="store_true")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.tool_timeout_sec > 0:
        os.environ["SATA_TOOL_CMD_TIMEOUT_SEC"] = str(args.tool_timeout_sec)

    final_output = run_pipeline(
        image_path=str(args.image.resolve()),
        profile_path=str(args.profile.resolve()),
        mock_vlm=args.mock_vlm,
        force_full_moe=args.force_full_moe,
        output_dir=str((out_root / "pipeline").resolve()),
        tool_run_gpu_id=args.gpu_id,
        strict_real_execution=args.strict_real_execution,
        require_true_models=args.require_true_models,
    )

    backend_json = report_dir / "backend_coverage.json"
    backend_md = report_dir / "backend_coverage.md"
    battery_json = report_dir / "tool_readiness.json"
    battery_md = report_dir / "tool_readiness.md"

    if args.skip_backend_report:
        backend_payload = {"summary": {}}
    else:
        backend_payload = generate_backend_coverage(backend_json, backend_md)

    if args.skip_battery:
        battery_payload = {"summary": {"total": 0, "ready": 0, "blocked": 0}}
    else:
        battery_payload = run_battery(
            output_json=battery_json,
            output_md=battery_md,
            workspace=out_root / "workspace" / "readiness_battery",
            gpu_id=args.gpu_id,
            strict_real_execution=args.strict_real_execution,
            require_true_models=args.require_true_models,
            tool_timeout_sec=args.tool_timeout_sec,
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image": str(args.image.resolve()),
        "profile": str(args.profile.resolve()),
        "final_output": str(Path(final_output).resolve()),
        "strict_real_execution": bool(args.strict_real_execution),
        "require_true_models": bool(args.require_true_models),
        "reports": {
            "backend_json": str(backend_json.resolve()),
            "backend_md": str(backend_md.resolve()),
            "battery_json": str(battery_json.resolve()),
            "battery_md": str(battery_md.resolve()),
        },
        "backend_summary": backend_payload.get("summary", {}),
        "battery_summary": battery_payload.get("summary", {}),
    }

    summary_json = report_dir / "paper_ready_summary.json"
    summary_md = report_dir / "paper_ready_summary.md"
    summary_json.write_text(json.dumps(payload, indent=2))
    _write_markdown(summary_md, payload)

    print(json.dumps(payload, indent=2))
    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD  : {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
