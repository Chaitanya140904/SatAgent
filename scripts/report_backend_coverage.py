from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sata.executor import executor

PHASE_BY_SUBTASK = {
    "roi_masking": "0",
    "cloud_removal": "1A",
    "dehazing": "1B",
    "shadow_correction": "1C",
    "denoising": "2A",
    "deblurring": "2B",
    "destriping": "2C",
    "artifact_removal": "2D",
    "pan_sharpen": "3A",
    "temporal_fusion": "3B",
    "super_resolution": "4",
}

PHASE_LABEL = {
    "0": "ROI Masking",
    "1A": "Thick Cloud Removal",
    "1B": "Global Dehazing",
    "1C": "Shadow Correction",
    "2A": "Thermal/ISO Denoising",
    "2B": "Motion Deblurring",
    "2C": "Destriping/Dead-Pixel",
    "2D": "Artifact Removal",
    "3A": "Pan-Sharpening",
    "3B": "Multi-Temporal Fusion",
    "4": "Super-Resolution (4K)",
}


def _backend_kind(script_path: Path) -> str:
    if not script_path.exists():
        return "missing"
    if script_path.suffix.lower() != ".py":
        return "non_python"

    txt = script_path.read_text(encoding="utf-8", errors="ignore")

    if "from adapter_infer import execute_from_cli" in txt and "_load_adapter" in txt:
        return "adapter"
    if "from native_proxy import parse_common_args, run_backend_with_opencv_fallback" in txt:
        return "native_proxy_real_first"
    if "from native_proxy import parse_common_args, run_backend" in txt:
        return "native_proxy_opencv_only"
    return "native_script"


def _write_md(path: Path, payload: dict) -> None:
    lines = []
    lines.append("# SatAgent Backend Coverage")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at_utc']}")
    lines.append(f"- Total model paths: {payload['summary']['total']}")
    lines.append(f"- Native scripts: {payload['summary']['native_script']}")
    lines.append(f"- Native proxies (real-first): {payload['summary']['native_proxy_real_first']}")
    lines.append(f"- Native proxies (opencv-only): {payload['summary']['native_proxy_opencv_only']}")
    lines.append(f"- Adapter wrappers: {payload['summary']['adapter']}")
    lines.append("")
    lines.append("## Phase Summary")
    for p in payload["phase_summary"]:
        lines.append(
            f"- Phase {p['phase']} ({p['label']}): total={p['total']}, "
            f"native_script={p['native_script']}, "
            f"proxy_real_first={p['native_proxy_real_first']}, "
            f"proxy_opencv_only={p['native_proxy_opencv_only']}, "
            f"adapter={p['adapter']}"
        )
    lines.append("")
    lines.append("## Model Paths")
    for r in payload["results"]:
        lines.append(
            f"- Phase {r['phase']} | {r['subtask']} | {r['tool_name']} | backend={r['backend']} | script={r['script_path']}"
        )
    path.write_text("\n".join(lines) + "\n")


def generate(output_json: Path, output_md: Path) -> dict:
    rows = []

    for subtask in sorted(executor.subtasks):
        phase = PHASE_BY_SUBTASK.get(subtask, "unknown")
        label = PHASE_LABEL.get(phase, "Unknown")
        routing = "pre-process" if subtask == "roi_masking" else executor.get_routing_mode(subtask)

        for tool in executor.toolbox_router.get(subtask, []):
            script_path = Path(str(getattr(tool, "script_path", "")))
            backend = _backend_kind(script_path)
            rows.append(
                {
                    "phase": phase,
                    "phase_label": label,
                    "subtask": subtask,
                    "routing": routing,
                    "tool_name": tool.tool_name,
                    "script_path": str(script_path),
                    "backend": backend,
                }
            )

    summary = {
        "total": len(rows),
        "native_script": sum(1 for r in rows if r["backend"] == "native_script"),
        "native_proxy_real_first": sum(1 for r in rows if r["backend"] == "native_proxy_real_first"),
        "native_proxy_opencv_only": sum(1 for r in rows if r["backend"] == "native_proxy_opencv_only"),
        "adapter": sum(1 for r in rows if r["backend"] == "adapter"),
        "missing": sum(1 for r in rows if r["backend"] == "missing"),
    }

    phases = []
    for phase in sorted({r["phase"] for r in rows}, key=lambda p: (p != "0", p)):
        pr = [r for r in rows if r["phase"] == phase]
        phases.append(
            {
                "phase": phase,
                "label": PHASE_LABEL.get(phase, "Unknown"),
                "total": len(pr),
                "native_script": sum(1 for r in pr if r["backend"] == "native_script"),
                "native_proxy_real_first": sum(1 for r in pr if r["backend"] == "native_proxy_real_first"),
                "native_proxy_opencv_only": sum(1 for r in pr if r["backend"] == "native_proxy_opencv_only"),
                "adapter": sum(1 for r in pr if r["backend"] == "adapter"),
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "phase_summary": phases,
        "results": rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    _write_md(output_md, payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser("Report backend coverage per model path.")
    ap.add_argument("--output-json", type=Path, default=Path("sata/tests/report_backend_coverage.json"))
    ap.add_argument("--output-md", type=Path, default=Path("sata/tests/report_backend_coverage.md"))
    args = ap.parse_args()

    payload = generate(args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))
    print(f"JSON report: {args.output_json}")
    print(f"MD report  : {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
