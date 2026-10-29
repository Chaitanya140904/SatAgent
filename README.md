# SatAgent (SATA) – 4K Satellite Restoration

Closed-loop 3-agent MoE pipeline for satellite image restoration:
- Agent 1: Perception + JSON planning
- Agent 2: Local/Global routed restoration
- Agent 3: QA rollback (NIQE/entropy + task-aware checks)

## Quick Start

```bash
python3 sata/main.py \
  --image sample_img.jpeg \
  --profile sata/configs/profiles/Aer4K_F.yaml \
  --mock-vlm \
  --force-full-moe \
  --output-dir sata/outputs/quickstart
```

This writes:
- `satagent_final_output.png` (final 4K output)
- `execution_plan_runtime.json` (runtime plan + rewrites + statuses)

## Production Batch Run

```bash
python3 sata/infer_satagent.py \
  --input_dir sata/datasets \
  --output_dir sata/outputs/batch \
  --profile_name sata/configs/profiles/Aer4K_P.yaml \
  --force-full-moe
```

## Paper-Ready Orchestration

One command to run pipeline + generate backend coverage and tool readiness reports:

```bash
python3 sata/scripts/run_paper_ready.py \
  --image sample_img.jpeg \
  --profile sata/configs/profiles/Aer4K_F.yaml \
  --output-root sata/outputs/paper_ready \
  --force-full-moe \
  --strict-real-execution \
  --require-true-models
```

Generated reports:
- `backend_coverage.{json,md}`
- `tool_readiness.{json,md}`
- `paper_ready_summary.{json,md}`

## Profiles

- `Aer4K_F.yaml`: Fidelity (PSNR/SSIM/SAM-oriented)
- `Aer4K_P.yaml`: Perception (NIQE/LPIPS/FID-oriented)
- `Aer4K_H.yaml`: Hybrid
- `MultiTemp_F.yaml`: enables multi-temporal fusion phase (3B)

