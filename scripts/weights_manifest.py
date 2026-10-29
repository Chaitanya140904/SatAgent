#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
SATA_ROOT = HERE.parents[1]
if str(SATA_ROOT) not in sys.path:
    sys.path.insert(0, str(SATA_ROOT))

from executor import executor


CKPT_KEYS = {
    "--ckpt", "--ckpts", "--model_path", "--weight", "--weights",
    "--ckpt_path", "--pretrained_path", "--ram_ft_path", "--ram_path",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _extract_ckpts(opts: list[str]) -> list[Path]:
    out: list[Path] = []
    i = 0
    while i < len(opts):
        tok = str(opts[i])
        key = tok.split("=", 1)[0]
        if key in CKPT_KEYS:
            if "=" in tok:
                val = tok.split("=", 1)[1]
            elif i + 1 < len(opts):
                val = str(opts[i + 1])
                i += 1
            else:
                val = ""
            if val and str(val).strip().lower() not in {"none", "null", "na", "n/a"}:
                out.append(Path(val).expanduser())
        i += 1
    return out


def build_manifest(root: Path) -> dict:
    root = root.resolve()
    tmp = root / "outputs" / "_weights_manifest_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    tiny = np.full((64, 64, 3), 127, dtype=np.uint8)
    tiny_in = tmp / "in.png"
    cv2.imwrite(str(tiny_in), tiny)

    entries = []
    for subtask, tools in executor.toolbox_router.items():
        for tool in tools:
            rec = {
                "subtask": subtask,
                "model": tool.tool_name,
                "script_path": str(getattr(tool, "script_path", "")),
                "weights": [],
            }
            try:
                out = tmp / f"{subtask}_{tool.tool_name}.png"
                setattr(tool, "input_path", tiny_in)
                setattr(tool, "output_path", out)
                opts = [str(x) for x in tool._get_cmd_opts()]
                paths = _extract_ckpts(opts)
                for p in paths:
                    p_abs = p if p.is_absolute() else (root / p)
                    item = {
                        "path": str(p_abs),
                        "exists": p_abs.exists(),
                    }
                    if p_abs.exists() and p_abs.is_file():
                        item["size_bytes"] = p_abs.stat().st_size
                        item["sha256"] = _sha256(p_abs)
                    entries.append({
                        **rec,
                        "weights": [item],
                    })
                if not paths:
                    entries.append(rec)
            except Exception as e:
                entries.append({**rec, "error": str(e)})

    return {
        "version": 1,
        "root": str(root),
        "entries": entries,
    }


def cmd_scan(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    manifest = build_manifest(root)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {out}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    mf = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    bad = 0
    total_w = 0
    for e in mf.get("entries", []):
        for w in e.get("weights", []):
            total_w += 1
            p = Path(w.get("path", ""))
            if not p.exists():
                print(f"[MISSING] {e.get('subtask')}::{e.get('model')} -> {p}")
                bad += 1
            elif args.check_hash and w.get("sha256"):
                cur = _sha256(p)
                if cur != w["sha256"]:
                    print(f"[HASH MISMATCH] {p}")
                    bad += 1
    print(f"Checked weight refs: {total_w}, failures: {bad}")
    return 1 if bad else 0


def cmd_fetch(args: argparse.Namespace) -> int:
    mf = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fetched = 0
    skipped = 0
    failed = 0
    for e in mf.get("entries", []):
        for w in e.get("weights", []):
            p = Path(w.get("path", ""))
            if p.exists():
                skipped += 1
                continue
            url = w.get("url")
            if not url:
                print(f"[NO URL] {p}")
                failed += 1
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                print(f"[FETCH] {url} -> {p}")
                urllib.request.urlretrieve(url, str(p))
                fetched += 1
            except Exception as ex:
                print(f"[FAIL] {p}: {ex}")
                failed += 1
    print(f"fetched={fetched} skipped={skipped} failed={failed}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Weights manifest scan/verify/fetch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Scan tools and build manifest")
    p_scan.add_argument("--root", default="sata")
    p_scan.add_argument("--out", default="sata/weights_manifest.json")
    p_scan.set_defaults(func=cmd_scan)

    p_verify = sub.add_parser("verify", help="Verify manifest paths/hashes")
    p_verify.add_argument("--manifest", default="sata/weights_manifest.json")
    p_verify.add_argument("--check-hash", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_fetch = sub.add_parser("fetch", help="Fetch missing weights when URL is provided")
    p_fetch.add_argument("--manifest", default="sata/weights_manifest.json")
    p_fetch.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
