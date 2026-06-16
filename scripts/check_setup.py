#!/usr/bin/env python
"""Sanity-check the machine setup before launching a full run.

Prints the resolved data paths (from env vars) and whether the expected files exist, the
output root is writable, and CUDA is visible. Run after `source paths.env`:

    pixi run python scripts/check_setup.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def check(label: str, path: Path, must_contain=()) -> bool:
    ok = path.exists()
    print(f"  [{'OK ' if ok else 'MISS'}] {label}: {path}")
    all_ok = ok
    for sub in must_contain:
        sok = (path / sub).exists()
        all_ok = all_ok and sok
        print(f"         {'ok' if sok else '??'}  {sub}")
    return all_ok


def main() -> None:
    nuroot = os.environ.get("NUSCENES_ROOT", "data/nuscenes")
    extra = os.environ.get("NUSCENES_EXTRA_ROOT", "data/nuscenes_extra")
    ngpu = os.environ.get("NUM_GPUS", "(unset)")

    print("Resolved configuration:")
    print(f"  NUSCENES_ROOT       = {nuroot}")
    print(f"  NUSCENES_EXTRA_ROOT = {extra}")
    print(f"  NUM_GPUS            = {ngpu}")
    if "NUSCENES_ROOT" not in os.environ:
        print("  (NUSCENES_ROOT not set — did you `source paths.env`? falling back to ./data)")

    print("Checks:")
    ok = True
    ok &= check("nuScenes root", Path(nuroot), ["samples", "sweeps", "v1.0-trainval"])
    ok &= check("v1.0-trainval meta", Path(nuroot) / "v1.0-trainval", ["scene.json", "sample_data.json"])

    out = Path(extra)
    try:
        out.mkdir(parents=True, exist_ok=True)
        writable = os.access(out, os.W_OK)
    except OSError:
        writable = False
    print(f"  [{'OK ' if writable else 'FAIL'}] output root writable: {out}")
    ok &= writable

    try:
        import torch

        n = torch.cuda.device_count()
        avail = torch.cuda.is_available()
        print(f"  [{'OK ' if avail else 'FAIL'}] torch CUDA available: {avail}, device_count={n}")
        ok &= avail
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] torch import/CUDA check failed: {exc}")
        ok = False

    print("\nALL GOOD — ready to run." if ok else "\nSome checks FAILED — fix paths.env / env before running.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
