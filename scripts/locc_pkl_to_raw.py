#!/usr/bin/env python
"""Convert LOcc GT-generation output (label_ovo.pkl) -> our locc_raw/*.npz format.

LOcc writes, per key-frame, <locc_gts>/<scene>/<sample_token>/label_ovo.pkl with
``ovo_gt`` = [200,200,16] uint8 (open-vocab vocabulary indices; 255 = empty), on the same
Occ3D-nuScenes grid as occ3d-transfer. This maps each sample token -> the LIDAR sample-data
token (the key every stage uses) and writes
<extra>/locc_raw/<scene>/LIDAR_TOP/<lidar_token>.npz with `semantics` (-1 = empty), so the
existing `locc-transfer` stage can lift it onto our geometry. Run in the pixi env:

    pixi run python scripts/locc_pkl_to_raw.py \
        --locc_gts /home/adteam/Documents/LOcc/data/occ3d/san_gts_qwen_scene \
        --vocab    /home/adteam/Documents/LOcc/data/occ3d/san_qwen_scene/vocab.txt
"""
from __future__ import annotations

import torch  # noqa: F401  (import before scipy/scene_reconstruction for libstdc++ ordering)
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locc_gts", required=True, help="LOcc san_gts_qwen_scene dir (<scene>/<sample_token>/label_ovo.pkl)")
    ap.add_argument("--vocab", default="", help="LOcc vocab.txt (label index -> name); copied to locc_raw/vocab.json")
    ap.add_argument("--data_root", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--extra", default="data/nuscenes_extra")
    args = ap.parse_args()

    ds = NuscenesDataset(data_root=args.data_root, version=args.version, key_frames_only=True)
    scene = ds.join(ds.scene, ds.sample)
    scene = ds.load_sample_data(scene, "LIDAR_TOP", with_data=False)
    scene = scene.filter(pl.col("LIDAR_TOP.sample_data.is_key_frame"))
    sample_to_lidar = {
        row["sample.token"]: row["LIDAR_TOP.sample_data.token"] for row in scene.iter_rows(named=True)
    }

    locc_gts = Path(args.locc_gts)
    extra = Path(args.extra)
    n = 0
    for pkl in sorted(locc_gts.glob("*/*/label_ovo.pkl")):
        sample_token = pkl.parent.name
        scene_name = pkl.parent.parent.name
        lidar_token = sample_to_lidar.get(sample_token)
        if lidar_token is None:
            print(f"skip {scene_name}/{sample_token[:8]}: not a key-frame in {args.version}")
            continue
        ovo = pickle.load(open(pkl, "rb"))["ovo_gt"].astype(np.int16)  # [200,200,16], 255=empty
        sem = np.where(ovo == 255, np.int16(-1), ovo)
        out = extra / "locc_raw" / scene_name / "LIDAR_TOP" / f"{lidar_token}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            semantics=sem,
            feat_coords=np.zeros((0, 3), np.int16),  # label-only (no CLIP features in this pass)
            feat=np.zeros((0, 128), np.float16),
        )
        n += 1
    print(f"wrote {n} locc_raw npz files to {extra/'locc_raw'}")

    if args.vocab and Path(args.vocab).exists():
        vocab = [w.strip() for w in open(args.vocab) if w.strip()]
        vocab_json = extra / "locc_raw" / "vocab.json"
        vocab_json.write_text(json.dumps({i: w for i, w in enumerate(vocab)}, indent=0))
        print(f"wrote vocab ({len(vocab)} classes) to {vocab_json}")


if __name__ == "__main__":
    main()
