#!/usr/bin/env python
"""Run the data-generation pipeline on one shard of scenes (one GPU).

The dataset's scenes are split into `--num-gpus` contiguous shards; this process handles
shard `--gpu-index` through the requested steps, in order. Launch it once per GPU with
CUDA_VISIBLE_DEVICES set (see scripts/run_full_multi_gpu.sh) so N GPUs cover the whole
dataset in parallel. Each shard is self-contained: temporal accumulation only ever reads
frames from within the same scene, so per-scene sharding is correct.

Paths come from the config (which reads ${oc.env:NUSCENES_ROOT,...} etc.), so nothing here
is hard-coded.

Examples:
    # one GPU's share of a 5-way split, all three steps:
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_shard.py --gpu-index 0 --num-gpus 5
    # just print the plan (no GPU work):
    python scripts/generate_shard.py --gpu-index 0 --num-gpus 5 --config-name preview --dry-run
    # build the shared annotation cache once, before launching workers:
    python scripts/generate_shard.py --gpu-index 0 --num-gpus 5 --prime-cache
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from scene_reconstruction.core.config import initialize_config

STEP_NODES = {
    "transmissions-reflections": "export.transmissions_reflections",
    "scene-flow": "export.scene_flow",
    "temporal-accumulation": "export.temporal_accumulation",
    "occupancy-export": "export.evidence_export",
    "occ3d-transfer": "export.occ3d_transfer",
}
# No-external-dependency stages that always run end-to-end per shard.
DEFAULT_STEPS = [
    "transmissions-reflections",
    "scene-flow",
    "temporal-accumulation",
    "occupancy-export",
    "occ3d-transfer",
]


def count_scenes(cfg) -> int:
    """Total scenes in the dataset version (from scene.json) without a full dataset load."""
    ds = cfg.export.transmissions_reflections.ds
    scene_json = Path(str(ds.data_root)) / str(ds.version) / "scene.json"
    with open(scene_json) as fh:
        return len(json.load(fh))


def shard_range(total: int, gpu_index: int, num_gpus: int) -> tuple[int, int]:
    """Contiguous [offset, count) shard for this GPU (last shard may be shorter)."""
    chunk = math.ceil(total / num_gpus)
    offset = gpu_index * chunk
    return offset, chunk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default="./conf")
    ap.add_argument("--config-name", default="full")
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--num-gpus", type=int, required=True)
    ap.add_argument("--steps", nargs="+", default=DEFAULT_STEPS, choices=list(STEP_NODES))
    ap.add_argument("--dry-run", action="store_true", help="Instantiate + print the plan, but run nothing.")
    ap.add_argument(
        "--prime-cache",
        action="store_true",
        help="Only build the shared LIDAR sample-annotation cache, then exit. Run ONCE before "
        "launching parallel workers (avoids a write race).",
    )
    args = ap.parse_args()

    cfg = initialize_config(Path(args.config_dir), args.config_name)

    if args.prime_cache:
        print("building sample-annotation cache (one-time, single process)...", flush=True)
        sf = instantiate(cfg.export.scene_flow)
        sf.ds.load_sensor_sample_annotation("LIDAR_TOP")
        print("annotation cache ready", flush=True)
        return

    total = count_scenes(cfg)
    offset, count = shard_range(total, args.gpu_index, args.num_gpus)
    end = min(offset + count, total)
    print(
        f"[gpu {args.gpu_index}/{args.num_gpus}] scenes [{offset}, {end}) of {total} | steps={args.steps}",
        flush=True,
    )
    if offset >= total:
        print(f"[gpu {args.gpu_index}] no scenes in this shard; nothing to do", flush=True)
        return

    for step in args.steps:
        node = OmegaConf.select(cfg, STEP_NODES[step])
        with open_dict(node):
            node.scene_offset = offset
            node.num_scenes = count
        print(f"[gpu {args.gpu_index}] === {step}: scenes {offset}..{end} ===", flush=True)
        obj = instantiate(node)
        if args.dry_run:
            print(f"[gpu {args.gpu_index}] dry-run: would run {type(obj).__name__}.process_data()", flush=True)
            continue
        obj.process_data()
    print(f"[gpu {args.gpu_index}] DONE", flush=True)


if __name__ == "__main__":
    main()
