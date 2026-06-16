#!/usr/bin/env python
"""Interactive 3D (viser) viewer for the evidential-occupancy output.

The data-processing pipeline (`pixi run data-processing`) writes, per LIDAR key-frame,
an accumulated reflection/transmission evidence volume to

    data/nuscenes_extra/reflection_and_transmission_multi_frame/<scene>/LIDAR_TOP/<token>.arrow

This viewer loads those `.arrow` files directly, turns the evidence into a binary
occupancy grid using Dempster-Shafer belief (a voxel is occupied when the belief
mass for "occupied" exceeds the mass for "free"), and shows the occupied voxels as a
height-colored point cloud you can fly around in the browser.

It is meant as a sanity-check on a handful of samples *before* running the full
pipeline on v1.0-trainval.

Usage (inside the pixi env):
    pixi run python scripts/vis_occupancy_viser.py --num_samples 5
    pixi run python scripts/vis_occupancy_viser.py \
        --data_dir data/nuscenes_extra/reflection_and_transmission_multi_frame \
        --num_samples 5 --p_fn 0.8 --p_fp 0.2

Then open the printed URL (default http://localhost:8080).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import polars as pl
import torch
import viser

from scene_reconstruction.core.volume import Volume
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch
from scene_reconstruction.math.dempster_shafer import belief_from_reflection_and_transmission_stacked

RT_COL = "LIDAR_TOP.reflection_and_transmission_multi_frame"
LOWER_COL = f"{RT_COL}.volume.lower"
UPPER_COL = f"{RT_COL}.volume.upper"


def discover_files(data_dir: Path) -> list[Path]:
    """All accumulated-evidence arrow files, sorted by scene then token."""
    files = sorted(data_dir.glob("*/LIDAR_TOP/*.arrow"))
    return files


def pick_samples(files: list[Path], num_samples: int) -> list[Path]:
    """Evenly spread `num_samples` picks across all files (variety across scenes)."""
    if num_samples >= len(files):
        return files
    idx = np.linspace(0, len(files) - 1, num_samples).round().astype(int)
    # keep unique + ordered in case of rounding collisions
    seen, picked = set(), []
    for i in idx:
        i = int(i)
        while i in seen and i < len(files) - 1:
            i += 1
        seen.add(i)
        picked.append(files[i])
    return picked


def load_sample(path: Path):
    """Load one arrow file -> (rt[2,X,Y,Z] float32, lower[3], upper[3], scene, token)."""
    df = pl.read_ipc(path, memory_map=False)
    rt = series_to_torch(df[RT_COL])[0].float()  # [2, X, Y, Z] (reflections, transmissions)
    lower = series_to_torch(df[LOWER_COL])[0].float()  # [3]
    upper = series_to_torch(df[UPPER_COL])[0].float()  # [3]
    token = df["LIDAR_TOP.sample_data.token"].item()
    scene = path.parent.parent.name
    return rt, lower, upper, scene, token


def occupancy_points(rt, lower, upper, p_fn, p_fp):
    """Evidence volume -> occupied voxel centers + per-voxel confidence (m_o - m_f)."""
    bba = belief_from_reflection_and_transmission_stacked(rt[None], p_fn=p_fn, p_fp=p_fp)  # [1, 2, X, Y, Z]
    m_o, m_f = bba[0, 0], bba[0, 1]
    occ = m_o > m_f
    volume = Volume.new_volume(lower.tolist(), upper.tolist())
    grid = volume.new_coord_grid(tuple(occ.shape))[0]  # [X, Y, Z, 3] voxel centers
    pts = grid[occ]  # [N, 3]
    conf = (m_o - m_f)[occ]  # [N]
    return pts.numpy().astype(np.float32), conf.numpy().astype(np.float32)


def colorize(points: np.ndarray, mode: str, conf: np.ndarray, z_range: tuple[float, float]) -> np.ndarray:
    """Map points -> uint8 RGB colors."""
    if mode == "confidence":
        vals = np.clip(conf, 0.0, 1.0)
        cmap = "viridis"
    else:  # height
        z0, z1 = z_range
        vals = np.clip((points[:, 2] - z0) / max(z1 - z0, 1e-6), 0.0, 1.0)
        cmap = "turbo"
    rgb = mpl.colormaps[cmap](vals)[:, :3]
    return (rgb * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/nuscenes_extra/reflection_and_transmission_multi_frame",
                    help="Directory with <scene>/LIDAR_TOP/<token>.arrow files.")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--p_fn", type=float, default=0.8, help="False-negative prob (occupied|transmission).")
    ap.add_argument("--p_fp", type=float, default=0.2, help="False-positive prob (free|reflection).")
    ap.add_argument("--point_size", type=float, default=0.18, help="~0.9 * 0.2m voxel.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("VISER_PORT", "8080")))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = discover_files(data_dir)
    if not files:
        raise SystemExit(
            f"No .arrow files under {data_dir}. Run the pipeline first, e.g.\n"
            f"  pixi run python -m scene_reconstruction.cli.main export ./conf preview temporal-accumulation"
        )
    picked = pick_samples(files, args.num_samples)
    print(f"Found {len(files)} accumulated samples; showing {len(picked)}:")
    samples = []
    for p in picked:
        rt, lower, upper, scene, token = load_sample(p)
        samples.append({"rt": rt, "lower": lower, "upper": upper, "scene": scene, "token": token})
        print(f"  - {scene}/{token}")

    z_range = (float(samples[0]["lower"][2]), float(samples[0]["upper"][2]))

    server = viser.ViserServer(host=os.environ.get("VISER_HOST", "0.0.0.0"), port=args.port)
    server.scene.add_frame("/ego", show_axes=True, axes_length=3.0, axes_radius=0.1)
    server.scene.add_grid("/ground", width=80.0, height=80.0, position=(0.0, 0.0, z_range[0]))
    pc = server.scene.add_point_cloud("/occupancy", points=np.zeros((1, 3), np.float32),
                                      colors=np.zeros((1, 3), np.uint8),
                                      point_size=args.point_size, point_shape="square")

    server.gui.add_markdown("## Evidential occupancy — preview\n"
                            "Occupied = belief(occupied) > belief(free). "
                            "Tune `p_fn`/`p_fp` to see the threshold change.")
    info = server.gui.add_markdown("")
    with server.gui.add_folder("Navigate samples"):
        prev_b = server.gui.add_button("Prev", icon=viser.Icon.CHEVRON_LEFT)
        next_b = server.gui.add_button("Next", icon=viser.Icon.CHEVRON_RIGHT)
        sld = server.gui.add_slider("Sample", min=0, max=len(samples) - 1, step=1, initial_value=0)
    with server.gui.add_folder("Belief thresholds"):
        p_fn_sld = server.gui.add_slider("p_fn", min=0.5, max=0.99, step=0.01, initial_value=args.p_fn)
        p_fp_sld = server.gui.add_slider("p_fp", min=0.01, max=0.5, step=0.01, initial_value=args.p_fp)
    with server.gui.add_folder("Display"):
        color_mode = server.gui.add_dropdown("Color by", options=("height", "confidence"), initial_value="height")
        psize = server.gui.add_slider("Point size", min=0.02, max=0.4, step=0.01, initial_value=args.point_size)

    state = {"idx": 0}

    def render():
        s = samples[state["idx"]]
        pts, conf = occupancy_points(s["rt"], s["lower"], s["upper"], p_fn_sld.value, p_fp_sld.value)
        if len(pts) == 0:
            pts = np.zeros((1, 3), np.float32)
            cols = np.zeros((1, 3), np.uint8)
            n_occ = 0
        else:
            cols = colorize(pts, color_mode.value, conf, z_range)
            n_occ = len(pts)
        pc.points = pts
        pc.colors = cols
        pc.point_size = psize.value
        info.content = (
            f"### Sample {state['idx'] + 1}/{len(samples)}\n"
            f"**scene:** `{s['scene']}`\n\n"
            f"**token:** `{s['token']}`\n\n"
            f"**occupied voxels:** {n_occ:,}\n\n"
            f"p_fn={p_fn_sld.value:.2f}  p_fp={p_fp_sld.value:.2f}"
        )
        print(f"sample {state['idx'] + 1}/{len(samples)}: {s['scene']}/{s['token']}  occupied={n_occ}")

    def show(i):
        state["idx"] = int(np.clip(i, 0, len(samples) - 1))
        sld.value = state["idx"]
        render()

    @prev_b.on_click
    def _(_):
        show(state["idx"] - 1)

    @next_b.on_click
    def _(_):
        show(state["idx"] + 1)

    @sld.on_update
    def _(_):
        show(int(sld.value))

    for ctrl in (p_fn_sld, p_fp_sld, color_mode, psize):
        ctrl.on_update(lambda _: render())

    show(0)
    print(f"\nviser running — open http://localhost:{args.port}  (Ctrl+C to exit)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
