#!/usr/bin/env python
"""Interactive 3D (viser) viewer for the evidential-occupancy output and its labels.

Reads the accumulated reflection/transmission volumes (``temporal-accumulation``) and turns
them into occupancy via Dempster-Shafer belief. If the per-stage label outputs exist under
``--labels_root`` (``evidence/``, ``occ3d_transfer/``), the occupied voxels can also be
colored by **epistemic uncertainty** (m_omega) or **Occ3D semantic class**.

Usage (inside the pixi env):
    pixi run python scripts/vis_occupancy_viser.py --num_samples 5
    pixi run python scripts/vis_occupancy_viser.py \
        --data_dir data/nuscenes_extra/reflection_and_transmission_multi_frame \
        --labels_root data/nuscenes_extra --num_samples 5

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
from scene_reconstruction.visualization.colormap import occupancy_color_map, occupancy_to_color

RT_COL = "LIDAR_TOP.reflection_and_transmission_multi_frame"
LOWER_COL = f"{RT_COL}.volume.lower"
UPPER_COL = f"{RT_COL}.volume.upper"
CLASS_NAMES = list(occupancy_to_color.keys())  # index -> name, 0=other .. 17=free
COLOR_MODES = ("height", "confidence", "uncertainty", "occ3d_class")


def discover_files(data_dir: Path) -> list[Path]:
    """All accumulated-evidence arrow files, sorted by scene then token."""
    return sorted(data_dir.glob("*/LIDAR_TOP/*.arrow"))


def pick_samples(files: list[Path], num_samples: int) -> list[Path]:
    """Evenly spread `num_samples` picks across all files (variety across scenes)."""
    if num_samples >= len(files):
        return files
    idx = np.linspace(0, len(files) - 1, num_samples).round().astype(int)
    seen, picked = set(), []
    for i in idx:
        i = int(i)
        while i in seen and i < len(files) - 1:
            i += 1
        seen.add(i)
        picked.append(files[i])
    return picked


def _maybe_load_semantics(labels_root: Path, scene: str, token: str) -> torch.Tensor | None:
    """Occ3D-transfer semantics [X,Y,Z] for this token, or None if not generated."""
    f = labels_root / "occ3d_transfer" / scene / "LIDAR_TOP" / f"{token}.arrow"
    if not f.exists():
        return None
    df = pl.read_ipc(f, memory_map=False)
    return series_to_torch(df["LIDAR_TOP.occ3d_transfer.semantics"])[0].to(torch.int64)  # [X,Y,Z]


def load_sample(path: Path, labels_root: Path | None):
    """Load one key-frame: RT volume, bounds, scene/token, and optional Occ3D semantics."""
    df = pl.read_ipc(path, memory_map=False)
    rt = series_to_torch(df[RT_COL])[0].float()  # [2, X, Y, Z]
    lower = series_to_torch(df[LOWER_COL])[0].float()
    upper = series_to_torch(df[UPPER_COL])[0].float()
    token = df["LIDAR_TOP.sample_data.token"].item()
    scene = path.parent.parent.name
    sem = _maybe_load_semantics(labels_root, scene, token) if labels_root is not None else None
    return {"rt": rt, "lower": lower, "upper": upper, "scene": scene, "token": token, "sem": sem}


def occupancy(sample, p_fn, p_fp):
    """Occupied voxel centers + per-voxel m_o-m_f, m_omega, and Occ3D class (if available)."""
    rt = sample["rt"]
    bba = belief_from_reflection_and_transmission_stacked(rt[None], p_fn=p_fn, p_fp=p_fp, with_omega=True)
    m_o, m_f, m_omega = bba[0, 0], bba[0, 1], bba[0, 2]  # [X, Y, Z] each
    occ = m_o > m_f
    volume = Volume.new_volume(sample["lower"].tolist(), sample["upper"].tolist())
    grid = volume.new_coord_grid(tuple(occ.shape))[0]  # [X, Y, Z, 3]
    pts = grid[occ].numpy().astype(np.float32)  # [N, 3]
    conf = (m_o - m_f)[occ].numpy().astype(np.float32)
    omega = m_omega[occ].numpy().astype(np.float32)
    cls = sample["sem"][occ].numpy().astype(np.int64) if sample["sem"] is not None else None
    return pts, conf, omega, cls


def colorize(pts, conf, omega, cls, mode, z_range) -> np.ndarray:
    """Map per-point scalars/classes -> uint8 RGB."""
    if mode == "occ3d_class" and cls is not None:
        return occupancy_color_map[np.clip(cls, 0, len(CLASS_NAMES) - 1)].numpy().astype(np.uint8)
    if mode == "uncertainty":
        return (mpl.colormaps["turbo"](np.clip(omega, 0.0, 1.0))[:, :3] * 255).astype(np.uint8)
    if mode == "confidence":
        return (mpl.colormaps["viridis"](np.clip(conf, 0.0, 1.0))[:, :3] * 255).astype(np.uint8)
    z0, z1 = z_range  # height
    vals = np.clip((pts[:, 2] - z0) / max(z1 - z0, 1e-6), 0.0, 1.0)
    return (mpl.colormaps["turbo"](vals)[:, :3] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/nuscenes_extra/reflection_and_transmission_multi_frame")
    ap.add_argument("--labels_root", default="data/nuscenes_extra",
                    help="Root with evidence/ and occ3d_transfer/ (enables class/uncertainty coloring).")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--p_fn", type=float, default=0.8)
    ap.add_argument("--p_fp", type=float, default=0.2)
    ap.add_argument("--point_size", type=float, default=0.18)
    ap.add_argument("--port", type=int, default=int(os.environ.get("VISER_PORT", "8080")))
    args = ap.parse_args()

    files = discover_files(Path(args.data_dir))
    if not files:
        raise SystemExit(f"No .arrow files under {args.data_dir}. Run temporal-accumulation first.")
    labels_root = Path(args.labels_root) if args.labels_root else None
    picked = pick_samples(files, args.num_samples)
    print(f"Found {len(files)} accumulated samples; showing {len(picked)}:")
    samples = []
    for p in picked:
        s = load_sample(p, labels_root)
        samples.append(s)
        print(f"  - {s['scene']}/{s['token']}  occ3d_labels={'yes' if s['sem'] is not None else 'no'}")
    z_range = (float(samples[0]["lower"][2]), float(samples[0]["upper"][2]))
    have_sem = any(s["sem"] is not None for s in samples)

    server = viser.ViserServer(host=os.environ.get("VISER_HOST", "0.0.0.0"), port=args.port)
    server.scene.add_frame("/ego", show_axes=True, axes_length=3.0, axes_radius=0.1)
    server.scene.add_grid("/ground", width=80.0, height=80.0, position=(0.0, 0.0, z_range[0]))
    pc = server.scene.add_point_cloud("/occupancy", points=np.zeros((1, 3), np.float32),
                                      colors=np.zeros((1, 3), np.uint8),
                                      point_size=args.point_size, point_shape="square")

    server.gui.add_markdown("## Evidential occupancy + labels\n"
                            "Color by height / uncertainty (m_omega) / Occ3D class. "
                            "Tune `p_fn`/`p_fp` for the occupancy threshold.")
    info = server.gui.add_markdown("")
    with server.gui.add_folder("Navigate samples"):
        prev_b = server.gui.add_button("Prev", icon=viser.Icon.CHEVRON_LEFT)
        next_b = server.gui.add_button("Next", icon=viser.Icon.CHEVRON_RIGHT)
        sld = server.gui.add_slider("Sample", min=0, max=len(samples) - 1, step=1, initial_value=0)
    with server.gui.add_folder("Belief thresholds"):
        p_fn_sld = server.gui.add_slider("p_fn", min=0.5, max=0.99, step=0.01, initial_value=args.p_fn)
        p_fp_sld = server.gui.add_slider("p_fp", min=0.01, max=0.5, step=0.01, initial_value=args.p_fp)
    with server.gui.add_folder("Display"):
        default_mode = "occ3d_class" if have_sem else "height"
        color_mode = server.gui.add_dropdown("Color by", options=COLOR_MODES, initial_value=default_mode)
        psize = server.gui.add_slider("Point size", min=0.02, max=0.4, step=0.01, initial_value=args.point_size)

    state = {"idx": 0}

    def render():
        s = samples[state["idx"]]
        pts, conf, omega, cls = occupancy(s, p_fn_sld.value, p_fp_sld.value)
        mode = color_mode.value
        if mode == "occ3d_class" and cls is None:
            mode = "height"
        if len(pts) == 0:
            pts, cols = np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8)
        else:
            cols = colorize(pts, conf, omega, cls, mode, z_range)
        pc.points, pc.colors, pc.point_size = pts, cols, psize.value
        present = ""
        if cls is not None:
            uniq = sorted(set(int(c) for c in np.unique(cls)))
            present = ", ".join(CLASS_NAMES[c] for c in uniq if c != 17)
        info.content = (
            f"### Sample {state['idx'] + 1}/{len(samples)}\n"
            f"**scene:** `{s['scene']}`\n\n**token:** `{s['token']}`\n\n"
            f"**occupied voxels:** {len(pts):,}\n\n"
            f"**mean m_omega:** {float(omega.mean()):.3f}\n\n"
            f"**Occ3D classes:** {present or '(run occ3d-transfer)'}"
        )
        print(f"sample {state['idx']+1}/{len(samples)}: {s['scene']}/{s['token']} occ={len(pts)} mode={mode}")

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
