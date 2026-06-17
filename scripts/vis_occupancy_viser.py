#!/usr/bin/env python
"""Interactive 3D (viser) viewer for the evidential-occupancy output and its labels.

Reads the accumulated reflection/transmission volumes (``temporal-accumulation``) and turns
them into occupancy via Dempster-Shafer belief. If the per-stage label outputs exist under
``--labels_root`` (``evidence/``, ``occ3d_transfer/``), the occupied voxels can also be
colored by **epistemic uncertainty** (m_omega) or **Occ3D semantic class**. If the nuScenes
dataset is reachable, the 6 surround-camera **source images** for the current key-frame are
shown in the GUI.

Usage (inside the pixi env):
    pixi run python scripts/vis_occupancy_viser.py --num_samples 5
    pixi run python scripts/vis_occupancy_viser.py \
        --labels_root data/nuscenes_extra --nuscenes data/nuscenes --num_samples 5

Then open the printed URL (default http://localhost:8080).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import polars as pl
import torch
import viser
from PIL import Image

from scene_reconstruction.core.volume import Volume
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch
from scene_reconstruction.math.dempster_shafer import belief_from_reflection_and_transmission_stacked
from scene_reconstruction.visualization.colormap import occupancy_color_map, occupancy_to_color

RT_COL = "LIDAR_TOP.reflection_and_transmission_multi_frame"
LOWER_COL = f"{RT_COL}.volume.lower"
UPPER_COL = f"{RT_COL}.volume.upper"
CLASS_NAMES = list(occupancy_to_color.keys())  # index -> name, 0=other .. 17=free
FREE_CLASS = 17
FREE_GRAY = np.array([120, 120, 120], dtype=np.uint8)  # how we show occupied-but-unlabeled voxels
COLOR_MODES = ("height", "confidence", "uncertainty", "occ3d_class", "box_class", "locc_class", "locc_project")
BG_GRAY = np.array([50, 50, 50], dtype=np.uint8)  # background (no box) in box_class mode
# surround cameras, laid out front row then back row
CAM_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# LOcc open-vocab labels use their own (larger) vocabulary; names loaded from locc_raw/vocab.json,
# colored by a distinct 60-colour palette (-1 = unlabeled -> gray).
LOCC_NAMES: dict[int, str] = {}
LOCC_PALETTE = np.concatenate(
    [(np.array([mpl.colormaps[c](i)[:3] for i in range(20)]) * 255).astype(np.uint8)
     for c in ("tab20", "tab20b", "tab20c")]
)


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


def _maybe_load_occ3d(labels_root: Path, scene: str, token: str):
    """(semantics, fill_distance) [X,Y,Z] from occ3d-transfer, or (None, None) if not generated."""
    f = labels_root / "occ3d_transfer" / scene / "LIDAR_TOP" / f"{token}.arrow"
    if not f.exists():
        return None, None
    df = pl.read_ipc(f, memory_map=False)
    sem = series_to_torch(df["LIDAR_TOP.occ3d_transfer.semantics"])[0].to(torch.int64)
    fdist = series_to_torch(df["LIDAR_TOP.occ3d_transfer.fill_distance"])[0].float()
    return sem, fdist


def _maybe_load_box(labels_root: Path, scene: str, token: str) -> torch.Tensor | None:
    """Box-semantics category_index [X,Y,Z] (0=background, 1..10 foreground), or None."""
    f = labels_root / "box_semantics" / scene / "LIDAR_TOP" / f"{token}.arrow"
    if not f.exists():
        return None
    df = pl.read_ipc(f, memory_map=False)
    return series_to_torch(df["LIDAR_TOP.box_semantics.category_index"])[0].to(torch.int64)


def _maybe_load_locc(labels_root: Path, scene: str, token: str) -> torch.Tensor | None:
    """LOcc open-vocab class index [X,Y,Z] (-1 = empty/unlabeled), or None."""
    f = labels_root / "locc_transfer" / scene / "LIDAR_TOP" / f"{token}.arrow"
    if not f.exists():
        return None
    df = pl.read_ipc(f, memory_map=False)
    return series_to_torch(df["LIDAR_TOP.locc_transfer.semantics"])[0].to(torch.int64)


def _maybe_load_locc_project(labels_root: Path, scene: str, token: str) -> torch.Tensor | None:
    """Open-vocab class from projecting our occupancy into the OV-Seg images (-1 = unlabeled)."""
    f = labels_root / "locc_project" / scene / "LIDAR_TOP" / f"{token}.arrow"
    if not f.exists():
        return None
    df = pl.read_ipc(f, memory_map=False)
    return series_to_torch(df["LIDAR_TOP.locc_project.semantics"])[0].to(torch.int64)


def load_sample(path: Path, labels_root: Path | None):
    """Load one key-frame: RT volume, bounds, scene/token, and optional label grids."""
    df = pl.read_ipc(path, memory_map=False)
    rt = series_to_torch(df[RT_COL])[0].float()  # [2, X, Y, Z]
    lower = series_to_torch(df[LOWER_COL])[0].float()
    upper = series_to_torch(df[UPPER_COL])[0].float()
    token = df["LIDAR_TOP.sample_data.token"].item()
    scene = path.parent.parent.name
    sem, fdist = _maybe_load_occ3d(labels_root, scene, token) if labels_root is not None else (None, None)
    box = _maybe_load_box(labels_root, scene, token) if labels_root is not None else None
    locc = _maybe_load_locc(labels_root, scene, token) if labels_root is not None else None
    locc_proj = _maybe_load_locc_project(labels_root, scene, token) if labels_root is not None else None
    return {"rt": rt, "lower": lower, "upper": upper, "scene": scene, "token": token,
            "sem": sem, "fdist": fdist, "box": box, "locc": locc, "locc_proj": locc_proj}


def occ3d_source_points(sample):
    """Original Occ3D occupancy = voxels with fill_distance==0 (real Occ3D voxels), + their class."""
    sem, fdist = sample["sem"], sample["fdist"]
    if sem is None or fdist is None:
        return None
    mask = fdist == 0  # the densified field marks real Occ3D voxels with distance 0
    volume = Volume.new_volume(sample["lower"].tolist(), sample["upper"].tolist())
    grid = volume.new_coord_grid(tuple(sem.shape))[0]  # [X, Y, Z, 3]
    return grid[mask].numpy().astype(np.float32), sem[mask].numpy().astype(np.int64)


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
    box = sample["box"][occ].numpy().astype(np.int64) if sample["box"] is not None else None
    locc = sample["locc"][occ].numpy().astype(np.int64) if sample["locc"] is not None else None
    locc_proj = sample["locc_proj"][occ].numpy().astype(np.int64) if sample["locc_proj"] is not None else None
    return pts, conf, omega, cls, box, locc, locc_proj


def class_colors(cls: np.ndarray) -> np.ndarray:
    """Occ3D class -> RGB, with free/unlabeled (17) shown gray instead of the palette blue."""
    cols = occupancy_color_map[np.clip(cls, 0, len(CLASS_NAMES) - 1)].numpy().astype(np.uint8)
    cols[cls == FREE_CLASS] = FREE_GRAY
    return cols


def box_colors(box: np.ndarray) -> np.ndarray:
    """Box foreground class -> RGB, background (0) shown dim gray."""
    cols = occupancy_color_map[np.clip(box, 0, len(CLASS_NAMES) - 1)].numpy().astype(np.uint8)
    cols[box == 0] = BG_GRAY
    return cols


def locc_colors(locc: np.ndarray) -> np.ndarray:
    """LOcc open-vocab class -> RGB, unlabeled (-1) shown gray."""
    cols = LOCC_PALETTE[np.clip(locc, 0, len(LOCC_PALETTE) - 1)].copy()
    cols[locc == -1] = FREE_GRAY
    return cols


def colorize(pts, conf, omega, cls, mode, z_range, box=None, locc=None, locc_proj=None) -> np.ndarray:
    """Map per-point scalars/classes -> uint8 RGB."""
    if mode == "occ3d_class" and cls is not None:
        return class_colors(cls)
    if mode == "box_class" and box is not None:
        return box_colors(box)
    if mode == "locc_class" and locc is not None:
        return locc_colors(locc)
    if mode == "locc_project" and locc_proj is not None:
        return locc_colors(locc_proj)
    if mode == "uncertainty":
        return (mpl.colormaps["turbo"](np.clip(omega, 0.0, 1.0))[:, :3] * 255).astype(np.uint8)
    if mode == "confidence":
        return (mpl.colormaps["viridis"](np.clip(conf, 0.0, 1.0))[:, :3] * 255).astype(np.uint8)
    z0, z1 = z_range  # height
    vals = np.clip((pts[:, 2] - z0) / max(z1 - z0, 1e-6), 0.0, 1.0)
    return (mpl.colormaps["turbo"](vals)[:, :3] * 255).astype(np.uint8)


def class_swatch(mode: str, c: int):
    """(rgb in [0,1], name) for a class index in a class color mode."""
    if mode in ("locc_class", "locc_project"):
        if c == -1:
            return FREE_GRAY / 255.0, "unlabeled"
        return LOCC_PALETTE[c % len(LOCC_PALETTE)] / 255.0, LOCC_NAMES.get(c, str(c))
    if mode == "box_class" and c == 0:
        return BG_GRAY / 255.0, "background"
    if mode == "occ3d_class" and c == FREE_CLASS:
        return FREE_GRAY / 255.0, "free / no Occ3D label"
    return occupancy_color_map[c].numpy() / 255.0, CLASS_NAMES[c]


# colormap + scalar-field label per scalar color mode (must match `colorize`)
SCALAR_CMAP = {"uncertainty": "turbo", "confidence": "viridis", "height": "turbo"}
SCALAR_LABEL = {"uncertainty": "m_omega (epistemic uncertainty)", "confidence": "m_o - m_f", "height": "height z (m)"}


def legend_image(mode: str, present_classes: list[int], scalar_range: tuple[float, float]) -> np.ndarray:
    """Render a color guide: class swatches for occ3d_class, a colorbar for scalar modes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if mode in ("occ3d_class", "box_class", "locc_class", "locc_project"):
        classes = present_classes or [0]
        fig, ax = plt.subplots(figsize=(2.8, 0.30 * len(classes) + 0.2))
        ax.axis("off")
        for i, c in enumerate(classes):
            y = len(classes) - 1 - i
            color, name = class_swatch(mode, c)
            ax.add_patch(Rectangle((0, y), 0.8, 0.8, facecolor=color, edgecolor="k", lw=0.4))
            ax.text(1.0, y + 0.4, name, va="center", fontsize=8)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, len(classes))
    else:
        grad = np.linspace(scalar_range[0], scalar_range[1], 256)[None, :]
        fig, ax = plt.subplots(figsize=(2.8, 0.7))
        ax.imshow(grad, aspect="auto", cmap=SCALAR_CMAP[mode], extent=[scalar_range[0], scalar_range[1], 0, 1])
        ax.set_yticks([])
        ax.set_title(SCALAR_LABEL[mode], fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def detect_version(nuscenes_root: Path, samples) -> str:
    """Pick the nuScenes version whose scene.json contains all displayed scenes."""
    needed = {s["scene"] for s in samples}
    for v in ("v1.0-mini", "v1.0-trainval"):
        f = nuscenes_root / v / "scene.json"
        if f.exists():
            names = {row["name"] for row in json.load(open(f))}
            if needed <= names:
                return v
    return "v1.0-trainval"


def build_camera_map(samples, nuscenes_root: Path, version: str) -> dict[str, dict[str, str]]:
    """{lidar_token: {camera: image_path}} for the displayed key-frames."""
    from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset

    ds = NuscenesDataset(data_root=nuscenes_root, version=version, key_frames_only=True)
    scene_names = sorted({s["scene"] for s in samples})
    scene = ds.get_scene(ds.scene.filter(pl.col("scene.name").is_in(scene_names)))
    scene = ds.load_sample_data(scene, "LIDAR_TOP", with_data=False)
    scene = scene.filter(pl.col("LIDAR_TOP.sample_data.is_key_frame"))
    for cam in CAM_ORDER:
        scene = ds.load_sample_data(scene, cam, with_data=False)
    cmap: dict[str, dict[str, str]] = {}
    for row in scene.iter_rows(named=True):
        tok = row["LIDAR_TOP.sample_data.token"]
        cmap[tok] = {
            cam: str(nuscenes_root / row[f"{cam}.sample_data.filename"])
            for cam in CAM_ORDER
            if row.get(f"{cam}.sample_data.filename")
        }
    return cmap


def load_image_downscaled(path: str, width: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > width:
        img = img.resize((width, max(1, round(h * width / w))))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/nuscenes_extra/reflection_and_transmission_multi_frame")
    ap.add_argument("--labels_root", default="data/nuscenes_extra",
                    help="Root with evidence/ and occ3d_transfer/ (enables class/uncertainty coloring).")
    ap.add_argument("--nuscenes", default="data/nuscenes", help="nuScenes root for source camera images.")
    ap.add_argument("--nuscenes_version", default="", help="v1.0-mini / v1.0-trainval (auto-detect if empty).")
    ap.add_argument("--no_cameras", action="store_true", help="Disable the source-camera panel.")
    ap.add_argument("--cam_width", type=int, default=360, help="Downscaled camera-image width (px).")
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
    if labels_root is not None:
        # locc_transfer and locc_project share the SAN vocabulary; load whichever vocab.json exists.
        for vocab_json in (labels_root / "locc_raw" / "vocab.json", labels_root / "locc_project" / "vocab.json"):
            if vocab_json.exists():
                LOCC_NAMES.update({int(k): v for k, v in json.load(open(vocab_json)).items()})
    picked = pick_samples(files, args.num_samples)
    print(f"Found {len(files)} accumulated samples; showing {len(picked)}:")
    samples = []
    for p in picked:
        s = load_sample(p, labels_root)
        samples.append(s)
        print(f"  - {s['scene']}/{s['token']}  occ3d_labels={'yes' if s['sem'] is not None else 'no'}")
    z_range = (float(samples[0]["lower"][2]), float(samples[0]["upper"][2]))
    have_sem = any(s["sem"] is not None for s in samples)

    # optional source-camera index
    camera_map: dict[str, dict[str, str]] = {}
    cam_enabled = (not args.no_cameras) and Path(args.nuscenes).exists()
    if cam_enabled:
        nuroot = Path(args.nuscenes)
        version = args.nuscenes_version or detect_version(nuroot, samples)
        try:
            camera_map = build_camera_map(samples, nuroot, version)
            print(f"Loaded source-camera index for {len(camera_map)} key-frames (nuScenes {version}).")
        except Exception as exc:  # noqa: BLE001
            print(f"Source-camera panel disabled ({type(exc).__name__}: {exc}).")
            cam_enabled = False

    server = viser.ViserServer(host=os.environ.get("VISER_HOST", "0.0.0.0"), port=args.port)
    server.scene.add_frame("/ego", show_axes=True, axes_length=3.0, axes_radius=0.1)
    server.scene.add_grid("/ground", width=80.0, height=80.0, position=(0.0, 0.0, z_range[0]))
    pc = server.scene.add_point_cloud("/occupancy", points=np.zeros((1, 3), np.float32),
                                      colors=np.zeros((1, 3), np.uint8),
                                      point_size=args.point_size, point_shape="square")

    server.gui.add_markdown("## Evidential occupancy + labels\n"
                            "Color by height / uncertainty (m_omega) / Occ3D class. Gray = occupied but "
                            "free/unlabeled in Occ3D. Tune `p_fn`/`p_fp` for the occupancy threshold.")
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
        hide_free = server.gui.add_checkbox("Hide free/unlabeled", initial_value=False)
        show_orig = server.gui.add_checkbox("Show original Occ3D", initial_value=False)
        psize = server.gui.add_slider("Point size", min=0.02, max=0.4, step=0.01, initial_value=args.point_size)
    legend_folder = server.gui.add_folder("Color guide")
    cam_folder = server.gui.add_folder("Source cameras") if cam_enabled else None

    state = {"idx": 0, "legend": None, "legend_key": None, "cam_handles": []}

    def update_legend(mode, present_classes, scalar_range):
        is_class = mode in ("occ3d_class", "box_class", "locc_class", "locc_project")
        key = (mode, tuple(present_classes)) if is_class else (mode, scalar_range)
        if key == state["legend_key"]:
            return
        state["legend_key"] = key
        if state["legend"] is not None:
            state["legend"].remove()
        with legend_folder:
            state["legend"] = server.gui.add_image(legend_image(mode, present_classes, scalar_range), label=None)

    def update_cameras():
        if not cam_enabled:
            return
        for h in state["cam_handles"]:
            h.remove()
        state["cam_handles"].clear()
        cams = camera_map.get(samples[state["idx"]]["token"], {})
        with cam_folder:
            for cam in CAM_ORDER:
                path = cams.get(cam)
                if not path or not Path(path).exists():
                    continue
                try:
                    img = load_image_downscaled(path, args.cam_width)
                except Exception:  # noqa: BLE001
                    continue
                state["cam_handles"].append(server.gui.add_image(img, label=cam))

    def render():
        s = samples[state["idx"]]
        if show_orig.value and s["fdist"] is not None:
            pts, cls = occ3d_source_points(s)
            if len(pts) == 0:
                pc.points, pc.colors, pc.point_size = np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), psize.value
                return
            pc.points, pc.colors, pc.point_size = pts, class_colors(cls), psize.value
            present_classes = sorted(int(c) for c in np.unique(cls))
            update_legend("occ3d_class", present_classes, (0.0, 1.0))
            named = ", ".join(("free/unlabeled" if c == FREE_CLASS else CLASS_NAMES[c]) for c in present_classes)
            info.content = (
                f"### Sample {state['idx'] + 1}/{len(samples)} — **original Occ3D (source GT)**\n"
                f"**scene:** `{s['scene']}`\n\n**token:** `{s['token']}`\n\n"
                f"**Occ3D voxels:** {len(pts):,} (0.4 m)\n\n**classes:** {named}"
            )
            print(f"sample {state['idx']+1}/{len(samples)}: ORIGINAL Occ3D voxels={len(pts)}")
            return
        pts, conf, omega, cls, box, locc, locc_proj = occupancy(s, p_fn_sld.value, p_fp_sld.value)
        mode = color_mode.value
        if ((mode == "occ3d_class" and cls is None) or (mode == "box_class" and box is None)
                or (mode == "locc_class" and locc is None) or (mode == "locc_project" and locc_proj is None)):
            mode = "height"
        active = {"occ3d_class": cls, "box_class": box, "locc_class": locc, "locc_project": locc_proj}.get(mode)
        unlabeled = {"occ3d_class": FREE_CLASS, "box_class": 0, "locc_class": -1, "locc_project": -1}.get(mode)
        if hide_free.value and active is not None:
            keep = active != unlabeled
            pts, conf, omega = pts[keep], conf[keep], omega[keep]
            cls = cls[keep] if cls is not None else None
            box = box[keep] if box is not None else None
            locc = locc[keep] if locc is not None else None
            locc_proj = locc_proj[keep] if locc_proj is not None else None
            active = active[keep]
        if len(pts) == 0:
            pc.points, pc.colors, pc.point_size = np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), psize.value
            return
        pc.points, pc.colors, pc.point_size = (
            pts, colorize(pts, conf, omega, cls, mode, z_range, box, locc, locc_proj), psize.value)

        present_classes = sorted(int(c) for c in np.unique(active)) if active is not None else []
        named = ", ".join(class_swatch(mode, c)[1] for c in present_classes)
        scalar_range = z_range if mode == "height" else (0.0, 1.0)
        update_legend(mode, present_classes, scalar_range)
        label = {"occ3d_class": "Occ3D classes", "box_class": "box classes",
                 "locc_class": "LOcc classes", "locc_project": "LOcc (projected) classes"}.get(mode, "classes")
        info.content = (
            f"### Sample {state['idx'] + 1}/{len(samples)}\n"
            f"**scene:** `{s['scene']}`\n\n**token:** `{s['token']}`\n\n"
            f"**occupied voxels:** {len(pts):,}\n\n"
            f"**mean m_omega:** {float(omega.mean()):.3f}\n\n"
            f"**{label}:** {named or '(stage not run)'}"
        )
        print(f"sample {state['idx']+1}/{len(samples)}: {s['scene']}/{s['token']} occ={len(pts)} mode={mode}")

    def show(i):
        state["idx"] = int(np.clip(i, 0, len(samples) - 1))
        sld.value = state["idx"]
        render()
        update_cameras()

    @prev_b.on_click
    def _(_):
        show(state["idx"] - 1)

    @next_b.on_click
    def _(_):
        show(state["idx"] + 1)

    @sld.on_update
    def _(_):
        show(int(sld.value))

    for ctrl in (p_fn_sld, p_fp_sld, color_mode, hide_free, show_orig, psize):
        ctrl.on_update(lambda _: render())

    show(0)
    print(f"\nviser running — open http://localhost:{args.port}  (Ctrl+C to exit)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
