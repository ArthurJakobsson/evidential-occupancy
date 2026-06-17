"""Open-vocabulary labels by projecting OUR occupancy into LOcc's OV-Seg images.

The ``locc-transfer`` stage consumes LOcc's *own* occupancy (LOcc voxelizes lidar points and
labels only the voxels a lidar return landed in), so any evidential voxel LOcc's lidar pass
missed — including low-confidence ones — ends up unlabeled.

This stage decouples occupancy from labeling, like LOcc's
``PseudoOccGeneration-VoxelProjection.py`` but driven by *our* geometry: for every voxel our
evidence marks ``occupied``, it projects the voxel CENTER (ego frame) into the SAN open-vocab
segmentation of each camera and reads the class at that pixel.

TEMPORAL AGGREGATION (``temporal=True``, default). A single frame only labels what its cameras
see *now*: the patch under/around the ego (lidar+camera blind spot) stays unlabeled, and labels
flicker frame to frame. So after the per-frame projection we pool labels across the scene's
key-frames and vote, splitting static scene from objects exactly like LOcc's point-based GT:
  * **background** voxels (``scene_flow.scene_instance_index == 0``) are pooled in the **global
    frame** (via ``global_from_ego``) and majority-voted per global voxel — so the ground under
    the ego at frame *t* inherits the label a camera gave it at some other frame, and static
    structure is temporally consistent.
  * **boxed-object** voxels (instance id > 0, scene-global from scene-flow) are pooled
    **per-instance** and voted once per object, then written at each frame's (de-warped)
    position. Objects therefore never smear a trail of their label across the world frame.

Inputs (per LIDAR key-frame): our ``evidence`` occupancy + bounds, the ``scene_flow`` instance
grid, and SAN OV-Seg PNGs at ``<seg_root>/<camera filename>.png`` (uint8 class, 255 = none).
Output ``<extra>/locc_project/<scene>/LIDAR_TOP/<token>.arrow`` with
``LIDAR_TOP.locc_project.semantics`` int16 [1,400,400,32] (``-1`` = unlabeled), native 0.2 m.
``vocab.json`` (class id -> name) is written alongside.

CAVEAT (intentional, per design): voxel-center projection has no occlusion test, so a voxel
occluded from a camera inherits whatever surface is in front of it. A depth gate (keep the label
only where the voxel's projected depth matches the first surface) is the natural fix if needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import polars as pl
import torch
import tqdm
from PIL import Image

from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch, torch_to_series

# nuScenes surround cameras (360 deg coverage).
CAMERAS: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
INSTANCE_COL = "LIDAR_TOP.scene_flow.scene_instance_index"


@dataclass
class LoccProject:
    """Label our occupied voxels by projecting them into the SAN OV-Seg images, then temporally pool."""

    ds: NuscenesDataset
    extra_data_root: Union[Path, str]
    # SAN OV-Seg output dir; segmentation PNG for image ``f`` lives at ``seg_root/<f with .png>``.
    seg_root: Union[Path, str] = ""
    # LOcc vocab.txt (one class name per line); copied to ``locc_project/vocab.json`` and used to
    # bound class ids. Empty -> only the background value is treated as unlabeled.
    vocab_file: str = ""
    cameras: Sequence[str] = field(default_factory=lambda: CAMERAS)
    seg_background: int = 255  # SAN "no class" value -> unlabeled
    locc_free: int = -1
    # Pool labels across the scene's key-frames (fills blind spots + temporal consistency).
    temporal: bool = True
    global_voxel_size: float = 0.2  # quantization for the global-frame background vote
    evidence_name: str = "evidence"
    scene_flow_name: str = "scene_flow"
    name: str = "locc_project"
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None

    def evidence_path(self, scene_name: str, token: str) -> Path:
        return Path(self.extra_data_root) / self.evidence_name / scene_name / "LIDAR_TOP" / f"{token}.arrow"

    def scene_flow_path(self, scene_name: str, token: str) -> Path:
        return Path(self.extra_data_root) / self.scene_flow_name / scene_name / "LIDAR_TOP" / f"{token}.arrow"

    def save_path(self, scene_name: str, token: str) -> Path:
        path = Path(self.extra_data_root) / self.name / scene_name / "LIDAR_TOP" / f"{token}.arrow"
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def _seg_path(self, filename: str) -> Path:
        return Path(self.seg_root) / filename.replace(".jpg", ".png")

    def _write_vocab(self) -> Optional[int]:
        """Copy the SAN vocab to locc_project/vocab.json; return the class count (or None)."""
        if not self.vocab_file or not Path(self.vocab_file).exists():
            return None
        vocab = [w.strip() for w in open(self.vocab_file) if w.strip()]
        out = Path(self.extra_data_root) / self.name / "vocab.json"
        out.parent.mkdir(exist_ok=True, parents=True)
        out.write_text(json.dumps({i: w for i, w in enumerate(vocab)}, indent=0))
        return len(vocab)

    @staticmethod
    def _load_seg(path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        seg = np.asarray(Image.open(path))
        return seg[..., 0] if seg.ndim == 3 else seg  # [H, W] uint8

    @staticmethod
    def _project(centers: np.ndarray, proj: np.ndarray, w: int, h: int):
        """centers [N,3] (ego) -> (uv [N,2], in-image mask [N]) under proj (image_from_ego, 4x4)."""
        homog = np.concatenate([centers, np.ones((centers.shape[0], 1), centers.dtype)], axis=1)
        cam = homog @ proj.T  # [N,4]
        depth = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = cam[:, :2] / depth[:, None]
        valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        return uv, valid

    def _label_voxels(self, centers: np.ndarray, cam_proj_seg: list[tuple[np.ndarray, np.ndarray]],
                      num_classes: Optional[int]) -> np.ndarray:
        """Per voxel: OV-Seg class from the camera whose view-center it lands nearest.

        Only labelled (non-background, in-vocab) hits count, so a voxel that projects onto sky in
        one camera but a real object in another keeps the object label. ``-1`` = no labelled hit.
        """
        n = centers.shape[0]
        classes = np.full(n, self.locc_free, dtype=np.int16)
        best_d2 = np.full(n, np.inf, dtype=np.float64)
        for proj, seg in cam_proj_seg:
            h, w = seg.shape
            cx, cy = w / 2.0, h / 2.0
            uv, valid = self._project(centers, proj, w, h)
            idx = np.nonzero(valid)[0]
            if idx.size == 0:
                continue
            u, v = uv[idx, 0], uv[idx, 1]
            cls = seg[v.astype(np.int64), u.astype(np.int64)].astype(np.int16)
            labelled = cls != self.seg_background
            if num_classes is not None:
                labelled &= cls < num_classes
            if not labelled.any():
                continue
            idx, u, v, cls = idx[labelled], u[labelled], v[labelled], cls[labelled]
            d2 = (u - cx) ** 2 + (v - cy) ** 2
            take = d2 < best_d2[idx]
            sel = idx[take]
            classes[sel] = cls[take]
            best_d2[sel] = d2[take]
        return classes

    @staticmethod
    def _majority(keys: np.ndarray, labels: np.ndarray, num_classes: int):
        """Majority label per key. Returns (sorted unique keys [U], winning label per key [U])."""
        combo = keys * num_classes + labels.astype(np.int64)
        uc, cnt = np.unique(combo, return_counts=True)  # ascending by (key, label)
        ukey = uc // num_classes
        ulabel = (uc % num_classes).astype(np.int16)
        # within each key group (sorted), order by count so the max-count label is last
        order = np.lexsort((cnt, ukey))
        ukey_s, ulabel_s = ukey[order], ulabel[order]
        ends = np.concatenate([np.nonzero(np.diff(ukey_s))[0], [len(ukey_s) - 1]]) if len(ukey_s) else np.array([], int)
        return ukey_s[ends], ulabel_s[ends]

    def _global_keys(self, centers_ego: np.ndarray, global_from_ego: np.ndarray) -> np.ndarray:
        """Quantized global-voxel key per voxel (stable across frames)."""
        homog = np.concatenate([centers_ego, np.ones((centers_ego.shape[0], 1), centers_ego.dtype)], axis=1)
        g = (homog @ global_from_ego.T)[:, :3]
        gi = np.floor(g / self.global_voxel_size).astype(np.int64)
        b = 1 << 13  # offset -> non-negative; 14 bits/axis (scene extent well within +-1.6 km)
        return ((gi[:, 0] + b) * (1 << 14) + (gi[:, 1] + b)) * (1 << 14) + (gi[:, 2] + b)

    @staticmethod
    def _lookup(keys: np.ndarray, table_keys: np.ndarray, table_vals: np.ndarray, default: np.ndarray) -> np.ndarray:
        """Vectorized map: keys -> table_vals where table_keys matches, else ``default``."""
        out = default.copy()
        if table_keys.size == 0:
            return out
        pos = np.searchsorted(table_keys, keys)
        pos = np.clip(pos, 0, table_keys.size - 1)
        hit = table_keys[pos] == keys
        out[hit] = table_vals[pos[hit]]
        return out

    def _frame_data(self, sample: pl.DataFrame, num_classes: Optional[int]):
        """Per-frame projection. Returns dict with occ indices, per-frame labels, instance ids, global keys."""
        scene_name = sample["scene.name"].item()
        token = sample["LIDAR_TOP.sample_data.token"].item()
        ev_path = self.evidence_path(scene_name, token)
        if not ev_path.exists():
            return None
        cam_proj_seg: list[tuple[np.ndarray, np.ndarray]] = []
        for cam in self.cameras:
            seg = self._load_seg(self._seg_path(sample[f"{cam}.sample_data.filename"].item()))
            if seg is None:
                continue
            image_from_sensor = series_to_torch(sample[f"{cam}.transform.image_from_sensor"])[0].numpy()
            sensor_from_ego = series_to_torch(sample[f"{cam}.transform.sensor_from_ego"])[0].numpy()
            cam_proj_seg.append((image_from_sensor @ sensor_from_ego, seg))
        if not cam_proj_seg:
            return None

        ev = pl.read_ipc(ev_path, memory_map=False)
        occupied = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.occupied"])[0].bool().numpy()
        lower = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.volume.lower"])[0].numpy()
        upper = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.volume.upper"])[0].numpy()
        occ_idx = np.argwhere(occupied)  # [N,3]
        out = {"scene_name": scene_name, "token": token, "shape": occupied.shape, "occ_idx": occ_idx}
        if occ_idx.shape[0] == 0:
            out.update(labels=np.empty(0, np.int16), inst=np.empty(0, np.int64), gkey=np.empty(0, np.int64))
            return out

        voxel_size = (upper - lower) / np.array(occupied.shape, dtype=np.float64)
        centers = lower + (occ_idx.astype(np.float64) + 0.5) * voxel_size  # ego frame [N,3]
        labels = self._label_voxels(centers, cam_proj_seg, num_classes)

        # scene-flow instance id per occupied voxel (0 = background); fallback to all-background
        sf_path = self.scene_flow_path(scene_name, token)
        if sf_path.exists():
            inst_grid = series_to_torch(pl.read_ipc(sf_path, memory_map=False)[INSTANCE_COL])[0].to(torch.int64).numpy()
            inst = inst_grid[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]]
        else:
            inst = np.zeros(occ_idx.shape[0], dtype=np.int64)

        gfe = series_to_torch(sample["LIDAR_TOP.transform.global_from_ego"])[0].numpy()
        out.update(labels=labels, inst=inst, gkey=self._global_keys(centers, gfe))
        return out

    def _write_frame(self, frame: dict, semantics_flat: np.ndarray) -> None:
        sem = np.full((1,) + frame["shape"], self.locc_free, dtype=np.int16)
        occ_idx = frame["occ_idx"]
        if occ_idx.shape[0] > 0:
            sem[0, occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = semantics_flat
        out_df = pl.DataFrame({"LIDAR_TOP.sample_data.token": [frame["token"]]}).with_columns(
            torch_to_series(f"LIDAR_TOP.{self.name}.semantics", torch.from_numpy(sem)),
        )
        out_df.write_ipc(self.save_path(frame["scene_name"], frame["token"]), compression="zstd")

    def process_scene(self, scene: pl.DataFrame, num_classes: Optional[int]) -> None:
        scene = self.ds.join(scene, self.ds.sample)
        scene = self.ds.load_sample_data(scene, "LIDAR_TOP", with_data=False)
        scene = scene.filter(pl.col("LIDAR_TOP.sample_data.is_key_frame"))
        if len(scene) == 0:
            return
        scene_name = scene["scene.name"][0]
        if self.missing_only and all(
            self.save_path(scene_name, t).exists() for t in scene["LIDAR_TOP.sample_data.token"]
        ):
            return
        for cam in self.cameras:
            scene = self.ds.load_sample_data(scene, cam, with_data=False)

        # Pass 1: per-frame projection + vote accumulation.
        frames: list[dict] = []
        bg_keys, bg_labels, obj_inst, obj_labels = [], [], [], []
        for sample in tqdm.tqdm(scene.iter_slices(1), total=len(scene), desc=scene_name, position=1, leave=False):
            f = self._frame_data(sample, num_classes)
            if f is None:
                continue
            frames.append(f)
            if self.temporal and f["occ_idx"].shape[0]:
                lab = f["labels"]
                seen = lab >= 0
                is_bg = (f["inst"] == 0) & seen
                is_obj = (f["inst"] > 0) & seen
                bg_keys.append(f["gkey"][is_bg]); bg_labels.append(lab[is_bg])
                obj_inst.append(f["inst"][is_obj]); obj_labels.append(lab[is_obj])

        if not frames:
            return

        if not self.temporal or num_classes is None:
            for f in frames:
                if not (self.missing_only and self.save_path(f["scene_name"], f["token"]).exists()):
                    self._write_frame(f, f["labels"])
            return

        # Resolve scene-wide votes: background per global voxel, objects per instance.
        bg_k, bg_v = self._majority(np.concatenate(bg_keys), np.concatenate(bg_labels), num_classes) \
            if bg_keys else (np.array([], np.int64), np.array([], np.int16))
        obj_k, obj_v = self._majority(np.concatenate(obj_inst), np.concatenate(obj_labels), num_classes) \
            if obj_inst else (np.array([], np.int64), np.array([], np.int16))

        # Pass 2: write each frame with the pooled labels (per-frame projection as fallback).
        for f in frames:
            if self.missing_only and self.save_path(f["scene_name"], f["token"]).exists():
                continue
            n = f["occ_idx"].shape[0]
            sem = f["labels"].copy() if n else np.empty(0, np.int16)
            if n:
                is_bg = f["inst"] == 0
                sem[is_bg] = self._lookup(f["gkey"][is_bg], bg_k, bg_v, sem[is_bg])
                sem[~is_bg] = self._lookup(f["inst"][~is_bg], obj_k, obj_v, sem[~is_bg])
            self._write_frame(f, sem)

    def process_data(self) -> None:
        if not self.seg_root or not Path(self.seg_root).exists():
            print(f"[locc-project] seg_root not found ({self.seg_root!r}); run the LOcc OV-Seg step first.")
            return
        num_classes = self._write_vocab()
        scenes = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene in tqdm.tqdm(scenes.iter_slices(1), total=len(scenes), position=0, desc="LOcc project"):
            self.process_scene(scene, num_classes)
