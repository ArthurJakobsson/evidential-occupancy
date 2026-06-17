"""Open-vocabulary labels by projecting OUR occupancy into LOcc's OV-Seg images.

The ``locc-transfer`` stage consumes LOcc's *own* occupancy (LOcc voxelizes lidar points and
labels only the voxels a lidar return landed in), so any evidential voxel LOcc's lidar pass
missed — including low-confidence ones — ends up unlabeled.

This stage decouples occupancy from labeling, exactly like LOcc's
``PseudoOccGeneration-VoxelProjection.py`` but driven by *our* geometry: for every voxel our
evidence marks ``occupied``, it projects the voxel CENTER (ego frame) into the SAN open-vocab
segmentation of each camera and reads the class at that pixel. Occupancy comes entirely from
the evidence stage, so every occupied voxel — regardless of certainty — gets a class wherever
a camera sees it.

Inputs (per LIDAR key-frame):
  * our evidence occupancy ``<extra>/evidence/<scene>/LIDAR_TOP/<token>.arrow`` (``occupied``
    [1,400,400,32] + ``volume.lower``/``upper``), and
  * SAN OV-Seg PNGs at ``<seg_root>/<camera filename>.png`` (uint8 class id, 255 = none),
    the step-2 output of the LOcc pipeline (see LOCC_SETUP.md).

Output ``<extra>/locc_project/<scene>/LIDAR_TOP/<token>.arrow`` with
``LIDAR_TOP.locc_project.semantics`` int16 [1,400,400,32] (``-1`` = unlabeled), on our 0.2 m
grid — labelled natively at full resolution, no 2x2x2 upsample. ``vocab.json`` (class id ->
name) is written alongside so the viewer can name classes.

CAVEAT (intentional, per design): voxel-center projection has no occlusion test, so a voxel
occluded from a camera inherits whatever surface is in front of it. Worst for exactly the
low-certainty / occluded voxels this stage fills. A depth gate (keep the label only where the
voxel's projected depth matches the first surface) is the natural fix if results need it.
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


@dataclass
class LoccProject:
    """Label our occupied voxels by projecting their centers into the SAN OV-Seg images."""

    ds: NuscenesDataset
    extra_data_root: Union[Path, str]
    # SAN OV-Seg output dir; segmentation PNG for image ``f`` lives at ``seg_root/<f with .png>``.
    seg_root: Union[Path, str] = ""
    # LOcc vocab.txt (one class name per line); copied to ``locc_project/vocab.json`` and used to
    # bound class ids. Empty -> only the background value is treated as unlabeled.
    vocab_file: str = ""
    cameras: Sequence[str] = field(default_factory=lambda: CAMERAS)
    image_width: int = 1600
    image_height: int = 900
    seg_background: int = 255  # SAN "no class" value -> unlabeled
    locc_free: int = -1
    evidence_name: str = "evidence"
    name: str = "locc_project"
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None

    def evidence_path(self, scene_name: str, token: str) -> Path:
        return Path(self.extra_data_root) / self.evidence_name / scene_name / "LIDAR_TOP" / f"{token}.arrow"

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

    def _project(self, centers: np.ndarray, proj: np.ndarray, w: int, h: int):
        """centers [N,3] (ego) -> (uv [N,2], in-image mask [N]) under proj = K @ ego_from? (4x4)."""
        homog = np.concatenate([centers, np.ones((centers.shape[0], 1), centers.dtype)], axis=1)
        cam = homog @ proj.T  # [N,4]; rows of proj already (image_from_sensor @ sensor_from_ego)
        depth = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = cam[:, :2] / depth[:, None]
        valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        return uv, valid

    def _label_voxels(self, centers: np.ndarray, cam_proj_seg: list[tuple[np.ndarray, np.ndarray]],
                      num_classes: Optional[int]) -> np.ndarray:
        """Per voxel: the OV-Seg class from the camera whose view-center it lands nearest.

        Only labelled (non-background, in-vocab) hits are considered, so a voxel that projects
        onto sky in one camera but onto a real object in another keeps the object label.
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
            u = uv[idx, 0]
            v = uv[idx, 1]
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

    def process_scene(self, scene: pl.DataFrame, num_classes: Optional[int]) -> None:
        scene = self.ds.join(scene, self.ds.sample)
        scene = self.ds.load_sample_data(scene, "LIDAR_TOP", with_data=False)
        scene = scene.filter(pl.col("LIDAR_TOP.sample_data.is_key_frame"))
        if len(scene) == 0:
            return
        for cam in self.cameras:
            scene = self.ds.load_sample_data(scene, cam, with_data=False)

        scene_desc = scene["scene.name"][0]
        for sample in tqdm.tqdm(
            scene.iter_slices(1), total=len(scene), desc=scene_desc, position=1, leave=False
        ):
            scene_name = sample["scene.name"].item()
            token = sample["LIDAR_TOP.sample_data.token"].item()
            out_path = self.save_path(scene_name, token)
            if self.missing_only and out_path.exists():
                continue
            ev_path = self.evidence_path(scene_name, token)
            if not ev_path.exists():
                continue

            # gather cameras with an available segmentation for this frame
            cam_proj_seg: list[tuple[np.ndarray, np.ndarray]] = []
            for cam in self.cameras:
                seg = self._load_seg(self._seg_path(sample[f"{cam}.sample_data.filename"].item()))
                if seg is None:
                    continue
                image_from_sensor = series_to_torch(sample[f"{cam}.transform.image_from_sensor"])[0].numpy()
                sensor_from_ego = series_to_torch(sample[f"{cam}.transform.sensor_from_ego"])[0].numpy()
                cam_proj_seg.append((image_from_sensor @ sensor_from_ego, seg))
            if not cam_proj_seg:
                continue

            ev = pl.read_ipc(ev_path, memory_map=False)
            occupied = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.occupied"])[0].bool().numpy()
            lower = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.volume.lower"])[0].numpy()
            upper = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.volume.upper"])[0].numpy()
            occ_idx = np.argwhere(occupied)  # [N,3] grid indices
            shape = np.array(occupied.shape, dtype=np.float64)

            sem = np.full((1,) + occupied.shape, self.locc_free, dtype=np.int16)
            if occ_idx.shape[0] > 0:
                voxel_size = (upper - lower) / shape
                centers = lower + (occ_idx.astype(np.float64) + 0.5) * voxel_size  # ego frame [N,3]
                classes = self._label_voxels(centers, cam_proj_seg, num_classes)
                sem[0, occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = classes

            out_df = pl.DataFrame({"LIDAR_TOP.sample_data.token": [token]}).with_columns(
                torch_to_series(f"LIDAR_TOP.{self.name}.semantics", torch.from_numpy(sem)),
            )
            out_df.write_ipc(out_path, compression="zstd")

    def process_data(self) -> None:
        if not self.seg_root or not Path(self.seg_root).exists():
            print(f"[locc-project] seg_root not found ({self.seg_root!r}); run the LOcc OV-Seg step first.")
            return
        num_classes = self._write_vocab()
        scenes = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene in tqdm.tqdm(scenes.iter_slices(1), total=len(scenes), position=0, desc="LOcc project"):
            self.process_scene(scene, num_classes)
