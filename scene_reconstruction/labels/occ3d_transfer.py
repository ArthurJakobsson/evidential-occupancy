"""Transfer Occ3D semantic labels onto the high-fidelity evidential geometry.

Occ3D-nuScenes is in the same ego frame and bounds as our evidence volume but coarser
([200,200,16] @ 0.4 m vs [400,400,32] @ 0.2 m), so each Occ3D voxel maps to a 2x2x2 block
of evidential voxels (index ``floor(i/2)``).

We build a **densified nearest-class field**: every voxel is assigned the nearest non-free
Occ3D class (a 3D Euclidean distance transform on the 0.4 m grid), so that *any* occupied
voxel — regardless of the occupancy threshold used downstream — gets a semantic class.
Alongside it we store ``fill_distance`` = distance (m) to the nearest real Occ3D voxel:
0 means a direct Occ3D label, larger means our geometry has surface Occ3D doesn't (a signal
for the open-vocab / OOD tracks). Set ``max_fill_distance_m`` to fall back to
``unmatched_label`` beyond a distance instead of an arbitrary far class.

Occ3D taxonomy (scene_reconstruction/visualization/colormap.py): 0=other, 1..16=semantic,
17=free. Output is keyed by the LIDAR sample-data token to align with the other stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import torch
import tqdm
from scipy.ndimage import distance_transform_edt

from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch, torch_to_series

OCC3D_OTHER = 0
OCC3D_FREE = 17


@dataclass
class Occ3dTransfer:
    """Densified nearest-Occ3D-class transfer + fill-distance onto the evidential grid."""

    ds: NuscenesDataset
    extra_data_root: Union[Path, str]
    # Parent of ``nuscenes_occ3d/gts`` (what load_cvpr2023_occupancy expects). Defaults to extra_data_root.
    occ3d_root: Optional[Union[Path, str]] = None
    # None -> label every voxel with its nearest non-free class. A float caps the fill: voxels
    # whose nearest real Occ3D voxel is farther than this get ``unmatched_label`` instead.
    max_fill_distance_m: Optional[float] = None
    unmatched_label: int = OCC3D_OTHER
    voxel_size_occ3d: float = 0.4
    name: str = "occ3d_transfer"
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None

    @property
    def _occ3d_root(self) -> Path:
        return Path(self.occ3d_root) if self.occ3d_root is not None else Path(self.extra_data_root)

    def save_path(self, scene_name: str, token: str) -> Path:
        path = Path(self.extra_data_root) / self.name / scene_name / "LIDAR_TOP" / f"{token}.arrow"
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def _occ3d_file(self, scene_name: str, sample_token: str) -> Path:
        return self._occ3d_root / "nuscenes_occ3d" / "gts" / scene_name / sample_token / "labels.npz"

    @staticmethod
    def _upsample(x: torch.Tensor) -> torch.Tensor:
        """[..., 200, 200, 16] -> [..., 400, 400, 32] by nearest (2x along each spatial axis)."""
        return x.repeat_interleave(2, -3).repeat_interleave(2, -2).repeat_interleave(2, -1)

    def _nearest_class_and_distance(self, sem200: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per Occ3D voxel: nearest non-free class + distance (m) to that voxel (0 if itself non-free)."""
        nonfree = (sem200 != OCC3D_OTHER) & (sem200 != OCC3D_FREE)
        if not nonfree.any():
            return np.full_like(sem200, OCC3D_FREE), np.full(sem200.shape, np.inf, dtype=np.float32)
        # distance_transform_edt: distance from each nonzero cell to nearest zero cell + that cell's index.
        # The non-free cells are the "zero" targets, so every voxel gets its nearest non-free class.
        dist, idx = distance_transform_edt(
            ~nonfree, sampling=self.voxel_size_occ3d, return_distances=True, return_indices=True
        )
        nearest = sem200[idx[0], idx[1], idx[2]]
        return nearest, dist.astype(np.float32)

    def process_scene(self, scene: pl.DataFrame) -> None:
        scene = self.ds.join(scene, self.ds.sample)
        scene = self.ds.load_sample_data(scene, "LIDAR_TOP", with_data=False)
        scene = scene.filter(pl.col("LIDAR_TOP.sample_data.is_key_frame"))
        if len(scene) == 0:
            return
        keep = pl.Series(
            [self._occ3d_file(n, t).exists() for n, t in zip(scene["scene.name"], scene["sample.token"])]
        )
        scene = scene.filter(keep)
        if len(scene) == 0:
            return
        scene = self.ds.load_cvpr2023_occupancy(scene, root_path=self._occ3d_root)

        for sample in scene.iter_slices(1):
            scene_name = sample["scene.name"].item()
            token = sample["LIDAR_TOP.sample_data.token"].item()
            out_path = self.save_path(scene_name, token)
            if self.missing_only and out_path.exists():
                continue

            sem200 = series_to_torch(sample["sample.occ_gt.semantics"])[0].to(torch.int64).numpy().astype(np.int64)
            mask_cam = series_to_torch(sample["sample.occ_gt.mask_camera"])[0].to(torch.uint8)
            mask_lid = series_to_torch(sample["sample.occ_gt.mask_lidar"])[0].to(torch.uint8)

            nearest, dist = self._nearest_class_and_distance(sem200)  # [200,200,16]
            if self.max_fill_distance_m is not None:
                nearest = np.where(dist <= self.max_fill_distance_m, nearest, self.unmatched_label)

            sem400 = self._upsample(torch.from_numpy(nearest).to(torch.uint8)[None])  # [1,400,400,32]
            dist400 = self._upsample(torch.from_numpy(dist).to(torch.float32)[None])  # [1,400,400,32]

            out_df = sample.select("LIDAR_TOP.sample_data.token").with_columns(
                torch_to_series(f"LIDAR_TOP.{self.name}.semantics", sem400),
                torch_to_series(f"LIDAR_TOP.{self.name}.fill_distance", dist400),
                torch_to_series(f"LIDAR_TOP.{self.name}.mask_camera", self._upsample(mask_cam[None])),
                torch_to_series(f"LIDAR_TOP.{self.name}.mask_lidar", self._upsample(mask_lid[None])),
            )
            out_df.write_ipc(out_path, compression="zstd")

    def process_data(self) -> None:
        scenes = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene in tqdm.tqdm(scenes.iter_slices(1), total=len(scenes), position=0, desc="Occ3D transfer"):
            self.process_scene(scene)
