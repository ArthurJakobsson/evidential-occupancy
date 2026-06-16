"""Transfer Occ3D semantic labels onto the high-fidelity evidential geometry.

Occ3D-nuScenes is in the same ego frame and bounds as our evidence volume but coarser
([200,200,16] @ 0.4 m vs [400,400,32] @ 0.2 m), so each Occ3D voxel maps to a 2x2x2 block
of evidential voxels (index ``floor(i/2)``). For each evidential voxel we take the Occ3D
class of the block it falls in; where our geometry says *occupied* but Occ3D says
free(17)/other(0), we assign the nearest non-free Occ3D class within a small radius (a 3D
Euclidean distance transform on the 0.4 m grid), else leave it free.

Occ3D taxonomy (scene_reconstruction/visualization/colormap.py): 0=other, 1..16=semantic,
17=free. Output is keyed by the LIDAR sample-data token to align with the evidence stage.
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
    """Nearest-class transfer of Occ3D semantics onto the evidential occupied voxels."""

    ds: NuscenesDataset
    extra_data_root: Union[Path, str]
    # Parent of ``nuscenes_occ3d/gts`` (what load_cvpr2023_occupancy expects). Defaults to extra_data_root.
    occ3d_root: Optional[Union[Path, str]] = None
    fill_radius_m: float = 1.2
    voxel_size_occ3d: float = 0.4
    evidence_name: str = "evidence"
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

    def evidence_path(self, scene_name: str, token: str) -> Path:
        return Path(self.extra_data_root) / self.evidence_name / scene_name / "LIDAR_TOP" / f"{token}.arrow"

    def _occ3d_file(self, scene_name: str, sample_token: str) -> Path:
        return self._occ3d_root / "nuscenes_occ3d" / "gts" / scene_name / sample_token / "labels.npz"

    @staticmethod
    def _upsample(x: torch.Tensor) -> torch.Tensor:
        """[..., 200, 200, 16] -> [..., 400, 400, 32] by nearest (2x along each spatial axis)."""
        return x.repeat_interleave(2, -3).repeat_interleave(2, -2).repeat_interleave(2, -1)

    def _nearest_nonfree(self, sem200: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For every Occ3D voxel: class + distance of the nearest non-free Occ3D voxel."""
        nonfree = (sem200 != OCC3D_OTHER) & (sem200 != OCC3D_FREE)
        if not nonfree.any():
            return np.full_like(sem200, OCC3D_FREE), np.full(sem200.shape, np.inf, dtype=np.float32)
        # distance_transform_edt: distance from each nonzero cell to nearest zero cell.
        # We want distance to nearest non-free, so the non-free cells are the "zero" targets.
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
        # keep only samples whose Occ3D ground truth exists (robust to partial coverage)
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
            ev_path = self.evidence_path(scene_name, token)
            if not ev_path.exists():
                continue  # evidence stage must run first
            out_path = self.save_path(scene_name, token)
            if self.missing_only and out_path.exists():
                continue
            ev = pl.read_ipc(ev_path, memory_map=False)
            occupied = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.occupied"])[0].bool()  # [400,400,32]

            sem200 = series_to_torch(sample["sample.occ_gt.semantics"])[0].to(torch.int64)  # [200,200,16]
            mask_cam = series_to_torch(sample["sample.occ_gt.mask_camera"])[0].to(torch.uint8)
            mask_lid = series_to_torch(sample["sample.occ_gt.mask_lidar"])[0].to(torch.uint8)

            nearest_np, dist_np = self._nearest_nonfree(sem200.numpy().astype(np.int64))
            nearest = torch.from_numpy(nearest_np).to(torch.int64)
            within = torch.from_numpy(dist_np <= self.fill_radius_m)

            sem400 = self._upsample(sem200)
            near400 = self._upsample(nearest)
            within400 = self._upsample(within)

            out = sem400.clone()
            need_fill = occupied & ((sem400 == OCC3D_OTHER) | (sem400 == OCC3D_FREE)) & within400
            out[need_fill] = near400[need_fill]

            out_df = sample.select("LIDAR_TOP.sample_data.token").with_columns(
                torch_to_series(f"LIDAR_TOP.{self.name}.semantics", out.to(torch.uint8)[None]),
                torch_to_series(f"LIDAR_TOP.{self.name}.mask_camera", self._upsample(mask_cam)[None]),
                torch_to_series(f"LIDAR_TOP.{self.name}.mask_lidar", self._upsample(mask_lid)[None]),
            )
            out_df.write_ipc(out_path, compression="zstd")

    def process_data(self) -> None:
        scenes = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene in tqdm.tqdm(scenes.iter_slices(1), total=len(scenes), position=0, desc="Occ3D transfer"):
            self.process_scene(scene)
