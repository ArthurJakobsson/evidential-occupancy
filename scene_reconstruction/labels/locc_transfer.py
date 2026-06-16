"""Transfer LOcc open-vocabulary labels + CLIP features onto the evidential geometry.

LOcc (ICCV 2025) runs in its own mmdet3d/BEVDet env (see LOCC_SETUP.md) and exports, per
key-frame, an Occ3D-grid ([200,200,16]) result to
``<extra>/locc_raw/<scene>/LIDAR_TOP/<token>.npz`` with:
  * ``semantics``    int16 [200,200,16]  -- LOcc open-vocab class id (``locc_free`` = empty)
  * ``feat_coords``  int16 [M,3]         -- Occ3D voxel indices that carry a CLIP feature
  * ``feat``         float16 [M,128]     -- L2-normalized CLIP features at those voxels

This stage maps that onto our 0.2 m grid: the discrete labels by 2x2x2 upsample (dense
arrow), and the CLIP field sampled at OUR occupied voxels and stored SPARSE (npz: coords +
feat) -- a dense 128-D field would be ~1.3 GB/frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import torch
import tqdm

from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch, torch_to_series


@dataclass
class LoccTransfer:
    """Map LOcc open-vocab labels (+ sparse CLIP) from the Occ3D grid onto the evidential grid."""

    ds: NuscenesDataset
    extra_data_root: Union[Path, str]
    locc_raw_name: str = "locc_raw"
    evidence_name: str = "evidence"
    name: str = "locc_transfer"
    clip_name: str = "locc_clip"
    locc_free: int = -1
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None

    def raw_dir(self, scene_name: str) -> Path:
        return Path(self.extra_data_root) / self.locc_raw_name / scene_name / "LIDAR_TOP"

    def evidence_path(self, scene_name: str, token: str) -> Path:
        return Path(self.extra_data_root) / self.evidence_name / scene_name / "LIDAR_TOP" / f"{token}.arrow"

    def save_path(self, scene_name: str, token: str) -> Path:
        path = Path(self.extra_data_root) / self.name / scene_name / "LIDAR_TOP" / f"{token}.arrow"
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def clip_path(self, scene_name: str, token: str) -> Path:
        path = Path(self.extra_data_root) / self.clip_name / scene_name / "LIDAR_TOP" / f"{token}.npz"
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    @staticmethod
    def _upsample(x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(2, -3).repeat_interleave(2, -2).repeat_interleave(2, -1)

    def process_scene(self, scene_name: str) -> None:
        raw_dir = self.raw_dir(scene_name)
        if not raw_dir.exists():
            return  # LOcc outputs must be exported first (see LOCC_SETUP.md)
        for filename in tqdm.tqdm(sorted(raw_dir.glob("*.npz")), desc=scene_name, position=1, leave=False):
            token = filename.stem
            out_path = self.save_path(scene_name, token)
            if self.missing_only and out_path.exists():
                continue
            raw = np.load(filename)
            sem200 = torch.from_numpy(raw["semantics"].astype(np.int16))  # [200,200,16]
            sem400 = self._upsample(sem200[None])  # [1,400,400,32]
            out_df = pl.DataFrame({"LIDAR_TOP.sample_data.token": [token]}).with_columns(
                torch_to_series(f"LIDAR_TOP.{self.name}.semantics", sem400),
            )
            out_df.write_ipc(out_path, compression="zstd")

            # sparse CLIP at our occupied voxels (needs the evidence + a feat field)
            ev_path = self.evidence_path(scene_name, token)
            if "feat" not in raw or "feat_coords" not in raw or not ev_path.exists():
                continue
            occupied = series_to_torch(
                pl.read_ipc(ev_path, memory_map=False)[f"LIDAR_TOP.{self.evidence_name}.occupied"]
            )[0].bool()  # [400,400,32]
            occ_idx = occupied.nonzero(as_tuple=False)  # [N,3] (0.2 m grid indices)
            if occ_idx.shape[0] == 0:
                continue
            # dense Occ3D-grid -> feature-row lookup (-1 = no feature)
            coords = torch.from_numpy(raw["feat_coords"].astype(np.int64))  # [M,3] occ3d indices
            row_grid = torch.full((200, 200, 16), -1, dtype=torch.long)
            row_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = torch.arange(coords.shape[0])
            occ200 = (occ_idx // 2).clamp(min=torch.tensor([0, 0, 0]), max=torch.tensor([199, 199, 15]))
            rows = row_grid[occ200[:, 0], occ200[:, 1], occ200[:, 2]]  # [N]
            valid = rows >= 0
            feat = torch.from_numpy(raw["feat"]).float()[rows[valid]].to(torch.float16)  # [Nv,128]
            np.savez_compressed(
                self.clip_path(scene_name, token),
                coords=occ_idx[valid].to(torch.int16).numpy(),
                feat=feat.numpy(),
                shape=np.array([400, 400, 32], np.int16),
            )

    def process_data(self) -> None:
        scenes = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene_name in tqdm.tqdm(scenes["scene.name"], position=0, desc="LOcc transfer"):
            self.process_scene(scene_name)
