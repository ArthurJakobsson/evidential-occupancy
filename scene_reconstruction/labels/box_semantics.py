"""Foreground semantics from 3D boxes, transferred onto the evidential geometry.

Reuses the per-voxel instance ids already produced by the ``scene-flow`` stage
(``LIDAR_TOP.scene_flow.scene_instance_index``: 0 = background, 1..K = the scene's instances)
and maps each instance to its Occ3D foreground class (1..10) via the annotation tables and
``taxonomy.nuscenes_name_to_occ3d``. This gives precise object-class + instance labels on the
high-fidelity grid for the 10 "thing" classes that boxes cover (stuff classes need lidarseg /
Occ3D / open-vocab).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import polars as pl
import torch
import tqdm

from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch, torch_to_series
from scene_reconstruction.data.nuscenes.scene_flow import SceneFlow
from scene_reconstruction.labels.taxonomy import nuscenes_name_to_occ3d

INSTANCE_COL = "LIDAR_TOP.scene_flow.scene_instance_index"


@dataclass
class BoxSemantics:
    """Per-voxel Occ3D foreground class + instance id from 3D boxes (via scene-flow ids)."""

    ds: NuscenesDataset
    extra_data_root: Union[Path, str]
    scene_flow_name: str = "scene_flow"
    name: str = "box_semantics"
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None

    def scene_flow_dir(self, scene_name: str) -> Path:
        return Path(self.extra_data_root) / self.scene_flow_name / scene_name / "LIDAR_TOP"

    def save_path(self, scene_name: str, token: str) -> Path:
        path = Path(self.extra_data_root) / self.name / scene_name / "LIDAR_TOP" / f"{token}.arrow"
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def _instance_to_class(self, scene_name: str) -> torch.Tensor:
        """Lookup tensor [K+1]: scene_instance.index -> Occ3D class (index 0 = background = 0)."""
        sf = SceneFlow(ds=self.ds, extra_data_root=self.extra_data_root)
        _, scene_with_annos, scene_instance_index = sf.load_scene(scene_name)
        num_instances = int(scene_instance_index["scene_instance.index"].max() or 0)
        lut = torch.zeros(num_instances + 1, dtype=torch.long)  # 0 = background
        idx_cat = (
            scene_with_annos.select("scene_instance.index", "category.token")
            .unique()
            .join(self.ds.category.select("category.token", "category.name"), on="category.token", how="left")
        )
        for row in idx_cat.iter_rows(named=True):
            lut[int(row["scene_instance.index"])] = nuscenes_name_to_occ3d(row.get("category.name"))
        return lut

    def process_scene(self, scene: pl.DataFrame) -> None:
        scene_name = scene["scene.name"].item()
        sf_dir = self.scene_flow_dir(scene_name)
        if not sf_dir.exists():
            return  # scene-flow stage must run first
        lut = self._instance_to_class(scene_name)
        max_idx = lut.shape[0] - 1
        for filename in tqdm.tqdm(sorted(sf_dir.glob("*.arrow")), desc=scene_name, position=1, leave=False):
            token = filename.stem
            out_path = self.save_path(scene_name, token)
            if self.missing_only and out_path.exists():
                continue
            sf = pl.read_ipc(filename, memory_map=False)
            instance = series_to_torch(sf[INSTANCE_COL])[0].to(torch.int64)  # [400,400,32]
            category = lut[instance.clamp(0, max_idx)]  # [400,400,32], 0..10
            out_df = sf.select("LIDAR_TOP.sample_data.token").with_columns(
                torch_to_series(f"LIDAR_TOP.{self.name}.category_index", category.to(torch.int16)[None]),
                torch_to_series(f"LIDAR_TOP.{self.name}.scene_instance_index", instance.to(torch.int32)[None]),
            )
            out_df.write_ipc(out_path, compression="zstd")

    def process_data(self) -> None:
        self.ds.load_sensor_sample_annotation("LIDAR_TOP")
        scenes = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene in tqdm.tqdm(scenes.iter_slices(1), total=len(scenes), position=0, desc="Box semantics"):
            self.process_scene(scene)
