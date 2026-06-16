"""Export occupancy + per-voxel epistemic uncertainty from the accumulated evidence.

Reads the accumulated reflection/transmission volumes written by
``temporal-accumulation`` and, per LIDAR key-frame, derives the Dempster-Shafer belief
masses ``[m_o, m_f, m_omega]`` (occupied / free / ignorance) and the binary occupancy
``occupied = m_o > m_f``. ``m_omega`` is the per-voxel epistemic uncertainty label.
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
from scene_reconstruction.math.dempster_shafer import belief_from_reflection_and_transmission_stacked

RT_COL = "LIDAR_TOP.reflection_and_transmission_multi_frame"


@dataclass
class EvidenceExport:
    """Occupancy + belief masses [m_o, m_f, m_omega] from accumulated reflection/transmission."""

    ds: NuscenesDataset
    extra_data_root: Union[Path, str]
    p_fn: float = 0.8
    p_fp: float = 0.2
    device: str = "cuda"
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None
    input_name: str = "reflection_and_transmission_multi_frame"
    name: str = "evidence"

    def input_dir(self, scene_name: str) -> Path:
        """Directory of accumulated-evidence files for a scene."""
        return Path(self.extra_data_root) / self.input_name / scene_name / "LIDAR_TOP"

    def save_path(self, scene_name: str, token: str) -> Path:
        """Output path for one key-frame."""
        path = Path(self.extra_data_root) / self.name / scene_name / "LIDAR_TOP" / f"{token}.arrow"
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def process_scene(self, scene_name: str) -> None:
        """Derive occupancy + belief for every accumulated key-frame of a scene."""
        in_dir = self.input_dir(scene_name)
        if not in_dir.exists():
            return
        files = sorted(in_dir.glob("*.arrow"))
        for filename in tqdm.tqdm(files, desc=scene_name, position=1, leave=False):
            token = filename.stem
            out_path = self.save_path(scene_name, token)
            if self.missing_only and out_path.exists():
                continue
            df = pl.read_ipc(filename, memory_map=False)
            rt = series_to_torch(df[RT_COL]).to(self.device).float()  # [1, 2, 400, 400, 32]
            bba = belief_from_reflection_and_transmission_stacked(
                rt, p_fn=self.p_fn, p_fp=self.p_fp, with_omega=True
            )  # [1, 3, X, Y, Z] -> m_o, m_f, m_omega
            occupied = (bba[:, 0] > bba[:, 1]).to(torch.uint8).cpu()  # [1, X, Y, Z]
            belief = bba.float().cpu()  # [1, 3, X, Y, Z]
            out = df.select("LIDAR_TOP.sample_data.token").with_columns(
                torch_to_series(f"LIDAR_TOP.{self.name}.occupied", occupied),
                torch_to_series(f"LIDAR_TOP.{self.name}.belief", belief),
                df[f"{RT_COL}.volume.lower"].alias(f"LIDAR_TOP.{self.name}.volume.lower"),
                df[f"{RT_COL}.volume.upper"].alias(f"LIDAR_TOP.{self.name}.volume.upper"),
            )
            out.write_ipc(out_path, compression="zstd")

    def process_data(self) -> None:
        """Process the (sharded) set of scenes."""
        scenes = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene_name in tqdm.tqdm(scenes["scene.name"], position=0, desc="Evidence export"):
            self.process_scene(scene_name)
