"""OOD annotation layer for the evidential occupancy dataset.

Produces a per-key-frame OOD score + source bitmask, kept SEPARATE from the clean semantic
labels. The score fuses up to three signals (whichever are available):

  * epistemic uncertainty (the Dempster-Shafer ignorance mass ``m_omega`` from the evidence
    stage) -- a training-free OOD signal: low evidence -> more anomalous;
  * EchoOOD (ProOOD): an optional per-voxel score grid exported under ``<extra>/echo_ood/``
    by running ProOOD on a trained occupancy model (gated -- absent until that is run);
  * synthetic anomaly injection (OccOoD-style): deterministically placed anomaly blobs that
    give known-positive OOD ground truth for the benchmark. OFF by default
    (``num_synthetic_anomalies=0``) -- it is benchmark construction, not a clean label.

``ood_source`` is a bitmask: bit0 = high uncertainty (m_omega>tau), bit1 = synthetic,
bit2 = EchoOOD. ``is_synthetic`` marks the injected anomaly voxels.
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

SRC_UNCERTAIN = 1
SRC_SYNTHETIC = 2
SRC_ECHO = 4


@dataclass
class OodFusion:
    """Fuse epistemic uncertainty (+ optional EchoOOD / synthetic injection) into an OOD layer."""

    ds: NuscenesDataset
    extra_data_root: Union[Path, str]
    evidence_name: str = "evidence"
    echo_ood_name: str = "echo_ood"
    name: str = "ood"
    uncertainty_threshold: float = 0.6
    num_synthetic_anomalies: int = 0
    anomaly_radius_vox: int = 3
    seed_base: int = 0
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None

    def evidence_dir(self, scene_name: str) -> Path:
        return Path(self.extra_data_root) / self.evidence_name / scene_name / "LIDAR_TOP"

    def echo_path(self, scene_name: str, token: str) -> Path:
        return Path(self.extra_data_root) / self.echo_ood_name / scene_name / "LIDAR_TOP" / f"{token}.arrow"

    def save_path(self, scene_name: str, token: str) -> Path:
        path = Path(self.extra_data_root) / self.name / scene_name / "LIDAR_TOP" / f"{token}.arrow"
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def _inject_synthetic(self, occupied: torch.Tensor, token: str) -> torch.Tensor:
        """Deterministically place anomaly cubes at occupied voxels; returns a bool mask [X,Y,Z]."""
        mask = torch.zeros_like(occupied, dtype=torch.bool)
        occ_idx = occupied.nonzero(as_tuple=False)  # [M, 3]
        if occ_idx.shape[0] == 0:
            return mask
        rng = np.random.RandomState((hash(token) ^ (self.seed_base * 2654435761)) & 0xFFFFFFFF)
        choices = rng.randint(0, occ_idx.shape[0], size=self.num_synthetic_anomalies)
        r = self.anomaly_radius_vox
        X, Y, Z = occupied.shape
        for c in choices:
            cx, cy, cz = (int(v) for v in occ_idx[c])
            mask[max(cx - r, 0):cx + r + 1, max(cy - r, 0):cy + r + 1, max(cz - r, 0):cz + r + 1] = True
        return mask

    def process_scene(self, scene_name: str) -> None:
        ev_dir = self.evidence_dir(scene_name)
        if not ev_dir.exists():
            return  # evidence stage must run first
        for filename in tqdm.tqdm(sorted(ev_dir.glob("*.arrow")), desc=scene_name, position=1, leave=False):
            token = filename.stem
            out_path = self.save_path(scene_name, token)
            if self.missing_only and out_path.exists():
                continue
            ev = pl.read_ipc(filename, memory_map=False)
            belief = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.belief"])[0].float()  # [3, X, Y, Z]
            occupied = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.occupied"])[0].bool()  # [X, Y, Z]
            m_omega = belief[2].clamp(0.0, 1.0)

            score = m_omega.clone()  # [X, Y, Z] in [0, 1]
            source = torch.where(m_omega > self.uncertainty_threshold, SRC_UNCERTAIN, 0).to(torch.uint8)

            echo_path = self.echo_path(scene_name, token)
            if echo_path.exists():
                echo = series_to_torch(pl.read_ipc(echo_path, memory_map=False)["LIDAR_TOP.echo_ood.score"])[0].float()
                score = torch.maximum(score, echo)
                source = source | torch.where(echo > self.uncertainty_threshold, SRC_ECHO, 0).to(torch.uint8)

            is_syn = torch.zeros_like(occupied, dtype=torch.uint8)
            if self.num_synthetic_anomalies > 0:
                syn_mask = self._inject_synthetic(occupied, token)
                is_syn = syn_mask.to(torch.uint8)
                score = torch.where(syn_mask, torch.ones_like(score), score)
                source = source | (syn_mask.to(torch.uint8) * SRC_SYNTHETIC)

            out_df = ev.select("LIDAR_TOP.sample_data.token").with_columns(
                torch_to_series(f"LIDAR_TOP.{self.name}.score", score.clamp(0.0, 1.0)[None]),
                torch_to_series(f"LIDAR_TOP.{self.name}.source", source[None]),
                torch_to_series(f"LIDAR_TOP.{self.name}.is_synthetic", is_syn[None]),
            )
            out_df.write_ipc(out_path, compression="zstd")

    def process_data(self) -> None:
        scenes = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene_name in tqdm.tqdm(scenes["scene.name"], position=0, desc="OOD fusion"):
            self.process_scene(scene_name)
