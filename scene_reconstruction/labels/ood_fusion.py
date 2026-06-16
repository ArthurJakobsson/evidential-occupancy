"""OOD annotation layer for the evidential occupancy dataset.

A per-key-frame OOD score + source bitmask, kept SEPARATE from the clean semantic labels and
gated to OUR occupied voxels. It fuses the independent signals that are available:

  * geometry novelty -- our occupied surface that is far from any Occ3D class
    (``fill_distance`` from the occ3d-transfer stage). This is the discriminative signal:
    ~0 on Occ3D surfaces, high where our lidar geometry has structure Occ3D lacks.
  * synthetic injection (OccOoD-style) -- deterministically placed anomaly blobs that give
    known-positive OOD ground truth for the benchmark. OFF by default.
  * EchoOOD (ProOOD) -- an optional per-voxel score grid under ``<extra>/echo_ood/`` from
    running ProOOD on a trained model (gated; absent until that is run).

Epistemic uncertainty ``m_omega`` (lidar-evidence ignorance) is HIGH almost everywhere, so it
is NOT fused into the score (it would flood it); instead it is recorded in ``source`` (bit0)
and remains recoverable from the evidence stage. ``ood_source`` bits: 1=uncertain,
2=synthetic, 4=echo, 8=novelty. ``ood_score`` = max(novelty, synthetic, echo), in [0,1].
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
SRC_NOVELTY = 8


@dataclass
class OodFusion:
    """Fuse geometry-novelty (+ optional EchoOOD / synthetic) into an OOD layer; record uncertainty."""

    ds: NuscenesDataset
    extra_data_root: Union[Path, str]
    evidence_name: str = "evidence"
    occ3d_transfer_name: str = "occ3d_transfer"
    echo_ood_name: str = "echo_ood"
    name: str = "ood"
    uncertainty_threshold: float = 0.6
    novelty_scale: float = 2.0  # metres; fill_distance / novelty_scale, clamped to [0, 1]
    novelty_threshold: float = 0.5
    num_synthetic_anomalies: int = 0
    anomaly_radius_vox: int = 3
    seed_base: int = 0
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None

    def evidence_dir(self, scene_name: str) -> Path:
        return Path(self.extra_data_root) / self.evidence_name / scene_name / "LIDAR_TOP"

    def _stage_path(self, stage: str, scene_name: str, token: str) -> Path:
        return Path(self.extra_data_root) / stage / scene_name / "LIDAR_TOP" / f"{token}.arrow"

    def save_path(self, scene_name: str, token: str) -> Path:
        path = self._stage_path(self.name, scene_name, token)
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def _inject_synthetic(self, occupied: torch.Tensor, token: str) -> torch.Tensor:
        """Deterministically place anomaly cubes at occupied voxels; returns a bool mask [X,Y,Z]."""
        mask = torch.zeros_like(occupied, dtype=torch.bool)
        occ_idx = occupied.nonzero(as_tuple=False)
        if occ_idx.shape[0] == 0:
            return mask
        rng = np.random.RandomState((hash(token) ^ (self.seed_base * 2654435761)) & 0xFFFFFFFF)
        r = self.anomaly_radius_vox
        for c in rng.randint(0, occ_idx.shape[0], size=self.num_synthetic_anomalies):
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
            occupied = series_to_torch(ev[f"LIDAR_TOP.{self.evidence_name}.occupied"])[0].bool()
            occ_f = occupied.float()
            m_omega = belief[2].clamp(0.0, 1.0)

            score = torch.zeros_like(m_omega)
            source = (occupied & (m_omega > self.uncertainty_threshold)).to(torch.uint8) * SRC_UNCERTAIN

            ot_path = self._stage_path(self.occ3d_transfer_name, scene_name, token)
            if ot_path.exists():
                fdist = series_to_torch(pl.read_ipc(ot_path, memory_map=False)[
                    f"LIDAR_TOP.{self.occ3d_transfer_name}.fill_distance"])[0].float()
                novelty = (fdist / self.novelty_scale).clamp(0.0, 1.0) * occ_f
                score = torch.maximum(score, novelty)
                source = source | (novelty > self.novelty_threshold).to(torch.uint8) * SRC_NOVELTY

            echo_path = self._stage_path(self.echo_ood_name, scene_name, token)
            if echo_path.exists():
                echo = series_to_torch(pl.read_ipc(echo_path, memory_map=False)[
                    "LIDAR_TOP.echo_ood.score"])[0].float() * occ_f
                score = torch.maximum(score, echo)
                source = source | (echo > self.uncertainty_threshold).to(torch.uint8) * SRC_ECHO

            is_syn = torch.zeros_like(occupied, dtype=torch.uint8)
            if self.num_synthetic_anomalies > 0:
                syn = self._inject_synthetic(occupied, token)
                is_syn = syn.to(torch.uint8)
                score = torch.where(syn, torch.ones_like(score), score)
                source = source | (syn.to(torch.uint8) * SRC_SYNTHETIC)

            score = (score * occ_f).clamp(0.0, 1.0)
            out_df = ev.select("LIDAR_TOP.sample_data.token").with_columns(
                torch_to_series(f"LIDAR_TOP.{self.name}.score", score[None]),
                torch_to_series(f"LIDAR_TOP.{self.name}.source", source[None]),
                torch_to_series(f"LIDAR_TOP.{self.name}.is_synthetic", is_syn[None]),
            )
            out_df.write_ipc(out_path, compression="zstd")

    def process_data(self) -> None:
        scenes = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene_name in tqdm.tqdm(scenes["scene.name"], position=0, desc="OOD fusion"):
            self.process_scene(scene_name)
