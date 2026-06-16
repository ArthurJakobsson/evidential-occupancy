"""Validation commands for the generated label stages (run on mini).

Asserts shapes / value ranges / cross-stage invariants on a sample of key-frames, so each
stage can be checked in isolation before scaling up. Usage:

    python -m scene_reconstruction.cli.main check ./conf default labels
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
import typer

from scene_reconstruction.cli.config import make_cfg
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch

app = typer.Typer(name="check", callback=make_cfg, help="Validate generated labels.", no_args_is_help=True)

SHAPE = (400, 400, 32)


def _sample_files(files: list[Path], n: int = 5) -> list[Path]:
    if len(files) <= n:
        return files
    step = max(1, len(files) // n)
    return files[::step][:n]


@app.command(name="labels")
def labels(ctx: typer.Context, num_samples: int = 5) -> None:
    """Check the evidence + occ3d-transfer outputs for a few key-frames."""
    cfg = ctx.meta["cfg"]
    extra = Path(str(cfg.export.evidence_export.extra_data_root))

    ev_files = sorted((extra / "evidence").glob("*/LIDAR_TOP/*.arrow"))
    if not ev_files:
        raise SystemExit(f"No evidence files under {extra/'evidence'} — run `occupancy-export` first.")

    checked = 0
    for f in _sample_files(ev_files, num_samples):
        scene_name, token = f.parent.parent.name, f.stem
        ev = pl.read_ipc(f, memory_map=False)
        occupied = series_to_torch(ev["LIDAR_TOP.evidence.occupied"])[0]  # [400,400,32]
        belief = series_to_torch(ev["LIDAR_TOP.evidence.belief"])[0].float()  # [3,400,400,32]
        m_o, m_f, m_omega = belief[0], belief[1], belief[2]

        assert tuple(occupied.shape) == SHAPE, f"occupied shape {tuple(occupied.shape)}"
        assert tuple(belief.shape) == (3, *SHAPE), f"belief shape {tuple(belief.shape)}"
        assert belief.min() >= -1e-4 and belief.max() <= 1 + 1e-4, "belief out of [0,1]"
        assert (m_o + m_f + m_omega - 1.0).abs().max() < 1e-3, "belief masses do not sum to 1"
        assert torch.equal(occupied.bool(), (m_o > m_f)), "occupied != (m_o > m_f)"

        sem_info = "NA"
        ot = extra / "occ3d_transfer" / scene_name / "LIDAR_TOP" / f"{token}.arrow"
        if ot.exists():
            otd = pl.read_ipc(ot, memory_map=False)
            sem = series_to_torch(otd["LIDAR_TOP.occ3d_transfer.semantics"])[0]  # [400,400,32]
            fdist = series_to_torch(otd["LIDAR_TOP.occ3d_transfer.fill_distance"])[0].float()  # [400,400,32]
            assert tuple(sem.shape) == SHAPE and tuple(fdist.shape) == SHAPE, "occ3d shape mismatch"
            assert int(sem.min()) >= 0 and int(sem.max()) <= 17, "semantics out of 0..17"
            assert float(fdist.min()) >= 0.0, "negative fill_distance"
            # densified field: every 2x2x2 evidential block maps to one Occ3D voxel -> one class.
            blocks = sem.reshape(200, 2, 200, 2, 16, 2).permute(0, 2, 4, 1, 3, 5).reshape(200, 200, 16, 8)
            assert bool((blocks == blocks[..., :1]).all()), "2x2x2 blocks are not class-consistent"
            occ_mask = occupied.bool()
            labeled = ((sem != 0) & (sem != 17) & occ_mask).sum().item()
            frac = labeled / max(int(occ_mask.sum()), 1)
            fo = fdist[occ_mask]
            mean_fill = float(fo[fo.isfinite()].mean()) if bool(fo.isfinite().any()) else float("inf")
            sem_info = f"labeled-occ={frac:.0%} mean_fill={mean_fill:.2f}m"

        print(
            f"OK {scene_name}/{token[:8]} occ={int(occupied.bool().sum()):>6} "
            f"m_omega[min,max]=[{m_omega.min():.3f},{m_omega.max():.3f}] | occ3d: {sem_info}"
        )
        checked += 1

    print(f"\nchecked {checked} key-frames — all assertions passed")
