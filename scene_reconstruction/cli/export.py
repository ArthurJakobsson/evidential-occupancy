"""Commands for data export."""


import typer
from hydra.utils import instantiate

from scene_reconstruction.cli.config import make_cfg
from scene_reconstruction.data.nuscenes.scene_flow import SceneFlow
from scene_reconstruction.labels.box_semantics import BoxSemantics
from scene_reconstruction.labels.occ3d_transfer import Occ3dTransfer
from scene_reconstruction.labels.ood_fusion import OodFusion
from scene_reconstruction.occupancy.evidence_export import EvidenceExport
from scene_reconstruction.occupancy.temporal_transmission_and_reflection import TemporalTransmissionAndReflection
from scene_reconstruction.occupancy.transmission_reflection import ReflectionTransmissionSpherical

app = typer.Typer(name="export", callback=make_cfg, help="Various export commands.", no_args_is_help=True)

SAVE_DIR = typer.Option(help="Directory to save data to.", dir_okay=True)
BATCH_SIZE = typer.Option(help="Batch size for data processing.")


@app.command(name="transmissions-reflections")
def transmissions_reflections(ctx: typer.Context) -> None:
    """Export sensor count maps to specified path."""
    cfg = ctx.meta["cfg"]

    transmission_and_reflections: ReflectionTransmissionSpherical = instantiate(cfg.export.transmissions_reflections)
    transmission_and_reflections.process_data()


REF_KEYFRAME_ONLY = typer.Option(help="Only accumulate for reference keyframes.")


@app.command(name="temporal-accumulation")
def temporal_transmissions_reflections(
    ctx: typer.Context,
) -> None:
    """Accumulate sensor count maps over time to specified path."""
    cfg = ctx.meta["cfg"]

    temporal_accumulation: TemporalTransmissionAndReflection = instantiate(cfg.export.temporal_accumulation)
    temporal_accumulation.process_data()


@app.command(name="scene-flow")
def scene_flow(ctx: typer.Context) -> None:
    """Accumulate sensor count maps over time to specified path."""
    cfg = ctx.meta["cfg"]

    scene_flow: SceneFlow = instantiate(cfg.export.scene_flow)
    scene_flow.process_data()


@app.command(name="occupancy-export")
def occupancy_export(ctx: typer.Context) -> None:
    """Derive occupancy + per-voxel epistemic uncertainty (Dempster-Shafer belief) from accumulated evidence."""
    cfg = ctx.meta["cfg"]

    evidence_export: EvidenceExport = instantiate(cfg.export.evidence_export)
    evidence_export.process_data()


@app.command(name="occ3d-transfer")
def occ3d_transfer(ctx: typer.Context) -> None:
    """Transfer Occ3D semantic class labels onto the evidential geometry (nearest-class)."""
    cfg = ctx.meta["cfg"]

    occ3d_transfer: Occ3dTransfer = instantiate(cfg.export.occ3d_transfer)
    occ3d_transfer.process_data()


@app.command(name="box-semantics")
def box_semantics(ctx: typer.Context) -> None:
    """Foreground class + instance labels from 3D boxes (via scene-flow instance ids)."""
    cfg = ctx.meta["cfg"]

    box_semantics: BoxSemantics = instantiate(cfg.export.box_semantics)
    box_semantics.process_data()


@app.command(name="ood")
def ood(ctx: typer.Context) -> None:
    """OOD layer: epistemic uncertainty (+ optional EchoOOD / synthetic injection)."""
    cfg = ctx.meta["cfg"]

    ood_fusion: OodFusion = instantiate(cfg.export.ood)
    ood_fusion.process_data()


@app.command(name="sensor-belief-maps", no_args_is_help=True)
def sensor_belief_maps(ctx: typer.Context) -> None:
    """Export sensor belief maps to specified path."""
