# Porting the multi-label occupancy pipeline to Waymo — scoping notes

Status: **analysis / not started.** This is a written scope to pick up later. No code has been
written for Waymo yet. The nuScenes pipeline (geometry + `occupancy-export` / `occ3d-transfer` /
`box-semantics` / `lidarseg-transfer` / `locc-transfer` + viewer + validation) is the reference;
see the build plan and `LOCC_SETUP.md`.

Waymo has the **same underlying information** as nuScenes (LiDAR, cameras, 3D boxes + tracking,
ego/sensor poses, calibrations) — it is just **packaged differently** (TFRecord protobuf, different
sensor rig, different taxonomies). So the port is bounded: most of the system is dataset-agnostic.

## Data already on the drive (`/media/adteam/ARTHUR/clean/`)
- **Raw Waymo v1.4.0 perception TFRecords** — `Datasets/Waymo/1.4.0/training/segment-*_with_camera_labels.tfrecord`
  (LiDAR range images, 5 cameras, 3D boxes + tracking ids, per-frame vehicle poses, calibrations).
  This is the source for the evidence pipeline. **Points live only in the TFRecords** — there are no
  pre-extracted `.bin` point clouds anywhere on the drive.
- **Occ3D-Waymo** — `Occ3D/Waymo/`:
  - `voxel04/voxel04/{training,validation}/<scene>/<frame>_04.npz` — **200×200×16 @ 0.4 m** (same
    shape/voxel-size as Occ3D-nuScenes). Keys: `voxel_label` (u8, ~23-class scheme — values observed
    up to 23, **different taxonomy from nuScenes**), `origin_voxel_state`/`final_voxel_state` (u8
    occupancy/visibility masks), `infov` (bool in-FOV), `ego2global` (4×4). Also `voxel01` (0.1 m).
  - `waymo_infos_{train,val}.pkl`, `cam_infos.pkl` / `cam_infos_vali.pkl` — BEVDet/mmdet3d-style
    per-frame metadata (poses, calib, boxes, camera/lidar paths). Likely reusable to avoid most
    TFRecord parsing — **schema to be confirmed**.

## The one idea that makes this tractable
The pipeline is **decoupled from the dataset behind a single interface** — `NuscenesDataset` (in
`scene_reconstruction/data/nuscenes/dataset.py`) exposes scene/sample iteration and produces polars
columns: per-frame LiDAR points (sensor/ego frame), `ego_from_global` / `sensor_from_ego` transforms,
boxes + instance/tracking ids, and camera paths. **Every stage either reads only the produced
`.arrow`/`.npz` files, or calls `ds.*`.** So porting ≈ implement a `WaymoDataset` with the same
interface + swap taxonomy/colormap/configs. Most stages then run unchanged.

That `WaymoDataset` is the linchpin. Almost everything else is "easy" or mechanical.

## Easy transitions (work as-is once `WaymoDataset` + grid/taxonomy exist)
- **`occupancy-export`** (`occupancy/evidence_export.py`) — pure Dempster-Shafer math on the
  accumulated RT tensor; zero dataset coupling. Unchanged.
- **Core math/utilities** — belief functions (`math/dempster_shafer.py`), `core/volume.py`,
  point→voxel scatter (`occupancy/grid.py`), transforms. Agnostic.
- **Multi-GPU shard runner, CLI/config layout, pixi tasks** — they iterate "scenes"; just point them
  at Waymo configs + a `WaymoDataset`.
- **`locc-transfer` stage + the pkl→raw converter** — the 2×2×2 upsample + sparse-CLIP logic is
  grid-agnostic.
- **Viewer core + check/validate harness** — render by occupancy/class/uncertainty + assert
  shapes/ranges; only need a Waymo colormap (and a Waymo camera map for the image panel).
- **Grid alignment looks free**: Occ3D-Waymo is **200×200×16 @ 0.4 m** — same as Occ3D-nuScenes — so
  a 0.2 m Waymo geometry on the same bounds keeps the 2×2×2 transfer (one origin/range check needed).

## Medium (same logic, needs Waymo specifics)
- **`occ3d-transfer`** — logic (upsample + nearest-fill) is grid-agnostic. Needs (a) a Waymo Occ3D
  loader for the different layout/keys (`voxel04/<scene>/<frame>_04.npz`, `voxel_label` +
  `final_voxel_state`/`infov` masks, vs nuScenes `labels.npz` `semantics`/`mask_*`), (b) a confirmed
  grid origin/range, (c) a Waymo taxonomy/colormap (~23-class, different free index).
- **`box-semantics`** — reuses the scene-flow instance ids; just a Waymo taxonomy mapping the **4 box
  classes** (Vehicle / Pedestrian / Cyclist / Sign) → the Waymo voxel taxonomy (simpler than
  nuScenes' 10, but new). Waymo boxes carry tracking ids → instance motion works the same.
- **`lidarseg-transfer`** — point→voxel majority-vote is agnostic; needs a Waymo 3D-semseg loader
  (range-image → per-point, 23-class) and must tolerate **sparse coverage** (Waymo labels only a
  subset of frames) — the existing "gate/skip when missing" logic handles that.

## Hard / the big rocks (with how to handle)
1. **`WaymoDataset` over TFRecords (the linchpin).** Waymo is TFRecord protobuf → needs
   `waymo-open-dataset` (TensorFlow), which won't coexist cleanly with the pixi torch-1.12 / CUDA-11.6
   env. **Handle it like Occ3D/LOcc: a one-time preprocessing step in a separate conda env** that
   extracts per-frame `points.bin` from the TFRecords and reuses `waymo_infos_*.pkl` for metadata
   (poses/calib/boxes/cam-paths), writing the repo's `.arrow`/`.bin` format. Then the pixi pipeline
   reads preprocessed files with **no TF at runtime**. The infos pkls likely supply most metadata, so
   the real new work is point extraction.
2. **Spherical LiDAR model retune (single vs multi-LiDAR).** The evidence step's spherical
   reflection/transmission grid is tuned to nuScenes' 32-beam top LiDAR. Waymo TOP is **64-beam,
   vertical FOV ≈ −17.6…+2.4°, 360° azimuth**, plus 4 short-range side LiDARs. **Handle:** start
   **TOP-LiDAR-only**, re-tune `spherical_lower/upper` (elevation) / `spherical_shape` in config (no
   code), validate via the depth-render-vs-LiDAR MAE that binning is sane. Multi-LiDAR fusion breaks
   the single-origin spherical assumption → defer (or add side LiDARs later via cartesian
   accumulation, skipping the spherical step for them).
3. **Taxonomy unification.** Occ3D-Waymo (~23-class), Waymo boxes (4), Waymo semseg (23) all differ
   from nuScenes (17 / 10 / 16). **Handle:** a `waymo_taxonomy.py` (parallel to `labels/taxonomy.py`)
   + a Waymo colormap defining the class list, **free/"GO" index** (confirm which `voxel_label` value
   is free — values went up to 23), and the box/semseg→voxel maps. This is the notes' "unified
   cross-dataset taxonomy."
4. **Camera coverage for LOcc.** Waymo has **5 cameras (~252°, no rear)** vs nuScenes 6 (360°), and
   LOcc's scripts are nuScenes-devkit-coded. **Handle:** adapt LOcc's I/O to Waymo (project to 5 cams
   via `cam_infos.pkl`; reuse SAN + `--fixed_vocab` unchanged) and accept that **open-vocab labels
   only cover the camera-visible front/sides** — the rear stays covered by Occ3D/lidarseg (closed-set)
   only. Inherent Waymo limitation, not a bug; document it.
5. **Convention/format papercuts** — Waymo box `heading`/size order, range-image → cartesian,
   vehicle-frame axes (x fwd, y left, z up), no-label-zones. Low conceptual risk but easy to get a
   sign/axis wrong; isolate in the preprocessor and sanity-check by rendering boxes + depth MAE early.

## Cross-cutting
- **Data volume** is larger than nuScenes (~798 train + 202 val segments, ~200k frames) → more
  compute, but the shard runner scales. Same fast-local-NVMe requirement as the nuScenes full run.
- **TF dependency isolation** — keep `waymo-open-dataset` in its own env; never import it at pipeline
  runtime.

## Suggested order
1. Preprocess Waymo (separate env): extract per-frame points + reuse `waymo_infos_*.pkl` metadata →
   repo `.arrow`/`.bin` format. *(the big lift)*
2. `WaymoDataset` implementing the `NuscenesDataset` interface over the preprocessed files.
3. Re-tune the spherical config (TOP LiDAR) + run geometry on a few segments; validate depth-MAE.
4. `occupancy-export` (free) + `occ3d-transfer` (Waymo loader + taxonomy + colormap) + `box-semantics`
   (Waymo taxonomy). Test on a few segments.
5. `lidarseg-transfer` (Waymo semseg loader, sparse coverage).
6. LOcc-Waymo: adapt I/O for 5 cameras + Waymo projections; SAN + `--fixed_vocab` reused.
7. Viewer Waymo colormap + camera map; validation (depth MAE / RayIoU / calibration).

## Things to verify first (cheap, scopes the rest)
- Occ3D-Waymo **origin/range** — confirm `[-40,40]×[-40,40]×[-1,5.4]` like nuScenes so the 2×2×2
  alignment holds (the 200³ @ 0.4 m shape already matches).
- Occ3D-Waymo **class scheme + free/GO index** (`voxel_label` values include 23 — map to a Waymo
  taxonomy; identify the free label).
- **`waymo_infos_*.pkl` schema** — what it contains (poses/calib/boxes/paths) decides how little
  preprocessing is needed vs. how much must come from TFRecords.
- **`waymo-open-dataset` install** — TF version vs the box (CPU extraction is fine for a one-time pass).

## Bottom line
~70% of the system is easy/mechanical (decoupled stages, configs, math). The real engineering is the
**TFRecord preprocessing + `WaymoDataset`**, a **spherical-model retune**, and **taxonomy / LOcc-I/O
adaptation** — all bounded, and the key data (raw TFRecords + Occ3D-Waymo) is already on the drive.
