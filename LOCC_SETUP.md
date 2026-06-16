# Open-vocabulary labels via LOcc

LOcc (ICCV 2025, https://github.com/pkqbajng/LOcc) generates open-vocabulary voxel labels +
a CLIP feature field for nuScenes. It runs in its **own** environment (mmdet3d / BEVDet on a
modern CUDA) — separate from this repo's pixi/CUDA-11.6 env — and we only consume its output
here, via the `locc-transfer` stage.

## 1. Set up LOcc (its own env, once)
```bash
git clone https://github.com/pkqbajng/LOcc.git ~/LOcc && cd ~/LOcc
# follow its README: create the conda env, install mmcv/mmdet3d/BEVDet, download the
# pretrained weights + the LVLM/open-vocab segmentation models it uses.
```
LOcc needs the nuScenes images + calibration (already on the SSD) — point its config at
`$NUSCENES_ROOT`.

## 2. Run LOcc + export to the format this repo expects
Run LOcc's label-generation pipeline, then dump, per LIDAR key-frame, an Occ3D-grid
([200,200,16]) `.npz` to:
```
$NUSCENES_EXTRA_ROOT/locc_raw/<scene>/LIDAR_TOP/<lidar_sample_data_token>.npz
```
with keys:
| key | dtype / shape | meaning |
|-----|---------------|---------|
| `semantics`   | int16  [200,200,16] | LOcc open-vocab class id per voxel (use `-1` for empty) |
| `feat_coords` | int16  [M,3]        | Occ3D voxel indices that carry a CLIP feature |
| `feat`        | float16 [M,128]     | L2-normalized CLIP features at those voxels |

Also save the label→text vocabulary once at `locc_raw/vocab.json` (your reference; this repo
keeps the integer ids). Key files by the **LIDAR sample_data token** (the `.arrow` stem used
by every other stage) so they align — map from the nuScenes sample token in LOcc's loader.

## 3. Transfer onto the evidential geometry (this repo's env)
```bash
source paths.env            # full run; or use ./conf default for mini
pixi run locc-transfer      # mini  (pixi run locc-transfer-full for v1.0-trainval)
```
This writes:
- `locc_transfer/<scene>/LIDAR_TOP/<token>.arrow` — `semantics` [400,400,32] (2x2x2 upsample of LOcc's labels),
- `locc_clip/<scene>/LIDAR_TOP/<token>.npz` — sparse CLIP at our occupied voxels: `coords` [Nv,3] + `feat` [Nv,128] f16 (dense would be ~1.3 GB/frame).

The CLIP step needs the `evidence` stage (for the occupied voxels); discrete labels do not.

## Notes
- Licensing: LOcc weights + nuScenes are restricted — release generation code + derived
  labels, not repackaged inputs.
- The transfer reuses the same 2x2x2 Occ3D↔evidential alignment as `occ3d-transfer`.
- For storage, CLIP is stored sparsely and (recommended) only for key-frames.
