# Open-vocabulary labels via LOcc — working recipe (validated on nuScenes-mini)

LOcc (ICCV 2025, https://github.com/pkqbajng/LOcc) generates open-vocabulary voxel labels for
nuScenes via a transitive pipeline: **LVLM vocabulary → OV-Seg (SAN) → point/voxel GT**. It
runs in its own conda envs; we consume its per-keyframe output via the `locc-transfer` stage
(`scene_reconstruction/labels/locc_transfer.py`) and the `scripts/locc_pkl_to_raw.py` converter.

This was brought up and validated end-to-end on this 12 GB RTX 3080 Ti against scene-0061.
Patched LOcc-side scripts are saved under `locc_patches/` (apply them into your LOcc clone).

## Key decisions for a 12 GB card
- **Skip the Qwen-VL LVLM step for the mini test** — `Qwen/Qwen-VL-Chat` is fp16 ~19 GB (OOM)
  and ~7 h over all of mini. SAN ships a built-in **`--fixed_vocab`** (36 nuScenes-relevant
  words), which gives a real open-vocab label without the LVLM. Use the full Qwen path on the
  L40S (see bottom).
- **GT-generation needs nuScenes lidarseg** (`PseudoOccGeneration.py` reads per-point lidarseg) —
  install it under `$NUSCENES_ROOT` (we did: `lidarseg/v1.0-*`). It needs **no** `can_bus` / bevdet
  infos (it uses the devkit directly).
- **detectron2 has no prebuilt wheel for torch 1.12/cu113**, and the system nvcc is 12.1 → build
  CPU-ops only (SAN uses no detectron2 custom CUDA ops): `CUDA_VISIBLE_DEVICES='' FORCE_CUDA=0 pip install -e .`

## Envs (this machine)
- `locc-san` — python 3.8, torch 1.12.1+cu113, SAN requirements, detectron2 v0.6 (CPU build), `ckpts/san_vit_large_14.pth`.
- `locc-gt` — clone of `daocc` (torch 1.10+cu113, mmcv-full 1.4, nuscenes-devkit) + `pip install open3d`. (scene variant needs no `chamfer_dist`.)
- `locc-llm` — python 3.10, torch 2.1.2+cu121 + 1-LVLM/requirements (only for the full Qwen path; `auto-gptq` needs a version pin compatible with transformers 4.37.2).

## Data layout (`LOcc/data/occ3d`, symlinks)
```bash
cd ~/LOcc && mkdir -p data/occ3d
NUSC=$NUSCENES_ROOT; OCC3D=<path-to>/Occ3D/nuScenes/gts
for d in samples sweeps maps v1.0-mini v1.0-test v1.0-trainval lidarseg; do ln -sfn "$NUSC/$d" data/occ3d/$d; done
ln -sfn "$OCC3D" data/occ3d/gts
```

## Run the pipeline (mini test, scene-0061)
Patched scripts (from `locc_patches/`): `2-OVSeg/SAN/main_mini.py`, `1-LVLM/qwen_vlm_step1_mini.py`,
and the `--version` patch for `3-GroundTruthGeneration/PseudoOccGeneration.py`
(`git apply locc_patches/PseudoOccGeneration_version.patch` in the LOcc clone).

```bash
ROOT=~/LOcc; OCC=$ROOT/data/occ3d
# 1) OV-Seg (SAN) with fixed vocabulary  [env locc-san]
conda run -n locc-san bash -c "cd $ROOT/2-OVSeg/SAN && python main_mini.py --fixed_vocab \
  --version v1.0-mini --scenes scene-0061 --data_root $OCC --output_root $OCC/san_qwen_scene"
# 2) GT generation -> label_ovo.pkl  [env locc-gt]  (scene-0061 = nusc.scene index 0)
conda run -n locc-gt bash -c "cd $ROOT/3-GroundTruthGeneration && python PseudoOccGeneration.py \
  --dataset nuscenes --version v1.0-mini --split all --start 0 --end 1 \
  --data_root $OCC --seg_root $OCC/san_qwen_scene --save_path $OCC/san_gts_qwen_scene"
```

## Bring it into our dataset  [pixi env]
```bash
cd ~/Documents/evidential-occupancy
pixi run python scripts/locc_pkl_to_raw.py \
  --locc_gts $OCC/san_gts_qwen_scene --vocab $OCC/san_qwen_scene/vocab.txt \
  --data_root data/nuscenes --version v1.0-mini --extra data/nuscenes_extra      # -> locc_raw/ + vocab.json
pixi run python -m scene_reconstruction.cli.main export ./conf preview locc-transfer  # -> locc_transfer/*.arrow
pixi run python scripts/vis_occupancy_viser.py --num_samples 5                        # Color by: locc_class
```
`locc_transfer` writes per-voxel open-vocab class on our 0.2 m grid (2×2×2 upsample of LOcc's
[200,200,16]); `vocab.json` is the class-index→name map. The viewer's **`locc_class`** mode colors
by it. On scene-0061 it distinguishes tree / building / grass / wall / fence / traffic light / sign —
finer than Occ3D's manmade/vegetation.

## `locc-project` — label OUR occupancy directly (recommended for full coverage)

`locc-transfer` (above) lifts LOcc's *own* occupancy: LOcc voxelizes lidar points and labels
only voxels a lidar return landed in, so evidential voxels LOcc's lidar pass missed — including
low-confidence ones — stay unlabeled (~72% of our occupied voxels get a class on mini).

`locc-project` (`scene_reconstruction/labels/locc_project.py`) decouples occupancy from
labeling, like LOcc's `PseudoOccGeneration-VoxelProjection.py` but driven by **our** geometry:
for every voxel our `evidence` marks `occupied`, it projects the voxel center (ego frame) into
each camera's SAN OV-Seg image and reads the class there. Occupancy comes entirely from the
evidence stage, so every occupied voxel gets a class wherever a camera sees it. It needs only the
**SAN OV-Seg PNGs** (LOcc step 2) — no GT-generation, no lidarseg, no `can_bus`.

**Temporal aggregation** (`temporal: true`, default): a single frame can't label the camera/lidar
blind spot around the ego, and per-frame labels flicker. So labels are pooled across the scene's
key-frames and voted, split like LOcc's point-based GT: **background** voxels
(`scene_flow.scene_instance_index == 0`) are pooled in the **global frame** (via `global_from_ego`)
— so the ground under the ego inherits the label a camera gave it at another frame — while
**boxed objects** are voted **per scene-instance** and written at each frame's de-warped position,
so objects can't streak a trail of their label across the world. On mini this lifts coverage to
**~99.9%** (vs ~72% for `locc-transfer`), fills the under-ego patch (0% → ~97% labeled there), and
makes every object a single consistent class. Set `temporal: false` for the raw per-frame labels.

```bash
cd ~/Documents/evidential-occupancy
# uses conf node export.locc_project (seg_root -> the SAN san_qwen_scene dir; writes vocab.json)
pixi run python -m scene_reconstruction.cli.main export ./conf preview locc-project
pixi run python scripts/vis_occupancy_viser.py --num_samples 5    # Color by: locc_project
```
Output `<extra>/locc_project/<scene>/LIDAR_TOP/<token>.arrow` carries
`LIDAR_TOP.locc_project.semantics` int16 [1,400,400,32] (`-1` = unlabeled), labelled natively at
0.2 m (no 2×2×2 upsample). The viewer's **`locc_project`** color mode shows it; keep
**`locc_class`** to compare against LOcc's own voxelization. On the L40S set `LOCC_SEG_ROOT` /
`LOCC_VOCAB` (see `conf/full.yaml`); the stage is opt-in in `scripts/generate_shard.py`
(`--steps … locc-project`).

CAVEAT (intentional): voxel-center projection has no occlusion test, so a voxel occluded from a
camera inherits whatever surface is in front of it. A depth gate (keep the label only where the
voxel's projected depth matches the first surface) is the natural fix if results need it.

## Full open-vocab (Qwen) path — recommended on the L40S
Run steps 1-3 in order with the real LVLM vocabulary instead of `--fixed_vocab`:
```bash
# step 1 (locc-llm): qwen_vlm_step1_mini.py (Int4 model on a 12 GB card; full model on L40S) -> qwen_texts_step1
#   then qwen_vlm_step2.py / step3.py -> qwen_texts   (pin auto-gptq for transformers 4.37.2, or use the fp16 model on 48 GB)
# step 2 (locc-san): drop --fixed_vocab, add --vocab_root $OCC/qwen_texts
# step 3 (locc-gt): same as above
```
For CLIP features (a re-queryable field), use `PseudoOccGeneration-Feat.py` + `4-Autoencoder/`
to compress CLIP 512→128, then extend the converter to emit `feat_coords`/`feat` (locc-transfer
already stores them sparsely under `locc_clip/`).
