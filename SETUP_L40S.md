# evidential-occupancy — L40S server setup & full data generation

End-to-end guide for generating the full nuScenes (v1.0-trainval) occupancy ground truth on a
multi-GPU box (tuned for 5× L40S). One-time setup, then a single command runs all 850 scenes
across every GPU in parallel. Everything is copy-paste, in order — no agent required.

The pipeline has three steps per scene: **transmissions-reflections** (per-LIDAR-frame spherical
reflection/transmission evidence) → **scene-flow** (per-frame instance motion from boxes) →
**temporal-accumulation** (accumulate frames into the ego volume; this is the final occupancy
evidence). Occupancy = Dempster-Shafer belief over those counts.

---

## Gotchas (read this first)

These are the things that bite on a fresh machine; the commands below already handle them.

1. **The code is single-GPU by default** (`.cuda()` = device 0). Multi-GPU happens by launching
   one process per GPU over a disjoint **shard of scenes** (`scripts/run_full_multi_gpu.sh`). Don't
   run the plain `pixi run data-processing-full` if you want all 5 GPUs — that uses one GPU.
2. **Paths are NOT hard-coded.** Set them once in `paths.env` (copied from `paths.env.example`) and
   `source paths.env`. The configs read `${oc.env:NUSCENES_ROOT,...}` / `${oc.env:NUSCENES_EXTRA_ROOT,...}`.
3. **Output disk: ~1.2 TB, and it must be FAST LOCAL NVMe.** The accumulation step does millions of
   small random reads of the intermediate `.arrow` files; on a network/spinning disk that IO becomes
   the bottleneck and erases the multi-GPU speedup.
4. **The annotation cache is built once, single-threaded, before the parallel launch.** The launcher
   does this (`--prime-cache`) so the 5 scene-flow workers don't race to write the same cache file.
   Priming loads the trainval devkit and takes a few minutes — this is expected.
5. **RAM:** each worker loads the trainval metadata (~1.3 GB JSON → a few GB resident). 5 workers
   start together, so expect ~15–25 GB RAM during startup. Fine on a server; just don't be surprised.
6. **Resumable.** Every step runs with `missing_only=true`, so if anything dies, just re-run the same
   launcher command — finished frames are skipped.
7. **Old CUDA stack on purpose.** pixi builds CUDA 11.6 / PyTorch 1.12 / kaolin / spconv. The L40S
   driver is forward-compatible with the 11.6 runtime, so this works; don't "upgrade" the toml.

Conventions: all commands run from the repo root on the host. `$` prompt = your shell.

---

## PART 0 — One-time setup

### 0.1 Clone the repo
```bash
git clone https://github.com/ArthurJakobsson/evidential-occupancy.git ~/evidential-occupancy
cd ~/evidential-occupancy
```

### 0.2 Install pixi (if not already on the machine)
```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc          # put `pixi` on PATH (or use ~/.pixi/bin/pixi explicitly below)
pixi --version
```

### 0.3 Build the environment (CUDA 11.6 / torch 1.12 / kaolin / spconv)
```bash
cd ~/evidential-occupancy
pixi install              # several minutes; downloads the full CUDA/torch stack
```

### 0.4 Set the per-machine paths — the ONE place to edit
```bash
cp paths.env.example paths.env
nano paths.env            # set NUSCENES_ROOT, NUSCENES_EXTRA_ROOT, NUM_GPUS
source paths.env
```
- `NUSCENES_ROOT` = the dir containing `samples/ sweeps/ maps/ v1.0-trainval/ ...`
- `NUSCENES_EXTRA_ROOT` = where outputs are written (fast local NVMe, ~1.2 TB free)
- `NUM_GPUS` = `5`
- If `pixi` is not on PATH, also set `export PIXI="$HOME/.pixi/bin/pixi"` in `paths.env`.

### 0.5 Sanity-check the setup
```bash
pixi run python scripts/check_setup.py
```
Expect `ALL GOOD — ready to run.` It verifies the dataset paths exist, the output root is writable,
and CUDA sees your GPUs. (If `NUSCENES_ROOT` shows the `./data` fallback, you forgot `source paths.env`.)

### 0.6 (optional) Preview viewer dependency
Only needed if you want the interactive viser viewer in Part 2; not needed for generation:
```bash
pixi run pip install viser
```

---

## PART 1 — Full generation across all GPUs

One command. It primes the annotation cache once, then launches `NUM_GPUS` workers, each taking a
contiguous shard of the 850 scenes through all three steps. With 5 GPUs that's 170 scenes/GPU.

```bash
source paths.env                       # if a new shell
nohup bash scripts/run_full_multi_gpu.sh > logs/run.log 2>&1 &
```
`nohup ... &` keeps it running if your SSH session drops. Watch progress:
```bash
tail -f logs/gpu_*.log                 # per-GPU progress bars (one file per GPU)
nvidia-smi                             # confirm all GPUs are busy
```

**Expected time (5× L40S, fast NVMe):** roughly **16–24 hours**, dominated by temporal-accumulation
(~0.7 day). For reference: transmissions-reflections ~1–2 h, scene-flow ~1–2 h, accumulation ~14–20 h.
(Measured single-GPU rates extrapolated to 850 scenes / 331,886 LIDAR frames / 34,149 keyframes; the
L40S per-GPU factor over the dev GPU is the softest part of the estimate.)

**If a worker dies:** just re-run the same `nohup bash scripts/run_full_multi_gpu.sh ...` line. It
resumes (skips finished frames) and the cache prime is a no-op the second time.

---

## PART 2 — After it finishes

### 2.1 Verify outputs
```bash
source paths.env
echo "accumulated keyframes:"; find "$NUSCENES_EXTRA_ROOT/reflection_and_transmission_multi_frame" -name '*.arrow' | wc -l
# expect ~34,149
du -sh "$NUSCENES_EXTRA_ROOT"/*/
```

### 2.2 Reclaim space (optional)
`reflection_and_transmission_multi_frame/` is the final ground truth. The other two dirs are
intermediates only needed during the run:
```bash
rm -rf "$NUSCENES_EXTRA_ROOT/reflection_and_transmission_spherical" \
       "$NUSCENES_EXTRA_ROOT/scene_flow"
```

### 2.3 (optional) Inspect a few samples in 3D
```bash
pixi run pip install viser    # if not done already
pixi run python scripts/vis_occupancy_viser.py \
  --data_dir "$NUSCENES_EXTRA_ROOT/reflection_and_transmission_multi_frame" --num_samples 5
# open the printed URL (forward the port over SSH: ssh -L 8080:localhost:8080 ...)
```

### 2.4 (optional) Evaluate
```bash
# eval configs also read ${oc.env:...} paths via data/nuscenes_extra; point them at the drive or
# symlink: ln -sfn "$NUSCENES_EXTRA_ROOT" data/nuscenes_extra && ln -sfn "$NUSCENES_ROOT" data/nuscenes
pixi run eval-bba
```

---

## Manual / single-GPU alternatives

**One GPU, whole dataset** (slowest, ~5–6 days):
```bash
source paths.env
pixi run data-processing-full
```

**Run a specific shard by hand** (e.g. GPU 2 of a 5-way split):
```bash
source paths.env
CUDA_VISIBLE_DEVICES=2 pixi run python scripts/generate_shard.py --gpu-index 2 --num-gpus 5
```

**Run a single step for a shard** (`--steps` takes any subset, in order):
```bash
CUDA_VISIBLE_DEVICES=0 pixi run python scripts/generate_shard.py \
  --gpu-index 0 --num-gpus 5 --steps transmissions-reflections
```

**Different GPU count:** set `NUM_GPUS` in `paths.env` (the launcher and shard math follow it).

**Dry-run the plan (no GPU work):**
```bash
pixi run python scripts/generate_shard.py --gpu-index 0 --num-gpus 5 --dry-run
```

---

## Quick reference

| Action | Command |
|--------|---------|
| Setup check | `pixi run python scripts/check_setup.py` |
| Full run, all GPUs | `nohup bash scripts/run_full_multi_gpu.sh > logs/run.log 2>&1 &` |
| Watch progress | `tail -f logs/gpu_*.log` |
| Resume after crash | re-run the launcher (same command) |
| One shard | `CUDA_VISIBLE_DEVICES=g pixi run python scripts/generate_shard.py --gpu-index g --num-gpus N` |
| Prime cache only | `pixi run python scripts/generate_shard.py --gpu-index 0 --num-gpus N --prime-cache` |
| Count outputs | `find "$NUSCENES_EXTRA_ROOT/reflection_and_transmission_multi_frame" -name '*.arrow' \| wc -l` |

---

## Troubleshooting

- **`check_setup.py` shows `./data` fallback / MISS** → you didn't `source paths.env`, or a path is
  wrong. Fix `paths.env`, `source paths.env`, re-check.
- **`pixi: command not found`** → `source ~/.bashrc`, or set `export PIXI="$HOME/.pixi/bin/pixi"` in
  `paths.env` (the launcher honors `$PIXI`).
- **`ColumnNotFoundError: index`** → your nuScenes lacks the lidarseg `category.index` field; this is
  already handled in `read_json` (synthesized). If you see it, you're on an old checkout — `git pull`.
- **A worker exits early / `CUDA out of memory`** → lower `batch_size` for transmissions_reflections
  in `conf/full.yaml` (4 → 2), then re-run the launcher (resumes).
- **Throughput much slower than ~1 day** → outputs are on a slow/network disk; move
  `NUSCENES_EXTRA_ROOT` to local NVMe. Also confirm `nvidia-smi` shows all GPUs active.
- **`Default process group not initialized` / distributed errors** → not used here; this pipeline is
  plain single-process-per-GPU. If you see it, you launched the wrong script.
- **Re-run seems to redo work** → make sure you're using `conf/full.yaml` (has `missing_only: true`);
  the per-shard runner uses it by default.
