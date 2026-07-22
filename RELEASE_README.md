# Breaking the Sparse-Refresh Cost Floor with Anchored Caching for FLUX Inpainting

Anonymous code release for AAAI 2027 submission.

Training-free anchored caching and selective token refresh for a frozen
FLUX.1 Fill (12B) rectified-flow inpainting transformer. Two exact-at-anchor
levers — anchor K/V caching and dual-stream image-token sparsification —
dismantle the ~0.49x structural cost floor of naive selective refresh to
0.24x (measured 0.20x at k=1), yielding a budget-tiered quality-latency
frontier that beats uniform step reduction at matched wall-clock.

## Environment

- Python 3.11, PyTorch >= 2.3 (CUDA), diffusers >= 0.30, transformers
- One 96 GB GPU reproduces every experiment at 1024^2-1536^2; 48 GB suffices
  for 768^2-1024^2 (peak VRAM 24.8-27.0 GB at 1024^2, see supplementary)
- `pip install torch diffusers transformers lpips clean-fid open_clip_torch timm`
- Model weights download from Hugging Face on first use:
  `black-forest-labs/FLUX.1-Fill-dev` (main),
  `stabilityai/stable-diffusion-3-medium-diffusers` and
  `stabilityai/stable-diffusion-3.5-large` (transfer studies)

## Repository layout

```
models/            frozen-transformer runners
  flux_sparse_transformer.py   manual dual/single blocks, sparse_forward,
                               anchor cache, teacache/blockcache baselines,
                               FLUX_PROFILE=1 region tags, analytic MAC model
  sd3_sparse_runner.py         SD3/SD3.5 MMDiT runner (transfer studies)
samplers/
  cached_flux_fill.py          every sampling method on one backend:
                               dense / reuse / cache_sparse / teacache /
                               blockcache / temporal_thresh, all selectors
token_selectors/   mask / boundary / frequency / delta / draft scoring,
                   rank-normalized combination, presets (mbd, delta_only, ...)
data/              COCO-val manifest builder, deterministic mask specs
eval/              region metrics (mask-LPIPS->ref etc.), latency, assemble
tools/             floor_curve, latency_breakdown, selector_overhead,
                   e2e_runtime_distribution, mask_local_quality,
                   sd3_exactness / sd3_quality / sd3_masked_transfer, ...
figures/           figure generators (edge-ink auto-check)
scripts/           one stage = one script (see table below)
tests/             13 test files; mock tests run on CPU without weights
training/          optional CNN router (reliability only; the system is
                   training-free without it)
```

## Quick start

```bash
export PYTHONPATH="$PWD"

# CPU sanity (no GPU, no weights): mock math/equivalence tests
python tests/test_sparse_math_mock.py
python tests/test_teacache_mock.py
python tests/test_blockcache_mock.py

# Build the benchmark manifest (local COCO val2017 + captions required)
python -m data.dataset --build --resolution 1024 --n 100 \
    --out data/coco_manifest_1024.json

# Numerical gates first (bit-exact official-vs-manual, fresh-cache sparse)
python -m tests.test_cache_exactness

# Headline arm (dual+K/V, c=2, r=0.3, tail 4, mbd selector)
python -m samplers.cached_flux_fill --manifest data/coco_manifest_1024.json \
    --out OUT --limit 100 --method cache_sparse --selector mbd \
    --cache-period 2 --ratio 0.3 --dense-tail 4 --kv-cache --dual-sparse \
    --tag headline
python -m eval.region_metrics --run OUT/headline --ref OUT/dense_s50 \
    --manifest data/coco_manifest_1024.json --out OUT/headline/metrics.json
```

## Reproducing the paper, stage by stage

Each script is self-contained and idempotent (finished runs are skipped),
and writes `run.json` (full config + per-image walls) and `metrics.json`
next to the images. Environment variables select paths; every script
documents its own usage at the top.

| Paper section | Script |
|---|---|
| Q1 redundancy measurement | `scripts/run_stage1_dense.sh`, `run_stage2_cache.sh` |
| Q3 selectors + ablations | `run_stage3_selectors.sh` |
| Levers and frontier (512^2-1536^2) | `run_stage3b`-`run_stage3e`, `run_stage5_final.sh` |
| Router (optional) | `run_stage4_drafts.sh`, `run_stage6_router.sh` |
| Robustness, N=500 | `run_stage7_robustness.sh`, `run_stage8_n500.sh` |
| 5k real-reference FID/KID/CLIP | `run_stage9b_5k.sh` |
| TeaCache (official policy, adapted) | `run_stage10_teacache.sh` |
| Operating-point robustness | `run_stage11_polish.sh` |
| SD3 cost-transfer gates | `run_stage12_sd3.sh` |
| Measured dense curve | `run_stage13_dense_curve.sh` |
| SD3.5 t2i quality control | `run_stage14_sd3_quality.sh` |
| Measured cost floor / breakdown / overhead | `run_stage15_cost_evidence.sh` |
| Mechanism-matched baselines + selector 3-seed | `run_stage16_baselines.sh` |
| SD3.5 masked-generation transfer | `run_stage17_sd3_masked.sh` |
| Mask-local real quality (5k) | `run_stage18_masklocal.sh` |

Notes:
- All sparse execution is numerically gated: run
  `tests/test_cache_exactness.py` before trusting a new environment
  (official-vs-manual 0.0; fp32 fresh-cache <= 1.2e-6; bf16 within a
  1-2 ulp budget).
- `FLUX_PROFILE=1` enables profiler region tags (zero-cost otherwise) for
  `tools/latency_breakdown.py`.
- Latency protocol: >= 20 warmup, >= 50 timed iters, median/p10/p90 with
  `torch.cuda.synchronize` around each call; environment metadata (GPU,
  torch/CUDA versions, commit) is recorded in the floor-curve JSON.
- Seeds: per-image latent seeds come from the manifest (`latent_seed`) and
  are shared by every method; multi-seed results use
  `--seed-offset {0,1000,2000}` with per-seed dense-50 references.

## Data

COCO val2017 images and captions (standard download) are the only external
data. Masks are generated deterministically from the manifest spec
(`data/masks.py`) — no mask files are shipped or needed. The 5k protocol
uses the same builder with `--n 5000`.

## License

Code: MIT (anonymized for review). FLUX.1 Fill and SD3/SD3.5 weights are
subject to their respective licenses; this release contains no weights.
