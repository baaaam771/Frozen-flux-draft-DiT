"""eval.latency — Gate F: CUDA-synced wall-clock of dense / sparse forwards.

DACE's warning applies with more force here in reverse: a 12B FLUX forward is
compute-bound (not kernel-launch bound like DiT-S), so token-sparse single-
stream execution SHOULD convert MAC savings into latency even at batch 1 —
this script measures exactly that, per hard ratio, plus end-to-end sampling
time including VAE and scheduler, with mean / median / p90 and peak VRAM.

    python -m eval.latency --resolution 512 --ratios 0.1 0.3 0.5 --iters 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from models.flux_cache import FluxAnchorCache
from models.flux_fill_loader import load_flux_fill
from models.flux_sparse_transformer import (FluxSparseRunner, estimate_transformer_macs,
                                            prepare_latent_image_ids)
from utils.token_mapping import TokenGrid


def _timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {"mean_ms": 1e3 * statistics.mean(ts),
            "median_ms": 1e3 * statistics.median(ts),
            "p90_ms": 1e3 * sorted(ts)[int(0.9 * len(ts)) - 1]}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--kv-cache", action="store_true")
    ap.add_argument("--dual-sparse", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="latency.json")
    a = ap.parse_args()

    comps = load_flux_fill(keep_text_encoders=False)
    dev, dtype = comps.device, comps.dtype
    runner = FluxSparseRunner(comps.transformer)
    grid = TokenGrid(a.resolution, a.resolution).validate()
    hp, wp = grid.token_hw
    N = grid.num_image_tokens

    x = torch.randn(1, N, 384, device=dev, dtype=dtype)
    pe = torch.randn(1, a.text_len, comps.transformer.config.joint_attention_dim,
                     device=dev, dtype=dtype)
    po = torch.randn(1, comps.transformer.config.pooled_projection_dim,
                     device=dev, dtype=dtype)
    ts = torch.full((1,), 0.5, device=dev, dtype=dtype)
    gd = torch.full((1,), 30.0, device=dev, dtype=torch.float32) \
        if comps.transformer.config.guidance_embeds else None
    img_ids = prepare_latent_image_ids(hp, wp, dev, dtype)
    txt_ids = torch.zeros(a.text_len, 3, device=dev, dtype=dtype)

    report = {"resolution": a.resolution, "tokens": N,
              "scope": "transformer-only (denoise loop, VAE, text enc excluded)",
              "kv_cache": a.kv_cache, "dual_sparse": a.dual_sparse}
    cache = FluxAnchorCache()
    torch.cuda.reset_peak_memory_stats()
    report["dense"] = _timeit(lambda: runner.dense_forward(
        x, pe, po, ts, gd, img_ids, txt_ids), a.iters)
    report["dense"]["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    report["anchor(record)"] = _timeit(lambda: runner.dense_forward(
        x, pe, po, ts, gd, img_ids, txt_ids, cache=cache, step_index=0,
        record_kv=a.kv_cache, record_dual=a.dual_sparse), a.iters)
    report["anchor(record)"]["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    report["cache_vram_gb"] = cache.vram_bytes() / 2**30

    for r in a.ratios:
        k = max(1, int(r * N))
        hard = torch.sort(torch.randperm(N, device=dev)[:k]).values[None]
        torch.cuda.reset_peak_memory_stats()
        report[f"sparse_r{r}"] = _timeit(lambda: runner.sparse_forward(
            x, pe, po, ts, gd, img_ids, txt_ids, cache, hard,
            kv_cache=a.kv_cache, dual_sparse=a.dual_sparse), a.iters)
        report[f"sparse_r{r}"]["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
        D = (comps.transformer.config.num_attention_heads
             * comps.transformer.config.attention_head_dim)
        report[f"sparse_r{r}"]["est_mac_ratio"] = estimate_transformer_macs(
            a.text_len, N, k, len(comps.transformer.transformer_blocks),
            len(comps.transformer.single_transformer_blocks), D,
            kv_cached=a.kv_cache, dual_sparse=a.dual_sparse)["mac_ratio"]
        report[f"sparse_r{r}"]["speedup_vs_dense"] = (
            report["dense"]["median_ms"] / report[f"sparse_r{r}"]["median_ms"])

    report["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
