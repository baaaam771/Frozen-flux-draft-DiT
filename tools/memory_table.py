"""tools.memory_table — P1-3: 해상도 x lever별 메모리 분해 표.

각 조합에서 anchor 1회 + sparse 1회 실행 후:
  peak allocated / peak reserved / anchor-cache 구성요소(states, single-KV,
  dual inputs+KV) / model weights.
결과는 markdown + json.

    python -m tools.memory_table --resolutions 768 1024 1536 --ratio 0.15 \
        --out memory_table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _one(pipe, runner, res, ratio, kv, dual):
    from samplers.dense_flux_fill import prepare_flux_fill_inputs
    from models.flux_cache import FluxAnchorCache
    from token_selectors.combo import select_hard_tokens
    from types import SimpleNamespace
    from PIL import Image
    import numpy as np

    dev = "cuda"
    img = Image.fromarray(np.full((res, res, 3), 127, np.uint8))
    mask = torch.zeros(1, res, res)
    mask[:, res // 4: res // 2, res // 4: res // 2] = 1
    state = prepare_flux_fill_inputs(pipe, img, mask, "a photo", 0, 4, 30.0,
                                     dev, torch.bfloat16)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cache = FluxAnchorCache()
    model_input = torch.cat([state.latents, state.cond], dim=2)
    t = state.timesteps[0].expand(1).to(state.latents.dtype) / 1000
    v, _ = runner.dense_forward(model_input, state.prompt_embeds, state.pooled,
                                t, state.guidance, state.img_ids, state.txt_ids,
                                cache=cache, record_kv=kv, record_dual=dual,
                                step_index=0)
    cache.set_anchor_context(state.latents, float(state.sigmas[0]))
    N = state.latents.shape[1]
    hp = wp = int(N ** 0.5)
    grid = SimpleNamespace(token_hw=(hp, wp))
    scores = torch.rand(1, N, device=dev)
    hard, _, _ = select_hard_tokens(scores, grid, ratio, block=1)
    _ = runner.sparse_forward(model_input, state.prompt_embeds, state.pooled,
                              t, state.guidance, state.img_ids, state.txt_ids,
                              cache, hard, kv_cache=kv, dual_sparse=dual)
    stats = torch.cuda.memory_stats()

    def _sz(ts):
        n = 0
        for t in ts:
            if t is None:
                continue
            if isinstance(t, (tuple, list)):
                n += sum(x.numel() * x.element_size() for x in t)
            else:
                n += t.numel() * t.element_size()
        return n / 2**30

    cache_gb = {
        "states": _sz(cache.single_block_inputs) + _sz(cache.dual_block_inputs)
                  + _sz([cache.entry_image_states, cache.entry_text_states,
                         cache.final_prediction, cache.anchor_latents,
                         cache.anchor_clean_estimate]),
        "single_kv": _sz(cache.single_block_kv),
        "dual_kv": _sz(cache.dual_block_kv),
    }
    assert abs(sum(cache_gb.values()) - cache.vram_bytes() / 2**30) < 1e-6
    return {
        "peak_alloc_gb": stats["allocated_bytes.all.peak"] / 2**30,
        "peak_reserved_gb": stats["reserved_bytes.all.peak"] / 2**30,
        "cache_gb": cache_gb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=int, nargs="+",
                    default=[768, 1024, 1536])
    ap.add_argument("--ratio", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from models.flux_fill_loader import load_flux_fill
    from models.flux_sparse_transformer import FluxSparseRunner
    comps = load_flux_fill(keep_text_encoders=True)
    pipe = comps.pipe
    runner = FluxSparseRunner(pipe.transformer)
    w_gb = sum(p.numel() * p.element_size()
               for p in pipe.transformer.parameters()) / 2**30

    rows, res_json = [], {"weights_gb": w_gb, "ratio": a.ratio, "cells": []}
    rows.append("| res | levers | peak alloc | peak reserved | cache total "
                "| cache 구성 |")
    rows.append("|---|---|---|---|---|---|")
    for res in a.resolutions:
        for kv, dual, name in [(False, False, "base"), (True, False, "+KV"),
                               (True, True, "+dual+KV")]:
            m = _one(pipe, runner, res, a.ratio, kv, dual)
            tot = sum(m["cache_gb"].values())
            comp = ", ".join(f"{k}={v:.2f}" for k, v in m["cache_gb"].items()
                             if v > 0.005)
            rows.append(f"| {res} | {name} | {m['peak_alloc_gb']:.1f} "
                        f"| {m['peak_reserved_gb']:.1f} | {tot:.2f} | {comp} |")
            res_json["cells"].append({"res": res, "levers": name, **m})
    md = (f"# Memory breakdown (transformer weights {w_gb:.1f} GB bf16, "
          f"r={a.ratio})\n\n" + "\n".join(rows) + "\n")
    Path(a.out).write_text(md)
    json.dump(res_json, open(str(Path(a.out).with_suffix(".json")), "w"),
              indent=1)
    print(md)


if __name__ == "__main__":
    main()
