"""samplers.cached_flux_fill — Stage 4–7: FreqSpec-Cache-FLUX sampling loop.

Per denoising step, one of three modes (plan Sec. 1):

  dense anchor step   every c steps: full transformer, record depth-aligned cache
  sparse refresh step selector -> hard tokens -> single-stream selective refresh;
                      easy tokens reuse same-depth cached states as K/V context
                      and the anchor's final prediction as their output
  (r = 0)             anchored prediction reuse: scheduler-only step, zero
                      model calls between anchors (DACE's strongest draft-free
                      baseline; the sampler falls back to this when --ratio 0)

Methods exposed through --method:
  dense           reduced-step dense baseline (use --steps to sweep 50/40/30/...)
  reuse           r = 0 anchored reuse, anchor period --cache-period
  cache_sparse    anchor + selective refresh with --selector
                    {mask, mask_boundary, mask_delta, mask_frequency,
                     mbd, mbfd, mbfd_draft, random, oracle}
  hetero          Q1 diagnostic: dense every step, log per-token temporal change
                  in/out mask (the DACE two-factor deployment test)

All methods consume the frozen manifest (data.dataset) so every method sees the
same image / mask / prompt / latent seed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data.dataset import FluxFillBenchmark
from data.prompt_cache import load_cached
from models.flux_cache import FluxAnchorCache
from models.flux_fill_loader import load_flux_fill, unload_text_encoders
from models.flux_sparse_transformer import FluxSparseRunner
from samplers.dense_flux_fill import (FluxFillState, decode_latents,
                                      prepare_flux_fill_inputs, scheduler_step,
                                      transformer_forward)
from token_selectors.boundary import boundary_score
from token_selectors.combo import (PRESETS, combo_score, oracle_score, random_score,
                             select_hard_tokens)
from token_selectors.delta import delta_score
from token_selectors.frequency import frequency_score
from token_selectors.mask import mask_score
from utils.flow_math import clean_estimate
from utils.token_mapping import TokenGrid


# --------------------------------------------------------------- selectors ----
class SelectorState:
    """Precomputes the static priors (M, B) once per sample and evaluates the
    dynamic terms (F, Δ, A^D) per sparse step from the cache."""

    def __init__(self, name: str, mask_px: torch.Tensor, grid: TokenGrid,
                 pipe, freq_source: str = "anchor_x0", draft=None):
        self.name = name
        self.grid = grid
        self.pipe = pipe
        self.freq_source = freq_source
        self.draft = draft
        self.mask_tok = mask_score(mask_px, grid)                       # [B, N]
        self.bnd_tok = boundary_score(self.mask_tok, grid)

    def _unpack(self, packed: torch.Tensor) -> torch.Tensor:
        return self.pipe._unpack_latents(
            packed, self.grid.height, self.grid.width, self.pipe.vae_scale_factor)

    @torch.no_grad()
    def scores(self, latents: torch.Tensor, sigma_t, cache: FluxAnchorCache,
               t: torch.Tensor, generator=None,
               v_dense_now: torch.Tensor | None = None) -> torch.Tensor:
        dev = latents.device
        B, N, _ = latents.shape
        if self.name == "random":
            return random_score(B, N, generator=generator, device=dev)
        if self.name == "oracle":
            assert v_dense_now is not None, "oracle needs the extra dense pass"
            return oracle_score(v_dense_now, cache.final_prediction)

        w = PRESETS[self.name]
        freq = delta = draft_term = None
        if w.gamma != 0.0:
            if self.freq_source == "noisy":
                src = self._unpack(latents)
            elif self.freq_source == "cached_v_current_x0":
                # z_t with the STALE anchor velocity — a mixed estimate, kept
                # only as an explicit ablation arm; this is NOT the anchor x0.
                src = self._unpack(clean_estimate(latents, cache.final_prediction, sigma_t))
            else:  # 'anchor_x0' (default): TRUE anchor clean estimate
                #   x0_hat_a = z_a - sigma_a * v_a, precomputed at the anchor
                assert cache.anchor_clean_estimate is not None, \
                    "anchor_x0 requires cache.set_anchor_context() at anchor steps"
                src = self._unpack(cache.anchor_clean_estimate)
            freq = frequency_score(src, self.grid).to(dev)
        if w.delta != 0.0 and cache.prev_final_prediction is not None:
            delta = delta_score(cache.final_prediction, cache.prev_final_prediction)
        if w.eta != 0.0 and self.draft is not None:
            # router-style draft outputs a per-token difficulty score directly
            draft_term = self.draft.scores(latents, self.mask_tok, cache, t, self.grid)
        return combo_score(w, mask=self.mask_tok.to(dev), boundary=self.bnd_tok.to(dev),
                           frequency=freq, delta=delta, draft=draft_term)


# ------------------------------------------------------------------ sampler ---
@torch.no_grad()
def sample_one(pipe, runner: FluxSparseRunner, state: FluxFillState, *,
               method: str, cache_period: int, ratio: float, selector: str,
               block: int, mask_px: torch.Tensor, freq_source: str,
               dense_head: int = 0, dense_tail: int = 0,
               draft=None, log: dict | None = None):
    """Runs one image through the chosen method; mutates state.latents; returns
    stats dict (target evals, sparse fraction, per-step records)."""
    grid = state.grid
    cache = FluxAnchorCache()
    sel = SelectorState(selector, mask_px, grid, pipe, freq_source, draft) \
        if method == "cache_sparse" else None
    g = torch.Generator(state.latents.device).manual_seed(0)

    n_anchor = n_sparse = 0
    attn_fracs, mac_ratios, actual_ratios = [], [], []
    hetero_rows = []
    v_prev = None
    n_steps = len(state.timesteps)

    for i, t in enumerate(state.timesteps):
        sigma = pipe.scheduler.sigmas[i]
        # schedule-aware policy: hetero 측정에서 마지막 ~4 step은 변화가 mask 밖으로
        # 퍼지고(in/out 25x -> 0.4) energy가 튐 -> 그 구간은 무조건 dense(anchor).
        forced_dense = (i < dense_head) or (i >= n_steps - dense_tail)
        is_anchor = (method in ("reuse", "cache_sparse")) and \
                    (i % cache_period == 0 or forced_dense)

        if method in ("dense", "hetero") or is_anchor:
            model_input_cache = cache if is_anchor else None
            model_input = torch.cat([state.latents, state.cond], dim=2)
            timestep = t.expand(1).to(state.latents.dtype) / 1000
            v, _ = runner.dense_forward(model_input, state.prompt_embeds, state.pooled,
                                        timestep, state.guidance, state.img_ids,
                                        state.txt_ids, cache=model_input_cache,
                                        step_index=i)
            n_anchor += 1
            if is_anchor:
                # Fix 1: precompute the TRUE anchor clean estimate x0_a = z_a - s_a*v_a
                cache.set_anchor_context(state.latents, sigma)
            if method == "hetero" and v_prev is not None:
                d = (v.float() - v_prev.float()).pow(2).mean(-1)        # [B, N]
                m = sel_mask(mask_px, grid, d.device)
                hetero_rows.append(_hetero_row(i, d, m, v))
            v_prev = v
        elif ratio == 0.0 or method == "reuse":
            v = cache.final_prediction                                   # r = 0
        else:
            v_dense_now = None
            if selector == "oracle":                                     # upper bound
                model_input = torch.cat([state.latents, state.cond], dim=2)
                timestep = t.expand(1).to(state.latents.dtype) / 1000
                v_dense_now, _ = runner.dense_forward(
                    model_input, state.prompt_embeds, state.pooled, timestep,
                    state.guidance, state.img_ids, state.txt_ids)
            scores = sel.scores(state.latents, sigma, cache, t, generator=g,
                                v_dense_now=v_dense_now)
            hard_idx, _, r_actual = select_hard_tokens(scores, grid, ratio, block=block)
            model_input = torch.cat([state.latents, state.cond], dim=2)
            timestep = t.expand(1).to(state.latents.dtype) / 1000
            v_hard, st = runner.sparse_forward(model_input, state.prompt_embeds,
                                               state.pooled, timestep, state.guidance,
                                               state.img_ids, state.txt_ids,
                                               cache, hard_idx)
            v = runner.merge_prediction(cache, hard_idx, v_hard)
            n_sparse += 1
            attn_fracs.append(st.single_attn_fraction)
            mac_ratios.append(st.est_transformer_mac_ratio)
            actual_ratios.append(r_actual)

        state.latents = scheduler_step(pipe, v, t, state.latents)

    stats = {"anchor_evals": n_anchor, "sparse_steps": n_sparse,
             "mean_single_attn_fraction": (sum(attn_fracs) / len(attn_fracs))
             if attn_fracs else None,
             # Fix 3: with block > 1 the realized refresh ratio != requested ratio
             "mean_actual_ratio": (sum(actual_ratios) / len(actual_ratios))
             if actual_ratios else None,
             # Fix 10: whole-transformer MAC estimate (dense dual + full K/V included)
             "mean_est_transformer_mac_ratio": (sum(mac_ratios) / len(mac_ratios))
             if mac_ratios else None}
    if hetero_rows:
        stats["heterogeneity"] = hetero_rows
    if log is not None:
        log.update(stats)
    return stats


def sel_mask(mask_px, grid, device):
    return mask_score(mask_px, grid).to(device)


def _hetero_row(step, d, m, v_now):
    """Q1 row — Factor A (spatial concentration) AND the per-step half of
    Factor B: E_rel = mean||v_t - v_{t-1}||² / mean||v_t||² (consequence scale).
    The other half of Factor B (step-reduction quality sensitivity) is joined
    in eval.heterogeneity from the dense-step-sweep metrics (Fix 4)."""
    dm = d.flatten()
    k = max(1, int(0.3 * dm.numel()))
    top = torch.topk(dm, k).values.sum() / dm.sum().clamp_min(1e-12)
    inm = (m.flatten() > 0.5)
    return {
        "step": int(step),
        "top30_share": top.item(),
        "cv": (dm.std() / dm.mean().clamp_min(1e-12)).item(),
        "in_mask_mean": dm[inm].mean().item() if inm.any() else 0.0,
        "out_mask_mean": dm[~inm].mean().item() if (~inm).any() else 0.0,
        "energy_ratio": (dm.mean() / v_now.float().pow(2).mean().clamp_min(1e-12)).item(),
    }


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["dense", "reuse", "cache_sparse", "hetero"],
                    required=True)
    ap.add_argument("--selector", default="mask",
                    choices=list(PRESETS) + ["random", "oracle"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cache-period", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--block", type=int, default=1,
                    help="structured selection window in tokens (Stage 7): 1, 2, 4")
    ap.add_argument("--freq-source", default="anchor_x0",
                    choices=["anchor_x0", "cached_v_current_x0", "noisy"])
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-cache", default="")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="added to each sample's manifest latent_seed (Stage 8 multi-seed)")
    ap.add_argument("--draft-ckpt", default="",
                    help="CNN router checkpoint for mbfd_draft (Stage 6)")
    ap.add_argument("--dense-head", type=int, default=0,
                    help="처음 K step 강제 dense (anchor)")
    ap.add_argument("--dense-tail", type=int, default=0,
                    help="마지막 K step 강제 dense — hetero 곡선의 말기 붕괴 구간 방어")
    ap.add_argument("--prefetch", action=__import__("argparse").BooleanOptionalAction,
                    default=True, help="background next-sample loading (--no-prefetch로 끔)")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    comps = load_flux_fill(keep_text_encoders=not a.prompt_cache)
    pipe, dev, dtype = comps.pipe, comps.device, comps.dtype
    runner = FluxSparseRunner(pipe.transformer)
    draft = None
    if a.draft_ckpt:
        from models.drafts.router_draft import RouterDraft
        draft = RouterDraft.load(a.draft_ckpt, dev)
    ds = FluxFillBenchmark(a.manifest)
    n = len(ds) if a.limit == 0 else min(a.limit, len(ds))

    # 다음 sample(이미지 로드 + mask 생성)을 GPU가 도는 동안 백그라운드로 준비.
    # sample당 50-step FLUX(수십 초) 대비 데이터(수십 ms)라 이득은 작지만 공짜이고,
    # rows[i]["data_s"]로 데이터가 GPU를 실제로 막는지 직접 확인 가능.
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1) if a.prefetch else None
    pending = pool.submit(ds.__getitem__, 0) if pool else None

    rows = []
    for i in range(n):
        t_data = time.perf_counter()
        if pool:
            s = pending.result()
            if i + 1 < n:
                pending = pool.submit(ds.__getitem__, i + 1)
        else:
            s = ds[i]
        data_s = time.perf_counter() - t_data
        pe = po = None
        if a.prompt_cache:
            pe, po = load_cached(a.prompt_cache, s["prompt"], dev, dtype)
        state = prepare_flux_fill_inputs(
            pipe, s["image"], s["mask"], s["prompt"],
            s["latent_seed"] + a.seed_offset,
            a.steps, a.guidance, dev, dtype, prompt_embeds=pe, pooled=po)

        # Fix 7: measure each sample from a clean slate; the first sample stays
        # flagged as warm-up (compile/cudnn autotune/allocator growth) and is
        # excluded from wall-clock aggregation in eval.assemble.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        log: dict = {}
        sample_one(pipe, runner, state, method=a.method,
                   cache_period=a.cache_period, ratio=a.ratio,
                   selector=a.selector, block=a.block,
                   mask_px=s["mask"].unsqueeze(0).to(dev), freq_source=a.freq_source,
                   dense_head=a.dense_head, dense_tail=a.dense_tail,
                   draft=draft, log=log)
        img = decode_latents(pipe, state)
        torch.cuda.synchronize()
        log.update({"sample_id": s["sample_id"], "bucket": s["bucket"],
                    "mask_type": s["mask_type"], "warmup": i == 0,
                    "data_s": data_s,
                    "wall_s": time.perf_counter() - t0,
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30})
        rows.append(log)

        from samplers.dense_flux_fill import _save_img
        stem = Path(s["sample_id"]).stem
        _save_img(img, out / f"{stem}.png")                     # raw model output
        # Fix 6: composited output x_paste = M*x_model + (1-M)*x_input — FLUX Fill
        # does not mathematically guarantee known-region identity in raw output.
        m_px = s["mask"].to(img.device, img.dtype)              # [1,H,W]
        inp = s["image"].to(img.device, img.dtype)              # [3,H,W]
        _save_img(m_px * img + (1 - m_px) * inp, out / f"{stem}_pasted.png")
        torch.save(s["mask"], out / f"{stem}_mask.pt")

    cfg = vars(a)
    json.dump({"config": cfg, "rows": rows}, open(out / "run.json", "w"), indent=1)
    print(f"[{a.tag}] {n} samples -> {out}")


if __name__ == "__main__":
    main()
