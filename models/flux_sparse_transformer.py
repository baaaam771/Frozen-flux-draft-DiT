"""models.flux_sparse_transformer — dense / anchor / sparse forward for FLUX Fill.

This is the execution layer the verifier papers deferred, ported from DACE
(ImageNet-64 DiT) to the FLUX.1 Fill [dev] architecture:

  * dual-stream 19 blocks : ALWAYS dense (plan Sec. 3 first-PoC safety rule)
  * single-stream 38 blocks:
        anchor step -> dense, recording each block's image-token INPUT states
        sparse step -> queries = [all text tokens ; hard image tokens] (fresh),
                       K/V ctx = [fresh text ; fresh hard ; anchor-cached easy]
                       — easy context is depth-correct and only time-stale,
                       never frozen across depth (DACE Sec. 4.1/4.2).

Three forward modes:
    dense_forward(...)                      Stage 1 (Gate A) & anchors
    sparse_forward(..., hard_idx)           Stage 4+ selective refresh
    (r = 0 needs no forward at all: the sampler reuses cache.final_prediction)

Exactness property (Gate B, tests/test_cache_exactness.py):
    with a fresh cache (anchor and sparse step at the same latent/timestep),
    sparse_forward's hard-token outputs equal dense outputs bit-for-bit modulo
    bf16 reduction order, at ANY hard ratio — because the scattered context is
    then literally the dense context.

Implementation notes:
  * We re-implement the single-stream block forward from its own submodules
    (norm / proj_mlp / act_mlp / attn.to_q|to_k|to_v|norm_q|norm_k / proj_out)
    for BOTH the dense and sparse paths, so the two paths share every numeric
    op and the exactness test is meaningful. Dual-stream blocks are invoked
    as-is. flux_fill_loader hard-asserts this module layout at load time.
  * Single-stream FLUX attention has no output projection: sdpa output is
    concatenated with the parallel MLP branch and passed through proj_out.
  * RoPE: diffusers FluxPosEmbed returns (cos, sin) of shape [S, head_dim];
    we gather rows for the query subset (per-batch) and keep full rows for K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import os

import torch
import torch.nn.functional as F

from .flux_cache import FluxAnchorCache


# ------------------------------------------------------------------ helpers ---
def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """diffusers Flux RoPE (use_real, unbind_dim=-1).
    x: [B, H, S, D]; cos/sin: [S, D] or [B, 1, S, D]."""
    if cos.dim() == 2:
        cos = cos[None, None]
        sin = sin[None, None]
    xf = x.float()
    x_pair = xf.reshape(*xf.shape[:-1], -1, 2)
    x_rot = torch.stack([-x_pair[..., 1], x_pair[..., 0]], dim=-1).reshape_as(xf)
    return (xf * cos.float() + x_rot * sin.float()).to(x.dtype)


def _heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, S, D = x.shape
    return x.view(B, S, n_heads, D // n_heads).transpose(1, 2)          # [B,H,S,d]


def _unheads(x: torch.Tensor) -> torch.Tensor:
    B, H, S, d = x.shape
    return x.transpose(1, 2).reshape(B, S, H * d)


_PROFILE = os.environ.get("FLUX_PROFILE") == "1"
if _PROFILE:
    from torch.profiler import record_function as _rf
else:
    import contextlib

    def _rf(name):                     # noqa: D401 — zero-cost stand-in
        return contextlib.nullcontext()


def _gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, S, D], idx [B, k] -> [B, k, D]."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _scatter_tokens(base: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """out-of-place scatter: base [B, S, D] with values [B, k, D] at idx [B, k]."""
    out = base.clone()
    out.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, base.shape[-1]), values)
    return out


def prepare_latent_image_ids(hp: int, wp: int, device, dtype) -> torch.Tensor:
    """Row-major (Hp, Wp) positional ids — mirrors FluxFillPipeline."""
    ids = torch.zeros(hp, wp, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(hp, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(wp, device=device, dtype=dtype)[None, :]
    return ids.reshape(hp * wp, 3)


# --------------------------------------------------------------- single blk ---
def _single_block_dense(block, x: torch.Tensor, temb: torch.Tensor,
                        cos: torch.Tensor, sin: torch.Tensor,
                        return_kv_img_from: int | None = None):
    """Manual FluxSingleTransformerBlock forward on the full [text;image] seq.
    return_kv_img_from=T additionally returns the image-token (k, v) — k post-rope
    (rope depends on position only), v raw — for the anchor K/V cache (Lever B).
    No extra GEMMs: the same k/v used for attention are sliced."""
    residual = x
    with _rf("single_q_mlp"):
        normed, gate = block.norm(x, emb=temb)
        mlp_h = block.act_mlp(block.proj_mlp(normed))
        attn = block.attn
        q = _heads(attn.to_q(normed), attn.heads)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
    with _rf("single_kv_projection"):
        k = _heads(attn.to_k(normed), attn.heads)
        v = _heads(attn.to_v(normed), attn.heads)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
    with _rf("single_attention"):
        q = _rope_apply(q, cos, sin)
        k = _rope_apply(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)
        o = _unheads(o).to(x.dtype)
    with _rf("single_q_mlp"):
        out = block.proj_out(torch.cat([o, mlp_h], dim=2))
        hidden = residual + gate.unsqueeze(1) * out
    if return_kv_img_from is not None:
        T = return_kv_img_from
        return hidden, k[:, :, T:], v[:, :, T:]
    return hidden


# ---------------------------------------------------------- dual stream (A) ---
def _dual_block_dense(block, x: torch.Tensor, ctx: torch.Tensor,
                      temb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      return_kv_img: bool = False):
    """Manual FluxTransformerBlock forward (diffusers 0.32.2 semantics):
    AdaLayerNormZero on both streams, joint attention with add_{q,k,v}_proj for
    the text stream (text-first concat), separate output projections and FFs.
    Optionally returns the image-token (k post-rope, v) for the dual KV cache."""
    T = ctx.shape[1]
    attn = block.attn

    n_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(x, emb=temb)
    n_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
        block.norm1_context(ctx, emb=temb)

    q_i = _heads(attn.to_q(n_x), attn.heads)
    k_i = _heads(attn.to_k(n_x), attn.heads)
    v_i = _heads(attn.to_v(n_x), attn.heads)
    if attn.norm_q is not None:
        q_i = attn.norm_q(q_i)
    if attn.norm_k is not None:
        k_i = attn.norm_k(k_i)
    q_c = _heads(attn.add_q_proj(n_c), attn.heads)
    k_c = _heads(attn.add_k_proj(n_c), attn.heads)
    v_c = _heads(attn.add_v_proj(n_c), attn.heads)
    if attn.norm_added_q is not None:
        q_c = attn.norm_added_q(q_c)
    if attn.norm_added_k is not None:
        k_c = attn.norm_added_k(k_c)

    q = _rope_apply(torch.cat([q_c, q_i], dim=2), cos, sin)
    k = _rope_apply(torch.cat([k_c, k_i], dim=2), cos, sin)
    v = torch.cat([v_c, v_i], dim=2)
    o = _unheads(F.scaled_dot_product_attention(q, k, v)).to(x.dtype)

    img_attn = attn.to_out[0](o[:, T:])
    ctx_attn = attn.to_add_out(o[:, :T])

    x = x + gate_msa.unsqueeze(1) * img_attn
    h = block.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    x = x + gate_mlp.unsqueeze(1) * block.ff(h)

    ctx = ctx + c_gate_msa.unsqueeze(1) * ctx_attn
    hc = block.norm2_context(ctx) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    ctx = ctx + c_gate_mlp.unsqueeze(1) * block.ff_context(hc)
    if ctx.dtype == torch.float16:
        ctx = ctx.clip(-65504, 65504)

    if return_kv_img:
        return ctx, x, k[:, :, T:], v[:, :, T:]
    return ctx, x


def _dual_block_sparse(block, x_hard: torch.Tensor, ctx: torch.Tensor,
                       hard_idx: torch.Tensor, q_pos: torch.Tensor, T: int,
                       temb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       cached_img_in: torch.Tensor | None = None,
                       kv_img_cache: tuple | None = None):
    """Sparse dual block: text stream fully fresh, image stream only for hard
    tokens. Image K/V come either from the anchor per-block image INPUT states
    with fresh hard rows scattered in (cached_img_in mode, exact under a fresh
    cache), or from the anchor KV cache with fresh hard K/V scattered
    (kv_img_cache mode, Lever B applied to the dual stream).
    Returns (ctx_out [B,T,D], x_hard_out [B,k,D])."""
    attn = block.attn
    k_h = hard_idx.shape[1]

    n_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
        block.norm1_context(ctx, emb=temb)
    q_c = _heads(attn.add_q_proj(n_c), attn.heads)
    k_c = _heads(attn.add_k_proj(n_c), attn.heads)
    v_c = _heads(attn.add_v_proj(n_c), attn.heads)
    if attn.norm_added_q is not None:
        q_c = attn.norm_added_q(q_c)
    if attn.norm_added_k is not None:
        k_c = attn.norm_added_k(k_c)

    if kv_img_cache is not None:
        # Lever B on dual: norm/qkv only on hard rows; easy K/V frozen at anchor
        n_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(x_hard, emb=temb)
        q_i = _heads(attn.to_q(n_h), attn.heads)
        k_new = _heads(attn.to_k(n_h), attn.heads)
        v_new = _heads(attn.to_v(n_h), attn.heads)
        if attn.norm_q is not None:
            q_i = attn.norm_q(q_i)
        if attn.norm_k is not None:
            k_new = attn.norm_k(k_new)
        cos_h = cos[q_pos[:, T:]].unsqueeze(1)
        sin_h = sin[q_pos[:, T:]].unsqueeze(1)
        k_new = _rope_apply(k_new, cos_h, sin_h)
        kc_img, vc_img = kv_img_cache
        idx = hard_idx[:, None, :, None].expand(-1, kc_img.shape[1], -1,
                                                kc_img.shape[-1])
        k_i = kc_img.scatter(2, idx, k_new.to(kc_img.dtype)).to(k_new.dtype)
        v_i = vc_img.scatter(2, idx, v_new.to(vc_img.dtype)).to(v_new.dtype)
        k_i_roped = True
    else:
        # exact mode: mixed full image set = anchor inputs with fresh hard rows
        mixed = _scatter_tokens(cached_img_in.to(x_hard.dtype), hard_idx, x_hard)
        n_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(mixed, emb=temb)
        n_h = _gather_tokens(n_x, hard_idx)
        q_i = _heads(attn.to_q(n_h), attn.heads)
        k_i = _heads(attn.to_k(n_x), attn.heads)
        v_i = _heads(attn.to_v(n_x), attn.heads)
        if attn.norm_q is not None:
            q_i = attn.norm_q(q_i)
        if attn.norm_k is not None:
            k_i = attn.norm_k(k_i)
        k_i_roped = False

    cos_q = cos[q_pos].unsqueeze(1)
    sin_q = sin[q_pos].unsqueeze(1)
    q = _rope_apply(torch.cat([q_c, q_i], dim=2), cos_q, sin_q)
    if k_i_roped:
        k_c_roped = _rope_apply(k_c, cos[q_pos[:, :T]].unsqueeze(1),
                                sin[q_pos[:, :T]].unsqueeze(1))
        K = torch.cat([k_c_roped, k_i], dim=2)
    else:
        K = _rope_apply(torch.cat([k_c, k_i], dim=2), cos, sin)
    V = torch.cat([v_c, v_i], dim=2)

    o = _unheads(F.scaled_dot_product_attention(q, K, V)).to(x_hard.dtype)
    img_attn = attn.to_out[0](o[:, T:])
    ctx_attn = attn.to_add_out(o[:, :T])

    x_hard = x_hard + gate_msa.unsqueeze(1) * img_attn
    h = block.norm2(x_hard) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    x_hard = x_hard + gate_mlp.unsqueeze(1) * block.ff(h)

    ctx = ctx + c_gate_msa.unsqueeze(1) * ctx_attn
    hc = block.norm2_context(ctx) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    ctx = ctx + c_gate_mlp.unsqueeze(1) * block.ff_context(hc)
    if ctx.dtype == torch.float16:
        ctx = ctx.clip(-65504, 65504)
    return ctx, x_hard


def _single_block_sparse_kv(
    block,
    q_fresh: torch.Tensor,        # [B, Sq, D]   fresh states: [text ; hard image]
    q_pos: torch.Tensor,          # [B, Sq]      absolute positions in the joint seq
    hard_idx: torch.Tensor,       # [B, k]       hard positions on the IMAGE grid
    T: int,
    temb: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    k_img_cache: torch.Tensor,    # [B, H, N, d] anchor image K (post-rope)
    v_img_cache: torch.Tensor,    # [B, H, N, d] anchor image V
) -> torch.Tensor:
    """Lever B: sparse block with the anchor K/V cache. Norm/K/V/MLP run only on
    the Sq fresh rows; easy image tokens contribute K/V frozen from the anchor
    (temb_a) — exact at the anchor step, a controlled temb-staleness
    approximation afterwards. Single-stream linear cost drops from
    D²(10·Sq + 2·S) to 12·Sq·D²."""
    residual = q_fresh
    with _rf("single_q_mlp"):
        q_norm, gate = block.norm(q_fresh, emb=temb)
        mlp_h = block.act_mlp(block.proj_mlp(q_norm))
        attn = block.attn
        q = _heads(attn.to_q(q_norm), attn.heads)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
    with _rf("single_kv_projection"):                 # Sq rows만 — Lever B 효과
        k_new = _heads(attn.to_k(q_norm), attn.heads)
        v_new = _heads(attn.to_v(q_norm), attn.heads)
        if attn.norm_k is not None:
            k_new = attn.norm_k(k_new)
    with _rf("single_attention"):
        cos_q = cos[q_pos].unsqueeze(1)
        sin_q = sin[q_pos].unsqueeze(1)
        q = _rope_apply(q, cos_q, sin_q)
        k_new = _rope_apply(k_new, cos_q, sin_q)      # text + hard, post-rope
    with _rf("single_kv_scatter"):
        # scatter fresh hard K/V into the anchor image cache (dim 2 = token)
        idx = hard_idx[:, None, :, None].expand(-1, k_img_cache.shape[1], -1,
                                                k_img_cache.shape[-1])
        k_img = k_img_cache.scatter(2, idx,
                                    k_new[:, :, T:].to(k_img_cache.dtype))
        v_img = v_img_cache.scatter(2, idx,
                                    v_new[:, :, T:].to(v_img_cache.dtype))
        K = torch.cat([k_new[:, :, :T], k_img.to(k_new.dtype)], dim=2)
        V = torch.cat([v_new[:, :, :T], v_img.to(v_new.dtype)], dim=2)
    with _rf("single_attention"):
        o = F.scaled_dot_product_attention(q, K, V)
        o = _unheads(o).to(q_fresh.dtype)
    with _rf("single_q_mlp"):
        out = block.proj_out(torch.cat([o, mlp_h], dim=2))
        return residual + gate.unsqueeze(1) * out


def _single_block_sparse(
    block,
    q_fresh: torch.Tensor,        # [B, Sq, D]   fresh states: [text ; hard image]
    ctx: torch.Tensor,            # [B, S,  D]   full ctx: fresh text/hard + cached easy
    q_pos: torch.Tensor,          # [B, Sq]      absolute positions of queries in seq
    temb: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,     # full-sequence rope tables [S, d]
) -> torch.Tensor:
    """Hard-query single-stream block: attention Sq x S, MLP on Sq only."""
    residual = q_fresh
    with _rf("single_q_mlp"):
        ctx_norm, gate = block.norm(ctx, emb=temb)          # per-token LN
        q_norm = _gather_tokens(ctx_norm, q_pos)            # == norm(q_fresh)
        mlp_h = block.act_mlp(block.proj_mlp(q_norm))
        attn = block.attn
        q = _heads(attn.to_q(q_norm), attn.heads)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
    with _rf("single_kv_projection"):                       # full-S K/V — floor의 원천
        k = _heads(attn.to_k(ctx_norm), attn.heads)
        v = _heads(attn.to_v(ctx_norm), attn.heads)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
    with _rf("single_attention"):
        cos_q = cos[q_pos].unsqueeze(1)                     # [B,1,Sq,d]
        sin_q = sin[q_pos].unsqueeze(1)
        q = _rope_apply(q, cos_q, sin_q)
        k = _rope_apply(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)
        o = _unheads(o).to(q_fresh.dtype)
    with _rf("single_q_mlp"):
        out = block.proj_out(torch.cat([o, mlp_h], dim=2))
        return residual + gate.unsqueeze(1) * out


# ------------------------------------------------------------------- runner ---
def estimate_transformer_macs(T: int, N: int, k: int, n_dual: int, n_single: int,
                              D: int, mlp_mult: int = 4,
                              kv_cached: bool = False,
                              dual_sparse: bool = False) -> dict:
    """Analytic MAC estimate for one forward (Fix 10). Sparse execution does NOT
    make everything sparse; per single-stream block only Q / MLP / proj_out and
    the query-side attention shrink to Sq = T + k, while context norm and K/V
    projections stay full-S, and the 19 dual blocks stay fully dense. This
    function makes that explicit so 'ran 30% of tokens' is never conflated with
    '30% of transformer compute'. Returns MACs for dense and sparse plus ratio."""
    S = T + N
    Sq = T + k
    m = mlp_mult
    # dual block (both streams dense in every mode): qkv+out 4·D² per token per
    # stream, MLP 2·m·D² per token, joint attention 2·S²·D
    dual = n_dual * ((4 + 2 * m) * (N + T) * D * D + 2 * S * S * D)
    # dual sparse (Lever A): text side full; image q/out/ff on k rows; image K/V
    # full N (exact mode) or k rows (with the dual KV cache); attn 2·Sq·S·D
    kv_img = 2 * k if kv_cached else 2 * N
    dual_sp = n_dual * (((2 + 2 * m) * k + kv_img + (4 + 2 * m) * T) * D * D
                        + 2 * Sq * S * D)
    # single block dense: q/k/v 3·S·D², mlp m·S·D², proj_out (1+m)·S·D², attn 2·S²·D
    single_dense = n_single * ((3 + m + 1 + m) * S * D * D + 2 * S * S * D)
    # single block sparse: q Sq + k/v (2·S dense / 2·Sq with the anchor KV cache),
    # mlp m·Sq, proj_out (1+m)·Sq, attn 2·Sq·S·D
    kv_rows = 2 * Sq if kv_cached else 2 * S
    single_sparse = n_single * (
        ((1 + m + 1 + m) * Sq + kv_rows) * D * D + 2 * Sq * S * D)
    final_dense = N * D * 64
    final_sparse = k * D * 64
    dense = dual + single_dense + final_dense
    sparse = (dual_sp if dual_sparse else dual) + single_sparse + final_sparse
    return {"dense_macs": dense, "sparse_macs": sparse,
            "mac_ratio": sparse / dense,
            "single_stream_share_dense": single_dense / dense}


@dataclass
class ForwardStats:
    mode: str                      # dense | anchor | sparse
    hard_ratio: float = 1.0
    single_attn_fraction: float = 1.0   # (Sq*S)/(S*S) averaged over single blocks
    single_linear_fraction: float = 1.0 # Sq/S (Q/MLP/proj_out side only; K/V stay full)
    est_transformer_mac_ratio: float = 1.0  # whole 57-block estimate incl. dense dual + full K/V


class FluxSparseRunner:
    """Owns the decomposed transformer forward. One instance per transformer."""

    def __init__(self, transformer):
        self.t = transformer

    # -------------------------------------------------------------- embeds ---
    def _embed(self, packed_model_input, prompt_embeds, pooled, timestep, guidance,
               img_ids, txt_ids):
        t = self.t
        x = t.x_embedder(packed_model_input)                       # [B, N, D]
        ctx = t.context_embedder(prompt_embeds)                    # [B, T, D]
        ts = timestep.to(x.dtype) * 1000
        if guidance is not None:
            g = guidance.to(x.dtype) * 1000
            temb = t.time_text_embed(ts, g, pooled)
        else:
            temb = t.time_text_embed(ts, pooled)
        ids = torch.cat([txt_ids, img_ids], dim=0)                 # [T+N, 3]
        rope = t.pos_embed(ids)
        cos, sin = (rope if isinstance(rope, tuple) else (rope[0], rope[1]))
        return x, ctx, temb, cos, sin

    def _dual_stream(self, x, ctx, temb, cos, sin):
        for block in self.t.transformer_blocks:
            out = block(hidden_states=x, encoder_hidden_states=ctx,
                        temb=temb, image_rotary_emb=(cos, sin))
            # diffusers returns (encoder_hidden_states, hidden_states)
            a, b = out
            if a.shape[1] == ctx.shape[1]:
                ctx, x = a, b
            else:
                x, ctx = a, b
        return x, ctx

    def _final(self, image_states, temb):
        h = self.t.norm_out(image_states, temb)
        return self.t.proj_out(h)                                   # [B, N, 64]

    # --------------------------------------------------------------- dense ---
    @torch.no_grad()
    def dense_forward(
        self,
        packed_model_input: torch.Tensor,      # [B, N, 384]
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        timestep: torch.Tensor,                # [B] in [0,1] (sigma-style, pipeline units)
        guidance: torch.Tensor | None,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        cache: FluxAnchorCache | None = None,  # pass to record an anchor
        step_index: int = -1,
        record_kv: bool = False,               # Lever B: also record image K/V per block
        record_dual: bool = False,             # Lever A: also record dual-block image inputs
    ):
        x, ctx, temb, cos, sin = self._embed(
            packed_model_input, prompt_embeds, pooled, timestep, guidance, img_ids, txt_ids)
        if cache is not None:
            cache.begin_anchor(timestep, step_index)
        if cache is not None and record_dual:
            # manual dual loop (== stock: Gate B0-dual) so inputs/K/V can be recorded
            for blk in self.t.transformer_blocks:
                with _rf("cache_record"):
                    cache.record_dual_input(x)
                if record_kv:
                    with _rf("dual_stream"):
                        ctx, x, k_img, v_img = _dual_block_dense(
                            blk, x, ctx, temb, cos, sin, return_kv_img=True)
                    with _rf("cache_record"):
                        cache.record_dual_kv(k_img, v_img)
                else:
                    with _rf("dual_stream"):
                        ctx, x = _dual_block_dense(blk, x, ctx, temb, cos, sin)
        else:
            with _rf("dual_stream"):
                x, ctx = self._dual_stream(x, ctx, temb, cos, sin)

        T = ctx.shape[1]
        cat = torch.cat([ctx, x], dim=1)
        if cache is not None:
            cache.entry_text_states = ctx.detach()
            cache.entry_image_states = x.detach()
        for block in self.t.single_transformer_blocks:
            if cache is not None:
                with _rf("cache_record"):
                    cache.record_single_input(cat[:, T:])
                if record_kv:
                    with _rf("single_kv_recompute"):
                        cat, k_img, v_img = _single_block_dense(
                            block, cat, temb, cos, sin, return_kv_img_from=T)
                    with _rf("cache_record"):
                        cache.record_single_kv(k_img, v_img)
                    continue
            with _rf("single_kv_recompute"):
                cat = _single_block_dense(block, cat, temb, cos, sin)

        with _rf("final_head"):
            v = self._final(cat[:, T:], temb)
        if cache is not None:
            cache.finish_anchor(v, ctx, x)
            cache.image_token_positions = torch.arange(
                T, T + x.shape[1], device=x.device)
        stats = ForwardStats(mode="anchor" if cache is not None else "dense")
        return v, stats

    # -------------------------------------------------------------- sparse ---
    @torch.no_grad()
    def teacache_forward(
        self,
        packed_model_input: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        tc: dict,                              # TeaCache state (아래 키 참조)
    ):
        """Faithful port of the OFFICIAL TeaCache4FLUX policy (ali-vilab/TeaCache,
        teacache_flux.py) to this runner. Preserved verbatim:
          - indicator: first dual block's AdaLN-modulated image input
            `transformer_blocks[0].norm1(x_embed, emb=temb)[0]`
          - accumulated rescaled rel-L1 with the official FLUX poly1d
            coefficients; skip while `accumulated < rel_l1_thresh`, else compute
            and reset
          - forced compute at the first and last step (cnt==0 / num_steps-1)
          - reuse: hidden-space residual added to the x_embedder output, then
            the final head (`norm_out`/`proj_out`) only
          - previous_modulated_input updated EVERY step (skip or not)
        Adaptations (documented in the paper): FLUX.1 *Fill* checkpoint
        (in_ch 384) and its guidance-distilled temb; rescaling coefficients
        transferred from FLUX-dev as the repo itself recommends for
        same-architecture models.
        tc keys: cnt, num_steps, rel_l1_thresh, accumulated, prev_mod,
        prev_residual. Returns (v, should_calc)."""
        import numpy as _np
        x, ctx, temb, cos, sin = self._embed(
            packed_model_input, prompt_embeds, pooled, timestep, guidance,
            img_ids, txt_ids)

        modulated_inp = self.t.transformer_blocks[0].norm1(x.clone(),
                                                           emb=temb.clone())[0]
        if tc["cnt"] == 0 or tc["cnt"] == tc["num_steps"] - 1:
            should_calc = True
            tc["accumulated"] = 0.0
        else:
            coeffs = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01,
                      -3.82021401e+00, 2.64230861e-01]          # 공식 FLUX 계수
            rescale = _np.poly1d(coeffs)
            rel = ((modulated_inp - tc["prev_mod"]).abs().mean()
                   / tc["prev_mod"].abs().mean()).cpu().item()
            tc["accumulated"] += float(rescale(rel))
            if tc["accumulated"] < tc["rel_l1_thresh"]:
                should_calc = False
            else:
                should_calc = True
                tc["accumulated"] = 0.0
        tc["prev_mod"] = modulated_inp
        tc["cnt"] += 1
        if tc["cnt"] == tc["num_steps"]:
            tc["cnt"] = 0

        T = ctx.shape[1]
        if not should_calc:
            x_out = x + tc["prev_residual"]
        else:
            ori = x.clone()
            x2, ctx2 = self._dual_stream(x, ctx, temb, cos, sin)
            cat = torch.cat([ctx2, x2], dim=1)
            for block in self.t.single_transformer_blocks:
                cat = _single_block_dense(block, cat, temb, cos, sin)
            x_out = cat[:, T:]
            tc["prev_residual"] = x_out - ori
        v = self._final(x_out, temb)
        return v, should_calc

    @torch.no_grad()
    def blockcache_forward(
        self,
        packed_model_input: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        bc: dict,
    ):
        """Mechanism-matched block-level temporal caching baseline
        (block-caching / FORA-style), 종합평가2 §2.1–2.2. 각 블록의 image-측
        residual을 이전 스텝에서 재사용한다.

        정책 (bc["policy"]):
          "delta_threshold": 블록 입력의 rel-L1 변화 < bc["thresh"]면 그
              블록의 이전 residual 재사용 (block-caching 계열).
          "fixed_period": bc["period"] 스텝마다 전체 계산, 그 외 스텝은 모든
              블록 residual 재사용 (FORA 계열).
        bc["mask_weight"]: [B, N] (0/1) 주어지면 rel-L1을 mask 토큰에서만
              계산 (mask-aware variant). None이면 global.
        첫 스텝(bc["cnt"]==0)과 마지막 스텝은 항상 전체 계산.
        상태: prev_in/prev_res (블록별 리스트, dual는 (ctx,x) 튜플 저장).
        Returns (v, n_blocks_computed).
        """
        x, ctx, temb, cos, sin = self._embed(
            packed_model_input, prompt_embeds, pooled, timestep, guidance,
            img_ids, txt_ids)
        T = ctx.shape[1]
        n_dual = len(self.t.transformer_blocks)
        n_single = len(self.t.single_transformer_blocks)
        n_blocks = n_dual + n_single
        first = bc["cnt"] == 0
        last = bc["cnt"] == bc["num_steps"] - 1
        force = first or last
        if bc["policy"] == "fixed_period" and not force:
            force_skip_all = (bc["cnt"] % max(bc.get("period", 2), 1)) != 0
        else:
            force_skip_all = False
        if first:
            bc["prev_in"] = [None] * n_blocks
            bc["prev_res"] = [None] * n_blocks
        mw = bc.get("mask_weight")                 # [B, N] or None

        def _rel(cur, prev):
            d = (cur - prev).abs()
            p = prev.abs()
            if mw is not None:
                m = mw.unsqueeze(-1)
                return ((d * m).sum() / (p * m).sum().clamp_min(1e-6)).item()
            return (d.mean() / p.mean().clamp_min(1e-6)).item()

        computed = 0
        for j, blk in enumerate(self.t.transformer_blocks):
            reuse = False
            if not force and bc["prev_in"][j] is not None:
                if force_skip_all or (
                        bc["policy"] == "delta_threshold"
                        and _rel(x, bc["prev_in"][j]) < bc["thresh"]):
                    reuse = True
            if reuse:
                ctx = ctx + bc["prev_res"][j][0]
                x = x + bc["prev_res"][j][1]
            else:
                bc["prev_in"][j] = x.detach()
                ci, xi = ctx, x
                ctx, x = _dual_block_dense(blk, x, ctx, temb, cos, sin)
                bc["prev_res"][j] = (
                    (ctx - ci).detach(), (x - xi).detach())
                computed += 1
        cat = torch.cat([ctx, x], dim=1)
        for i, blk in enumerate(self.t.single_transformer_blocks):
            j = n_dual + i
            reuse = False
            if not force and bc["prev_in"][j] is not None:
                if force_skip_all or (
                        bc["policy"] == "delta_threshold"
                        and _rel(cat[:, T:], bc["prev_in"][j]) < bc["thresh"]):
                    reuse = True
            if reuse:
                cat = cat + bc["prev_res"][j]
            else:
                bc["prev_in"][j] = cat[:, T:].detach()
                ci = cat
                cat = _single_block_dense(blk, cat, temb, cos, sin)
                bc["prev_res"][j] = (cat - ci).detach()
                computed += 1
        bc["cnt"] += 1
        if bc["cnt"] == bc["num_steps"]:
            bc["cnt"] = 0
        v = self._final(cat[:, T:], temb)
        return v, computed

    @torch.no_grad()
    def sparse_forward(
        self,
        packed_model_input: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        cache: FluxAnchorCache,
        hard_idx: torch.Tensor,                # [B, k] image-token indices
        kv_cache: bool = False,                # Lever B: anchor K/V for easy tokens
        dual_sparse: bool = False,             # Lever A: sparse image tokens in dual stream
    ):
        """Selective refresh. Returns (v_hard [B,k,64], stats).
        The sampler merges: v = scatter(cache.final_prediction, hard_idx, v_hard)."""
        assert not cache.is_empty(), "sparse_forward requires a recorded anchor"
        x, ctx, temb, cos, sin = self._embed(
            packed_model_input, prompt_embeds, pooled, timestep, guidance, img_ids, txt_ids)

        B, N, D = x.shape
        T = ctx.shape[1]
        k = hard_idx.shape[1]
        S = T + N
        dev = x.device
        text_pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
        hard_pos = hard_idx + T
        q_pos = torch.cat([text_pos, hard_pos], dim=1)              # [B, T+k]

        if dual_sparse:
            assert cache.dual_block_inputs or cache.dual_block_kv, \
                "dual_sparse requires dense_forward(..., record_dual=True) at the anchor"
            with _rf("sparse_overhead"):
                x_hard = _gather_tokens(x, hard_idx)
            for j, blk in enumerate(self.t.transformer_blocks):
                kv = cache.dual_block_kv[j] if (kv_cache and cache.dual_block_kv) else None
                cached_in = None if kv is not None else cache.dual_block_inputs[j]
                with _rf("dual_stream"):
                    ctx, x_hard = _dual_block_sparse(
                        blk, x_hard, ctx, hard_idx, q_pos, T, temb, cos, sin,
                        cached_img_in=cached_in, kv_img_cache=kv)
            q_text, q_hard = ctx, x_hard
        else:
            # dual stream dense — full image token set (PoC safety rule)
            with _rf("dual_stream"):
                x, ctx = self._dual_stream(x, ctx, temb, cos, sin)
            q_text = ctx
            q_hard = _gather_tokens(x, hard_idx)

        q_fresh = torch.cat([q_text, q_hard], dim=1)
        attn_frac, lin_frac = 0.0, 0.0
        if kv_cache:
            assert cache.single_block_kv, \
                "kv_cache=True requires dense_forward(..., record_kv=True) at the anchor"
        for j, block in enumerate(self.t.single_transformer_blocks):
            if kv_cache:
                k_img, v_img = cache.single_block_kv[j]
                with _rf("single_kv_cached"):
                    q_fresh = _single_block_sparse_kv(block, q_fresh, q_pos,
                                                      hard_idx, T, temb, cos,
                                                      sin, k_img, v_img)
                lin_frac += (T + k) / S              # K/V도 Sq행으로 축소
            else:
                cached_img = cache.single_block_inputs[j]           # [B, N, D] anchor depth-j
                with _rf("sparse_overhead"):
                    ctx_img = _scatter_tokens(cached_img.to(x.dtype), hard_idx,
                                              q_fresh[:, T:])
                    full_ctx = torch.cat([q_fresh[:, :T], ctx_img], dim=1)
                with _rf("single_kv_recompute"):
                    q_fresh = _single_block_sparse(block, q_fresh, full_ctx,
                                                   q_pos, temb, cos, sin)
                lin_frac += (T + k) / S
            attn_frac += (T + k) / S
        n_single = len(self.t.single_transformer_blocks)

        with _rf("final_head"):
            v_hard = self._final(q_fresh[:, T:], temb)              # [B, k, 64]
        macs = estimate_transformer_macs(
            T, N, k, len(self.t.transformer_blocks), n_single, D,
            kv_cached=kv_cache, dual_sparse=dual_sparse)
        stats = ForwardStats(
            mode="sparse",
            hard_ratio=k / N,
            single_attn_fraction=attn_frac / n_single,
            single_linear_fraction=lin_frac / n_single,
            est_transformer_mac_ratio=macs["mac_ratio"],
        )
        return v_hard, stats

    # ---------------------------------------------------------- accounting ---
    @staticmethod
    def merge_prediction(cache: FluxAnchorCache, hard_idx: torch.Tensor,
                         v_hard: torch.Tensor) -> torch.Tensor:
        """v_i = fresh for hard, anchor-cached for easy (plan Sec. 1 merge rule)."""
        return _scatter_tokens(cache.final_prediction.to(v_hard.dtype), hard_idx, v_hard)
