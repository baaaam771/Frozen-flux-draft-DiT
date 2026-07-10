"""tools.bisect_gate_b2 — Gate B2 실패 원인 특정 프로브 (GPU, FLUX weights 필요).

B0(실제 block dense-equiv)과 B1(전체 dense-equiv)은 0 오차 통과, mock에서는
block 수식과 runner glue 모두 exact — 남은 유일한 미검증 조합은 "실제 모듈 +
sparse 경로"다. 이 프로브는 다음 단계로 원인을 분리한다:

  Phase 0  환경/텐서 진단: rope shape·dtype, sdpa backend 가용성
  Phase 1  실제 block0에서 _single_block_sparse(q_pos=arange(S), ctx=입력 전체)
           vs _single_block_dense — 같아야 함. 다르면 sparse 함수 내부 op별
           (q_norm/mlp/q/k/v/rope/attn/out) max|d| 덤프.
  Phase 2  Phase 1과 동일하되 sdpa를 MATH backend로 강제 — Phase 1이 실패하고
           Phase 2가 통과하면 cross-length sdpa kernel 문제로 확정.
  Phase 3  부분 ratio에서 runner sparse 루프를 수동 재현, dense가 기록한
           block별 입·출력과 비교해 최초 발산 블록과 그 내부 op를 특정.

    PYTHONPATH=. python -m tools.bisect_gate_b2 --resolution 512
"""
from __future__ import annotations

import argparse
import contextlib

import torch
import torch.nn.functional as F

from models.flux_cache import FluxAnchorCache
from models.flux_fill_loader import load_flux_fill
from models.flux_sparse_transformer import (FluxSparseRunner, _gather_tokens,
                                            _heads, _rope_apply, _scatter_tokens,
                                            _single_block_dense,
                                            _single_block_sparse, _unheads,
                                            prepare_latent_image_ids)
from utils.token_mapping import TokenGrid


def _md(a, b):
    return (a.float() - b.float()).abs().max().item()


@contextlib.contextmanager
def _sdpa_math():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    except Exception:
        with torch.backends.cuda.sdp_kernel(enable_flash=False,
                                            enable_mem_efficient=False,
                                            enable_math=True):
            yield


def _sparse_block_probe(block, x_in, temb, cos, sin, q_pos, ctx, tag):
    """Run dense and sparse block side by side, dumping per-op divergence."""
    dense_out = _single_block_dense(block, x_in, temb, cos, sin)
    sparse_out = _single_block_sparse(block, _gather_tokens(x_in, q_pos), ctx,
                                      q_pos, temb, cos, sin)
    ref = _gather_tokens(dense_out, q_pos)
    err = _md(sparse_out, ref)
    print(f"[{tag}] block out max|d| = {err:.3e}")
    if err < 1e-2:
        return True

    # ---- op-by-op dump ----
    normed_d, gate = block.norm(x_in, emb=temb)
    ctx_norm, gate_s = block.norm(ctx, emb=temb)
    q_norm = _gather_tokens(ctx_norm, q_pos)
    print("   norm(q rows)      ", f"{_md(q_norm, _gather_tokens(normed_d, q_pos)):.3e}")
    print("   gate               ", f"{_md(gate_s, gate):.3e}")
    mlp_d = block.act_mlp(block.proj_mlp(normed_d))
    mlp_s = block.act_mlp(block.proj_mlp(q_norm))
    print("   mlp(q rows)       ", f"{_md(mlp_s, _gather_tokens(mlp_d, q_pos)):.3e}")

    attn = block.attn
    qd = _heads(attn.to_q(normed_d), attn.heads)
    kd = _heads(attn.to_k(normed_d), attn.heads)
    vd = _heads(attn.to_v(normed_d), attn.heads)
    qs = _heads(attn.to_q(q_norm), attn.heads)
    ks = _heads(attn.to_k(ctx_norm), attn.heads)
    vs = _heads(attn.to_v(ctx_norm), attn.heads)
    if attn.norm_q is not None:
        qd, qs = attn.norm_q(qd), attn.norm_q(qs)
    if attn.norm_k is not None:
        kd, ks = attn.norm_k(kd), attn.norm_k(ks)
    print("   K (pre-rope)      ", f"{_md(ks, kd):.3e}")
    print("   Q rows (pre-rope) ", f"{_md(qs, qd[:, :, q_pos[0]]):.3e}")
    qd_r = _rope_apply(qd, cos, sin)
    kd_r = _rope_apply(kd, cos, sin)
    qs_r = _rope_apply(qs, cos[q_pos].unsqueeze(1), sin[q_pos].unsqueeze(1))
    ks_r = _rope_apply(ks, cos, sin)
    print("   Q rows (post-rope)", f"{_md(qs_r, qd_r[:, :, q_pos[0]]):.3e}")
    print("   K     (post-rope) ", f"{_md(ks_r, kd_r):.3e}")
    od = _unheads(F.scaled_dot_product_attention(qd_r, kd_r, vd))
    os_ = _unheads(F.scaled_dot_product_attention(qs_r, ks_r, vs))
    print("   attn out rows     ", f"{_md(os_, od[:, q_pos[0]]):.3e}")
    with _sdpa_math():
        om = _unheads(F.scaled_dot_product_attention(qs_r, ks_r, vs))
    print("   attn out rows (MATH backend vs dense rows)",
          f"{_md(om, od[:, q_pos[0]]):.3e}")
    return False


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--ratio", type=float, default=0.3)
    a = ap.parse_args()

    comps = load_flux_fill(keep_text_encoders=False)
    t, dev, dtype = comps.transformer, comps.device, comps.dtype
    runner = FluxSparseRunner(t)
    grid = TokenGrid(a.resolution, a.resolution).validate()
    hp, wp = grid.token_hw
    N, T = grid.num_image_tokens, a.text_len
    S = T + N

    torch.manual_seed(0)
    x_pk = torch.randn(1, N, 384, device=dev, dtype=dtype)
    pe = torch.randn(1, T, t.config.joint_attention_dim, device=dev, dtype=dtype)
    po = torch.randn(1, t.config.pooled_projection_dim, device=dev, dtype=dtype)
    ts = torch.full((1,), 0.5, device=dev, dtype=dtype)
    gd = torch.full((1,), 30.0, device=dev, dtype=torch.float32) \
        if t.config.guidance_embeds else None
    img_ids = prepare_latent_image_ids(hp, wp, dev, dtype)
    txt_ids = torch.zeros(T, 3, device=dev, dtype=dtype)

    # ---------------- Phase 0: environment ----------------
    x, ctx, temb, cos, sin = runner._embed(x_pk, pe, po, ts, gd, img_ids, txt_ids)
    print(f"[P0] torch {torch.__version__} | device {torch.cuda.get_device_name(0)}")
    print(f"[P0] rope cos shape={tuple(cos.shape)} dtype={cos.dtype} dev={cos.device}")
    print(f"[P0] flash={torch.backends.cuda.flash_sdp_enabled()} "
          f"mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()} "
          f"math={torch.backends.cuda.math_sdp_enabled()}")

    x, ctx = runner._dual_stream(x, ctx, temb, cos, sin)
    cat0 = torch.cat([ctx, x], dim=1)                      # single-stack entry
    blk0 = t.single_transformer_blocks[0]

    # ---------------- Phase 1: real block0, full-seq sparse fn ----------------
    q_all = torch.arange(S, device=dev).unsqueeze(0)
    ok_full = _sparse_block_probe(blk0, cat0, temb, cos, sin, q_all, cat0,
                                  "P1 full-seq")

    # ---------------- Phase 1b: real block0, subset queries -------------------
    k = max(1, int(a.ratio * N))
    hard = torch.sort(torch.randperm(N, device=dev)[:k]).values[None]
    q_pos = torch.cat([torch.arange(T, device=dev)[None], hard + T], dim=1)
    ok_sub = _sparse_block_probe(blk0, cat0, temb, cos, sin, q_pos, cat0,
                                 "P1b subset-q")

    # ---------------- Phase 2: subset queries, MATH backend -------------------
    with _sdpa_math():
        ok_math = _sparse_block_probe(blk0, cat0, temb, cos, sin, q_pos, cat0,
                                      "P2 subset-q+MATH")

    # ---------------- Phase 3: per-block bisect through the stack -------------
    if ok_full and ok_sub:
        print("[P3] block fn exact — bisecting the runner loop")
        cache = FluxAnchorCache()
        v_dense, _ = runner.dense_forward(x_pk, pe, po, ts, gd, img_ids, txt_ids,
                                          cache=cache, step_index=0)
        # dense per-block outputs for reference
        cat = cat0.clone()
        outs = []
        for blk in t.single_transformer_blocks:
            cat = _single_block_dense(blk, cat, temb, cos, sin)
            outs.append(cat)
        q_fresh = torch.cat([ctx, _gather_tokens(x, hard)], dim=1)
        for j, blk in enumerate(t.single_transformer_blocks):
            cached_img = cache.single_block_inputs[j]
            ctx_img = _scatter_tokens(cached_img.to(x.dtype), hard, q_fresh[:, T:])
            full_ctx = torch.cat([q_fresh[:, :T], ctx_img], dim=1)
            in_err = _md(full_ctx, outs[j - 1] if j else cat0)
            q_fresh = _single_block_sparse(blk, q_fresh, full_ctx, q_pos, temb, cos, sin)
            out_err = _md(q_fresh, _gather_tokens(outs[j], q_pos))
            flag = "" if out_err < 5e-2 else "   <-- FIRST DIVERGENCE" 
            print(f"[P3] block {j:2d}: ctx_in max|d|={in_err:.3e}  "
                  f"q_out max|d|={out_err:.3e}{flag}")
            if out_err >= 5e-2:
                _sparse_block_probe(blk, outs[j - 1] if j else cat0, temb, cos, sin,
                                    q_pos, full_ctx, f"P3 block{j} drill-down")
                break

    print("\n해석: P1/P1b 실패 + P2(MATH) 통과 => cross-length sdpa kernel 문제 "
          "(수정: sparse attention에 backend 강제/우회). P1도 P2도 실패 => "
          "_single_block_sparse의 op-level 덤프에서 최초 발산 op 확인. "
          "P1/P1b/P2 모두 통과 => P3의 최초 발산 블록 로그 확인.")


if __name__ == "__main__":
    main()
