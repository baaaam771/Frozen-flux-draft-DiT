"""Gate B0-dual (GPU, requires FLUX weights): OFFICIAL FluxTransformerBlock
forward vs the manual `_dual_block_dense`, block-by-block on real weights.
Lever A(dual-stream sparsification)의 전제 gate — anchor 기록과 sparse 경로가
이 manual 재구현을 공유하므로, 이것이 0이어야 dual_sparse 결과를 신뢰할 수 있다.

    PYTHONPATH=. python tests/test_dual_block_equivalence.py --resolution 512 [--fp32]
"""
import argparse

import torch

from models.flux_fill_loader import load_flux_fill
from models.flux_sparse_transformer import _dual_block_dense, prepare_latent_image_ids
from utils.token_mapping import TokenGrid


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=0, help="0 = all 19 dual blocks")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    dtype = torch.float32 if a.fp32 else torch.bfloat16
    comps = load_flux_fill(dtype=dtype, keep_text_encoders=False)
    t, dev = comps.transformer, comps.device
    grid = TokenGrid(a.resolution, a.resolution).validate()
    hp, wp = grid.token_hw
    N, T = grid.num_image_tokens, a.text_len
    D = t.config.num_attention_heads * t.config.attention_head_dim

    torch.manual_seed(0)
    x = torch.randn(1, N, D, device=dev, dtype=dtype)
    ctx = torch.randn(1, T, D, device=dev, dtype=dtype)
    temb = torch.randn(1, D, device=dev, dtype=dtype)
    ids = torch.cat([torch.zeros(T, 3, device=dev, dtype=dtype),
                     prepare_latent_image_ids(hp, wp, dev, dtype)], dim=0)
    rope = t.pos_embed(ids)
    cos, sin = rope if isinstance(rope, tuple) else (rope[0], rope[1])

    blocks = t.transformer_blocks
    n = len(blocks) if a.blocks == 0 else min(a.blocks, len(blocks))
    tol = 1e-5 if a.fp32 else 3e-2
    worst = 0.0
    ok = True
    for j in range(n):
        blk = blocks[j]
        off = blk(hidden_states=x, encoder_hidden_states=ctx, temb=temb,
                  image_rotary_emb=(cos, sin))
        off_ctx, off_x = off
        man_ctx, man_x = _dual_block_dense(blk, x, ctx, temb, cos, sin)
        e = max((off_x.float() - man_x.float()).abs().max().item(),
                (off_ctx.float() - man_ctx.float()).abs().max().item())
        worst = max(worst, e)
        if e > tol:
            ok = False
            print(f"[Gate B0-dual] block {j:2d}: max|d| = {e:.3e}  FAIL")
    print(f"[Gate B0-dual] {n} blocks, worst max|d| = {worst:.3e}, tol = {tol}  "
          f"{'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
