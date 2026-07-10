"""Reproduce Gate B2 through the ACTUAL FluxSparseRunner glue on a mock
transformer implementing the full interface (x_embedder, context_embedder,
time_text_embed, pos_embed, dual blocks, single blocks, norm_out, proj_out).

The earlier mock test called _single_block_dense/_single_block_sparse directly
and passed; Gate B2 on the real model fails — so the suspect is the runner
glue (embed / dual stream / q_pos / cache recording), which this test covers.
"""
import math
import types

import torch
import torch.nn as nn

from models.flux_cache import FluxAnchorCache
from models.flux_sparse_transformer import FluxSparseRunner, prepare_latent_image_ids
from tests.test_sparse_math_mock import MockSingleBlock, _rope_tables


class MockDualBlock(nn.Module):
    """Joint attention over [text;image] like FluxTransformerBlock, returning
    (encoder_hidden_states, hidden_states)."""

    def __init__(self, d=64, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, hidden_states=None, encoder_hidden_states=None,
                temb=None, image_rotary_emb=None):
        x, ctx = hidden_states, encoder_hidden_states
        T = ctx.shape[1]
        h = torch.cat([ctx, x], dim=1)
        n = self.ln(h + temb[:, None])
        B, S, D = n.shape
        H = self.heads
        q = self.q(n).view(B, S, H, D // H).transpose(1, 2)
        k = self.k(n).view(B, S, H, D // H).transpose(1, 2)
        v = self.v(n).view(B, S, H, D // H).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, S, D)
        h = h + self.o(o)
        return h[:, :T], h[:, T:]


class MockTimeTextEmbed(nn.Module):
    def __init__(self, d, pooled_dim):
        super().__init__()
        self.t = nn.Linear(1, d)
        self.g = nn.Linear(1, d)
        self.p = nn.Linear(pooled_dim, d)

    def forward(self, ts, guidance, pooled):
        return self.t(ts[:, None].to(self.t.weight.dtype)).to(pooled.dtype) \
             + self.g(guidance[:, None].to(self.g.weight.dtype)).to(pooled.dtype) + self.p(pooled)


class MockPosEmbed(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, ids):
        # position = row-major composite of the 3 id axes (like Flux h/w axes)
        pos = (ids[:, 0] * 10_000 + ids[:, 1] * 100 + ids[:, 2]).float()
        freqs = torch.exp(-math.log(10_000)
                          * torch.arange(self.head_dim // 2) / (self.head_dim // 2))
        ang = pos[:, None] * freqs[None]
        cos = torch.repeat_interleave(ang.cos(), 2, dim=-1)
        sin = torch.repeat_interleave(ang.sin(), 2, dim=-1)
        return cos, sin


class MockNormOut(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mod = nn.Linear(d, d)

    def forward(self, x, temb):
        return self.ln(x) * (1 + self.mod(temb)[:, None])


class MockFluxTransformer(nn.Module):
    def __init__(self, d=64, heads=4, n_dual=3, n_single=6,
                 joint_dim=48, pooled_dim=24, out_ch=16, in_ch=32):
        super().__init__()
        self.x_embedder = nn.Linear(in_ch, d)
        self.context_embedder = nn.Linear(joint_dim, d)
        self.time_text_embed = MockTimeTextEmbed(d, pooled_dim)
        self.pos_embed = MockPosEmbed(d // heads)
        self.transformer_blocks = nn.ModuleList(MockDualBlock(d, heads) for _ in range(n_dual))
        self.single_transformer_blocks = nn.ModuleList(MockSingleBlock(d, heads) for _ in range(n_single))
        self.norm_out = MockNormOut(d)
        self.proj_out = nn.Linear(d, out_ch)
        self.config = types.SimpleNamespace(guidance_embeds=True)


def test_runner_glue_fresh_cache_exactness():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    B, T, hp, wp = 1, 7, 4, 6
    N = hp * wp
    t = MockFluxTransformer(in_ch=32, joint_dim=48, pooled_dim=24)
    runner = FluxSparseRunner(t)

    x = torch.randn(B, N, 32)
    pe = torch.randn(B, T, 48)
    po = torch.randn(B, 24)
    ts = torch.full((B,), 0.5)
    gd = torch.full((B,), 30.0)
    img_ids = prepare_latent_image_ids(hp, wp, "cpu", torch.float64)
    txt_ids = torch.zeros(T, 3)

    cache = FluxAnchorCache()
    v_dense, _ = runner.dense_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                                      cache=cache, step_index=0)

    for ratio in (0.15, 0.5, 1.0):
        k = max(1, int(ratio * N))
        hard = torch.sort(torch.randperm(N)[:k]).values[None]
        v_hard, _ = runner.sparse_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                                          cache, hard)
        ref = torch.gather(v_dense, 1, hard.unsqueeze(-1).expand(-1, -1, v_dense.shape[-1]))
        err = (v_hard - ref).abs().max().item()
        print(f"runner-glue exactness ratio={ratio}: max|dv| = {err:.3e}")
        assert err < 1e-10, f"runner glue diverges at ratio {ratio}: {err}"
    torch.set_default_dtype(torch.float32)


if __name__ == "__main__":
    test_runner_glue_fresh_cache_exactness()
    print("PASS runner-glue exactness")
