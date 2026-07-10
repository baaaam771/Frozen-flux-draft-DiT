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


class MockAdaLNZero(nn.Module):
    """AdaLayerNormZero: returns (modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp)."""
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 6 * d)
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        sm, cm, gm, s2, c2, g2 = self.lin(torch.nn.functional.silu(emb)).chunk(6, -1)
        return self.norm(x) * (1 + cm[:, None]) + sm[:, None], gm, s2, c2, g2


class MockDualAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        from tests.test_sparse_math_mock import MockRMSNorm
        self.heads = heads
        self.to_q = nn.Linear(d, d); self.to_k = nn.Linear(d, d); self.to_v = nn.Linear(d, d)
        self.add_q_proj = nn.Linear(d, d); self.add_k_proj = nn.Linear(d, d)
        self.add_v_proj = nn.Linear(d, d)
        hd = d // heads
        self.norm_q = MockRMSNorm(hd); self.norm_k = MockRMSNorm(hd)
        self.norm_added_q = MockRMSNorm(hd); self.norm_added_k = MockRMSNorm(hd)
        self.to_out = nn.ModuleList([nn.Linear(d, d)])
        self.to_add_out = nn.Linear(d, d)


class MockDualBlock(nn.Module):
    """Full FluxTransformerBlock-shaped mock; stock forward delegates to the
    manual _dual_block_dense so mock stock == manual by construction (the real
    stock-vs-manual equivalence is Gate B0-dual's job)."""
    def __init__(self, d=64, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = MockAdaLNZero(d)
        self.norm1_context = MockAdaLNZero(d)
        self.attn = MockDualAttn(d, heads)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm2_context = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(d, d * mlp_ratio), nn.GELU(approximate="tanh"),
                                nn.Linear(d * mlp_ratio, d))
        self.ff_context = nn.Sequential(nn.Linear(d, d * mlp_ratio),
                                        nn.GELU(approximate="tanh"),
                                        nn.Linear(d * mlp_ratio, d))

    def forward(self, hidden_states=None, encoder_hidden_states=None,
                temb=None, image_rotary_emb=None):
        from models.flux_sparse_transformer import _dual_block_dense
        cos, sin = image_rotary_emb
        return _dual_block_dense(self, hidden_states, encoder_hidden_states,
                                 temb, cos, sin)


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


def test_kv_cache_exact_at_anchor_step():
    """Lever B: anchor step에서는 temb가 같으므로 kv_cache 경로도 EXACT여야 하고,
    (mock에서) 다른 temb로 흉내낸 later step에서는 exact가 깨져야 한다(근사임 확인)."""
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    B, T, hp, wp = 1, 7, 4, 6
    N = hp * wp
    t = MockFluxTransformer(in_ch=32, joint_dim=48, pooled_dim=24)
    runner = FluxSparseRunner(t)
    x = torch.randn(B, N, 32); pe = torch.randn(B, T, 48); po = torch.randn(B, 24)
    ts = torch.full((B,), 0.5); gd = torch.full((B,), 30.0)
    img_ids = prepare_latent_image_ids(hp, wp, "cpu", torch.float64)
    txt_ids = torch.zeros(T, 3)

    cache = FluxAnchorCache()
    v_dense, _ = runner.dense_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                                      cache=cache, step_index=0, record_kv=True)
    assert len(cache.single_block_kv) == len(t.single_transformer_blocks)

    hard = torch.sort(torch.randperm(N)[:8]).values[None]
    v_hard, st = runner.sparse_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                                       cache, hard, kv_cache=True)
    ref = torch.gather(v_dense, 1, hard.unsqueeze(-1).expand(-1, -1, v_dense.shape[-1]))
    err = (v_hard - ref).abs().max().item()
    print(f"kv-cache anchor-step exactness: max|dv| = {err:.3e}")
    assert err < 1e-10
    assert st.est_transformer_mac_ratio < 1.0

    # later step 흉내: 같은 z, 다른 timestep -> kv 경로는 non-kv 경로와 달라야 함
    ts2 = torch.full((B,), 0.4)
    v_kv, _ = runner.sparse_forward(x, pe, po, ts2, gd, img_ids, txt_ids,
                                    cache, hard, kv_cache=True)
    v_nk, _ = runner.sparse_forward(x, pe, po, ts2, gd, img_ids, txt_ids,
                                    cache, hard, kv_cache=False)
    d = (v_kv - v_nk).abs().max().item()
    print(f"kv staleness at different temb: max|d| = {d:.3e} (>0 expected)")
    assert d > 1e-8
    torch.set_default_dtype(torch.float32)


def test_dual_sparse_fresh_cache_exactness():
    """Lever A: dual_sparse (exact 모드와 kv 모드 모두) fresh cache에서 dense와 일치."""
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)
    B, T, hp, wp = 1, 7, 4, 6
    N = hp * wp
    t = MockFluxTransformer(in_ch=32, joint_dim=48, pooled_dim=24)
    runner = FluxSparseRunner(t)
    x = torch.randn(B, N, 32); pe = torch.randn(B, T, 48); po = torch.randn(B, 24)
    ts = torch.full((B,), 0.5); gd = torch.full((B,), 30.0)
    img_ids = prepare_latent_image_ids(hp, wp, "cpu", torch.float64)
    txt_ids = torch.zeros(T, 3)

    cache = FluxAnchorCache()
    v_dense, _ = runner.dense_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                                      cache=cache, step_index=0,
                                      record_kv=True, record_dual=True)
    assert len(cache.dual_block_inputs) == len(t.transformer_blocks)
    assert len(cache.dual_block_kv) == len(t.transformer_blocks)

    for ratio in (0.2, 0.6):
        k = max(1, int(ratio * N))
        hard = torch.sort(torch.randperm(N)[:k]).values[None]
        ref = torch.gather(v_dense, 1, hard.unsqueeze(-1).expand(-1, -1, v_dense.shape[-1]))
        for kv in (False, True):
            v_hard, st = runner.sparse_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                                               cache, hard, kv_cache=kv,
                                               dual_sparse=True)
            err = (v_hard - ref).abs().max().item()
            print(f"dual_sparse ratio={ratio} kv={kv}: max|dv| = {err:.3e} "
                  f"(est MAC {st.est_transformer_mac_ratio:.3f})")
            assert err < 1e-9
    torch.set_default_dtype(torch.float32)
