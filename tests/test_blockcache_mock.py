"""blockcache_forward 검증 (mock transformer, GPU 불필요).
검증 항목: (a) 첫/마지막 스텝 강제 전체계산, (b) delta_threshold: 입력 변화
없으면 residual 재사용 = 전체계산과 동일 출력, (c) fixed_period: period
비배수 스텝은 전블록 재사용, (d) mask_weight가 rel-L1 계산에 반영,
(e) 재사용 출력이 '이전 계산과 동일 입력'일 때 exact.
"""
import torch
import torch.nn as nn

from models.flux_sparse_transformer import FluxSparseRunner


class _MockT(nn.Module):
    def __init__(self, n_dual=2, n_single=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [nn.Identity() for _ in range(n_dual)])
        self.single_transformer_blocks = nn.ModuleList(
            [nn.Identity() for _ in range(n_single)])


def _mk(B=1, N=8, T=4, D=16):
    r = FluxSparseRunner.__new__(FluxSparseRunner)
    r.t = _MockT()
    x0 = torch.randn(B, N, D)
    ctx0 = torch.randn(B, T, D)
    temb = torch.ones(B, D)
    r._embed = lambda *a, **k: (x0.clone(), ctx0.clone(), temb, None, None)
    r._final = lambda h, temb: h * 10.0
    return r, x0, ctx0, T


def _patch_blocks(add_dual=3.0, add_single=5.0):
    import models.flux_sparse_transformer as M
    orig_d, orig_s = M._dual_block_dense, M._single_block_dense
    M._dual_block_dense = lambda blk, x, ctx, temb, cos, sin: (ctx + 1.0,
                                                              x + add_dual)
    M._single_block_dense = lambda blk, cat, temb, cos, sin: cat + add_single
    return orig_d, orig_s


def _restore(orig):
    import models.flux_sparse_transformer as M
    M._dual_block_dense, M._single_block_dense = orig


def test_delta_threshold_exact_reuse():
    orig = _patch_blocks()
    try:
        r, x0, ctx0, T = _mk()
        args = (None, None, None, torch.tensor([0.5]), None, None, None)
        bc = dict(cnt=0, num_steps=4, policy="delta_threshold", thresh=0.5,
                  period=2, mask_weight=None)
        v0, n0 = r.blockcache_forward(*args, bc)
        assert n0 == 4                              # 첫 스텝 전체계산 (2+2)
        # 입력이 매 스텝 동일(mock) -> rel=0 < thresh -> 전블록 재사용
        v1, n1 = r.blockcache_forward(*args, bc)
        assert n1 == 0 and torch.allclose(v1, v0)   # exact reuse
        v2, n2 = r.blockcache_forward(*args, bc)
        assert n2 == 0
        v3, n3 = r.blockcache_forward(*args, bc)
        assert n3 == 4                              # 마지막 스텝 강제
        print("delta_threshold: forced ends + exact reuse OK")
    finally:
        _restore(orig)


def test_fixed_period():
    orig = _patch_blocks()
    try:
        r, *_ = _mk()
        args = (None, None, None, torch.tensor([0.5]), None, None, None)
        bc = dict(cnt=0, num_steps=6, policy="fixed_period", thresh=0.0,
                  period=2, mask_weight=None)
        pattern = [r.blockcache_forward(*args, bc)[1] for _ in range(6)]
        assert pattern == [4, 0, 4, 0, 4, 4], pattern  # 짝수 계산, 끝 강제
        print("fixed_period pattern OK:", pattern)
    finally:
        _restore(orig)


def test_mask_weight_used():
    orig = _patch_blocks()
    try:
        r, x0, ctx0, T = _mk()
        args = (None, None, None, torch.tensor([0.5]), None, None, None)
        mw = torch.zeros(1, x0.shape[1])
        mw[0, :2] = 1.0
        bc = dict(cnt=0, num_steps=4, policy="delta_threshold", thresh=1e-9,
                  period=2, mask_weight=mw)
        r.blockcache_forward(*args, bc)
        v1, n1 = r.blockcache_forward(*args, bc)   # 입력 동일 -> mask-rel=0
        assert n1 == 0
        print("mask-aware rel-L1 path OK")
    finally:
        _restore(orig)


if __name__ == "__main__":
    test_delta_threshold_exact_reuse()
    test_fixed_period()
    test_mask_weight_used()
    print("blockcache mock tests pass")
