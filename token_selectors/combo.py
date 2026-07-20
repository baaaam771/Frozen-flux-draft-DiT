"""selectors.combo — s_i = α·M_i + β·B_i + γ·RankNorm(F_i) + δ·RankNorm(Δ_i) + η·RankNorm(A^D_i)

The One-Verifier composition rule instantiated on Ω = FLUX packed image-token
grid. Bounded priors (M, B ∈ [0,1]) enter raw; unbounded signals (F, Δ, A^D)
are rank-normalized per image so their scales cannot dominate — the same
RankNorm treatment as the SR frequency-mixing ablation.

Named presets cover the full Stage-5 ablation, plus `random` and `oracle`
baselines. `oracle_score` ranks by the *true* current-vs-anchor target change
(requires an extra dense pass; excluded from compute accounting, as in DACE).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from utils.token_mapping import (TokenGrid, block_hard_easy_split, blockify_scores,
                                 hard_easy_split)


def rank_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-image rank normalization to [0, 1]. x: [B, N]."""
    B, N = x.shape
    order = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
    return order / max(N - 1, 1)


@dataclass
class ComboWeights:
    alpha: float = 1.0   # mask M
    beta: float = 0.5    # boundary B
    gamma: float = 0.0   # frequency F (rank-normed)
    delta: float = 0.0   # anchor delta (rank-normed)
    eta: float = 0.0     # draft disagreement (rank-normed)


# Stage-5 ablation presets (plan Sec. 12) --------------------------------------
PRESETS: dict[str, ComboWeights] = {
    "mask":            ComboWeights(1.0, 0.0, 0.0, 0.0, 0.0),
    "mask_boundary":   ComboWeights(1.0, 0.5, 0.0, 0.0, 0.0),
    "mask_delta":      ComboWeights(1.0, 0.0, 0.0, 1.0, 0.0),
    "mask_frequency":  ComboWeights(1.0, 0.0, 1.0, 0.0, 0.0),
    "mbd":             ComboWeights(1.0, 0.5, 0.0, 1.0, 0.0),   # M+B+Δ
    "mbfd":            ComboWeights(1.0, 0.5, 1.0, 1.0, 0.0),   # M+B+F+Δ  (핵심 비교)
    "delta_only":      ComboWeights(0.0, 0.0, 0.0, 1.0, 0.0),   # generic pruning
    "mbd_draft":  ComboWeights(alpha=1.0, beta=0.5, gamma=0.0, delta=1.0, eta=1.0),
    "mbfd_draft":      ComboWeights(1.0, 0.5, 1.0, 1.0, 1.0),
}


def combo_score(
    w: ComboWeights,
    mask: torch.Tensor | None = None,
    boundary: torch.Tensor | None = None,
    frequency: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
    draft: torch.Tensor | None = None,
) -> torch.Tensor:
    """All inputs [B, N] or None; returns [B, N]."""
    s = None

    def add(acc, coeff, term, ranked):
        if coeff == 0.0 or term is None:
            return acc
        t = rank_norm(term) if ranked else term
        return coeff * t if acc is None else acc + coeff * t

    s = add(s, w.alpha, mask, ranked=False)
    s = add(s, w.beta, boundary, ranked=False)
    s = add(s, w.gamma, frequency, ranked=True)
    s = add(s, w.delta, delta, ranked=True)
    s = add(s, w.eta, draft, ranked=True)
    assert s is not None, "combo_score received no active terms"
    return s


def random_score(B: int, N: int, generator: torch.Generator | None = None, device=None) -> torch.Tensor:
    return torch.rand(B, N, generator=generator, device=device)


def oracle_score(v_current_dense: torch.Tensor, v_anchor: torch.Tensor) -> torch.Tensor:
    """True current-vs-anchor target change per token — selection upper bound."""
    return (v_current_dense.float() - v_anchor.float()).pow(2).mean(dim=-1)


def select_hard_tokens(
    scores: torch.Tensor,
    grid: TokenGrid,
    ratio: float,
    block: int = 1,
):
    """Score -> (hard_idx, easy_idx, actual_ratio). block > 1 performs TRUE
    block-level Top-K (Stage 7): whole contiguous (block x block) windows are
    selected, and actual_ratio = kb*block^2 / N is what must be reported."""
    return block_hard_easy_split(scores, grid, ratio, block)
