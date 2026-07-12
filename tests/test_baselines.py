"""baseline 경로가 sampler에서 정상 작동하는지 mock 검증 (GPU 불필요한 부분만)."""
import torch
from types import SimpleNamespace
import samplers.cached_flux_fill as m
from token_selectors.combo import select_hard_tokens


def test_uniform_baselines_budget():
    grid = SimpleNamespace(token_hw=(48, 48))
    g = torch.Generator("cpu").manual_seed(0)
    N = 48 * 48
    for meth in ("fora", "blockcache"):
        s = m._uniform_baseline_scores(meth, N, grid, g, torch.device("cpu"))
        assert s.shape == (1, N)
        hard, _, ract = select_hard_tokens(s, grid, 0.3, block=1)
        assert abs(ract - 0.3) < 0.02, (meth, ract)
        # mask-blind: 점수가 mask 정보를 안 씀 -> 결정적/균등
    # fora는 격자 주기적 -> 선택이 공간 전역에 퍼져야 함
    s_fora = m._uniform_baseline_scores("fora", N, grid, g, torch.device("cpu"))
    hard_fora, _, _ = select_hard_tokens(s_fora, grid, 0.3, block=1)
    hp = wp = 48
    rows = (hard_fora[0] // wp)
    assert rows.max() > hp * 0.7 and rows.min() < hp * 0.3, "fora가 공간 전역 분포여야"
    print("PASS uniform baselines (fora spread, budget matched)")


if __name__ == "__main__":
    test_uniform_baselines_budget()
