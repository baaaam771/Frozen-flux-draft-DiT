"""mask-blind control baseline 검증 (리뷰 fix.docx 반영)."""
import torch
from types import SimpleNamespace
import samplers.cached_flux_fill as m
from token_selectors.combo import select_hard_tokens


def test_unknown_control_raises():
    grid = SimpleNamespace(token_hw=(16, 16))
    g = torch.Generator("cpu").manual_seed(0)
    for bad in ("fora", "blockcache", "teacache", ""):
        try:
            m._mask_blind_scores(bad, 256, grid, g, torch.device("cpu"))
            assert False, f"{bad} should raise"
        except ValueError:
            pass
    print("PASS unknown control raises (no silent misclassification)")


def test_uniform_grid_spatially_even():
    grid = SimpleNamespace(token_hw=(48, 48))
    g = torch.Generator("cpu").manual_seed(0)
    s = m._mask_blind_scores("uniform_grid", 48 * 48, grid, g, torch.device("cpu"))
    hard, _, ract = select_hard_tokens(s, grid, 0.3, block=1)
    assert abs(ract - 0.3) < 0.02
    rc, cc = hard[0] // 48, hard[0] % 48
    cells = {(int(r) // 12, int(c) // 12) for r, c in zip(rc, cc)}
    assert len(cells) == 16, f"only {len(cells)}/16 cells covered"
    print("PASS uniform_grid covers all coarse cells at r=0.3")


def test_contiguous_block_connected_and_moving():
    grid = SimpleNamespace(token_hw=(48, 48))
    g = torch.Generator("cpu").manual_seed(0)
    centers = []
    for seed in (0, 500, 999):
        s = m._mask_blind_scores("contiguous_block", 48 * 48, grid, g,
                                 torch.device("cpu"), seed=seed)
        hard, _, ract = select_hard_tokens(s, grid, 0.3, block=1)
        assert abs(ract - 0.3) < 0.02
        rc, cc = hard[0] // 48, hard[0] % 48
        # 연속: 선택 span이 전체(48)보다 작아야
        assert (rc.max() - rc.min()) < 46 and (cc.max() - cc.min()) < 46
        centers.append((int(rc.float().median()), int(cc.float().median())))
    assert len(set(centers)) > 1, "seed마다 위치가 이동해야"
    print(f"PASS contiguous_block connected + moves across seeds {centers}")


if __name__ == "__main__":
    test_unknown_control_raises()
    test_uniform_grid_spatially_even()
    test_contiguous_block_connected_and_moving()
