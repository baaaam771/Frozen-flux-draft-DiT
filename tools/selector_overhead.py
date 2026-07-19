"""tools/selector_overhead.py — Stage 15-D: selector/router 비용의 정밀
마이크로벤치. "unmeasurable" 대신 정확한 ms 수치를 만든다.

측정 항목 (각각 수백 회):
  mbd_scoring  : delta 계산 + mask/boundary/delta rank-normalized 결합
  rank_topk    : top-k + sort (select_hard_tokens)
  router       : learned CNN router forward (--draft-ckpt 제공 시)
  index_preparation : gather/scatter 인덱스 텐서 준비 (view/offset만)
  gather / scatter  : 실제 _gather_tokens / _scatter_tokens (D=3072 상태)

출력: ms/sparse-step + sparse-step transformer latency 대비 % +
50-step 샘플링 전체 대비 %.

  python -m tools.selector_overhead --resolution 1024 --ratio 0.3 \
      --sparse-step-ms 125.1 --steps 50 \
      [--draft-ckpt .../router_0060000.pt] --out selector_overhead.md
"""
import argparse
import statistics
import time

import torch


def bench(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    return statistics.median(ts)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--sparse-step-ms", type=float, required=True,
                    help="headline sparse-step transformer latency (ms)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num-sparse-steps", type=int, required=True,
                    help="실제 스케줄의 sparse step 수 (예: headline c2/t4 "
                         "50step -> 23; run.json evals a/s에서 읽을 것)")
    ap.add_argument("--draft-ckpt", default="")
    ap.add_argument("--out", default="selector_overhead.md")
    a = ap.parse_args()

    dev = "cuda"
    from utils.token_mapping import TokenGrid
    grid = TokenGrid(a.resolution, a.resolution).validate()
    hp, wp = grid.token_hw
    N = grid.num_image_tokens
    k = max(1, int(a.ratio * N))

    mask_token = (torch.rand(1, N, device=dev) > 0.7).float()
    v_prev = torch.randn(1, N, 64, device=dev, dtype=torch.bfloat16)
    v_curr = torch.randn(1, N, 64, device=dev, dtype=torch.bfloat16)

    from token_selectors.combo import ComboWeights, combo_score, \
        select_hard_tokens
    w = ComboWeights(alpha=1.0, beta=0.5, delta=1.0)  # mbd
    boundary = (torch.rand(1, N, device=dev) > 0.9).float()  # per-image 전처리

    def _score():
        # 스텝당 실비용: anchor-간 delta 계산 + rank-normalized 결합
        delta = (v_curr.float() - v_prev.float()).abs().mean(-1)
        return combo_score(w, mask=mask_token, boundary=boundary,
                           frequency=None, delta=delta)

    scores = _score()

    def _topk():
        return select_hard_tokens(scores, grid, a.ratio, block=1)

    hard, _, _ = _topk()

    def _index_prep():
        idx = hard.unsqueeze(-1).expand(-1, -1, 3072)
        pos = hard + 512
        return idx, pos

    # 실제 gather/scatter 비용 — runner와 동일 helper, 동일 shape/dtype
    from models.flux_sparse_transformer import _gather_tokens, _scatter_tokens
    x_state = torch.randn(1, N, 3072, device=dev, dtype=torch.bfloat16)
    fresh = torch.randn(1, k, 3072, device=dev, dtype=torch.bfloat16)

    results = {"mbd_scoring": bench(_score),
               "rank_topk": bench(_topk),
               "index_preparation": bench(_index_prep),
               "gather": bench(lambda: _gather_tokens(x_state, hard)),
               "scatter": bench(lambda: _scatter_tokens(x_state.clone(),
                                                        hard, fresh))}

    if a.draft_ckpt:
        from models.drafts.cnn_router import CNNRouter
        router = CNNRouter().to(dev).eval()
        sd = torch.load(a.draft_ckpt, map_location=dev)
        router.load_state_dict(sd.get("model", sd))
        lat = torch.randn(1, N, 64, device=dev)
        pred = torch.randn(1, N, 64, device=dev)
        t = torch.full((1,), 0.5, device=dev)
        results["router"] = bench(
            lambda: router(lat, mask_token, pred, t, (hp, wp)))

    rows = [f"# Selector/router overhead ({a.resolution}^2, r={a.ratio}, "
            f"median of 300 iters; scatter includes a clone of the [1,N,D] "
            f"state — an upper bound on the runner's in-place path)",
            "",
            "| component | ms/sparse step | % of sparse transformer step "
            "| % of full 50-step sampling |",
            "|---|---|---|---|"]
    # 전체 샘플링 대비: 실측 sparse-step 수 사용 (steps//2 추정 금지)
    full_ms = a.sparse_step_ms * a.steps          # 보수적 상한 (전부 sparse 가정)
    n_sp = a.num_sparse_steps
    for name, ms in results.items():
        rows.append(f"| {name} | {ms:.3f} | "
                    f"{100 * ms / a.sparse_step_ms:.2f}% | "
                    f"{100 * ms * n_sp / full_ms:.3f}% |")
        print(rows[-1])
    total = sum(results.values())
    rows.append(f"| **total** | {total:.3f} | "
                f"{100 * total / a.sparse_step_ms:.2f}% | "
                f"{100 * total * n_sp / full_ms:.3f}% |")
    open(a.out, "w").write("\n".join(rows) + "\n")
    print("->", a.out)


if __name__ == "__main__":
    main()