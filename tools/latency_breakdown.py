"""tools/latency_breakdown.py — Stage 15-C: 실제 GPU 시간의 cost breakdown.

"0.49 floor가 MAC 계산상의 숫자가 아니라 실제 GPU latency에서도 관찰되는
구조적 현상"임을 profiler로 보인다. FLUX_PROFILE=1 환경에서 transformer의
record_function 태그(dual_stream / single_kv_projection / single_kv_scatter /
single_q_mlp / single_attention / sparse_overhead / cache_record /
final_head)를 torch.profiler로 수집해 그룹으로 집계한다:

  dual_blocks  = dual_stream (블록 전체)
  single_kv    = single_kv_projection + single_kv_scatter (K/V만)
  single_other = single_q_mlp + single_attention
  overhead     = sparse_overhead + cache_record
  head         = final_head

비율은 profiled CUDA time 합 기준 share로 보고하고, clean wall-clock
total은 별도 열로 병기한다 (profiler CUDA time과 wall-clock은 측정
도메인이 달라 빼기 연산으로 "other"를 만들면 안 됨).

기대 결과: naive에서 dual_blocks와 single_kv가 dense 수준으로 남고
(=floor의 정체), dual+KV에서 두 고정비가 모두 감소하며 overhead
증가분은 작다.

  FLUX_PROFILE=1 python -m tools.latency_breakdown --resolution 1024 \
      --ratio 0.15 --iters 12 --out breakdown_1024.md
"""
import argparse
import json
import os

import torch

assert os.environ.get("FLUX_PROFILE") == "1", \
    "FLUX_PROFILE=1 환경에서 실행해야 태그가 활성화됩니다"

from torch.profiler import ProfilerActivity, profile

from eval.latency import load_transformer_only
from models.flux_cache import FluxAnchorCache
from models.flux_sparse_transformer import FluxSparseRunner

CONFIGS = [
    ("dense",  None),
    ("naive",  dict(kv_cache=False, dual_sparse=False)),
    ("kv",     dict(kv_cache=True,  dual_sparse=False)),
    ("dual",   dict(kv_cache=False, dual_sparse=True)),
    ("dualkv", dict(kv_cache=True,  dual_sparse=True)),
]
# 내부 세분 태그 기반 그룹 — single K/V projection이 별도 항목이므로
# "full-sequence K/V가 floor를 만든다"를 profiler로 직접 보일 수 있다.
GROUPS = {
    "dual_blocks": ["dual_stream"],
    "single_kv":   ["single_kv_projection", "single_kv_scatter"],
    "single_other": ["single_q_mlp", "single_attention"],
    "overhead":    ["sparse_overhead", "cache_record"],
    "head":        ["final_head"],
}


def _collect(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tag_ms = {}
    total_ms = 0.0
    for ev in prof.key_averages():
        cuda_ms = ev.device_time_total / 1e3 / iters
        if ev.key in sum(GROUPS.values(), []):
            tag_ms[ev.key] = tag_ms.get(ev.key, 0.0) + cuda_ms
        if ev.key == "ProfilerStep*":
            continue
    # 전체는 별도 벽시계로 측정 (프로파일 오버헤드 제외한 상대 분해가 목적)
    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    total_ms = 1e3 * (time.perf_counter() - t0) / iters
    return tag_ms, total_ms


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--ratio", type=float, default=0.15)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--out", default="breakdown.md")
    a = ap.parse_args()

    comps, x, pe, po, ts, gd, img_ids, txt_ids = load_transformer_only(
        a.resolution)
    runner = FluxSparseRunner(comps.transformer)
    dev = x.device
    N = x.shape[1]
    k = max(1, int(a.ratio * N))
    hard = torch.sort(torch.randperm(N, device=dev)[:k]).values[None]

    rows = [f"# Latency breakdown ({a.resolution}^2, r={a.ratio}) — "
            f"profiled CUDA time by tagged region ({a.iters} iters); "
            "shares are of the profiled-tag sum; clean total is a separate "
            "un-profiled wall-clock",
            "",
            "| config | dual blocks | single K/V proj | single attn+Q/MLP "
            "| overhead | head | clean total(ms) |",
            "|---|---|---|---|---|---|---|"]
    raw = {}
    for name, flags in CONFIGS:
        if name == "dense":
            fn = lambda: runner.dense_forward(x, pe, po, ts, gd, img_ids,
                                              txt_ids)
        else:
            cache = FluxAnchorCache()
            runner.dense_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                                 cache=cache, step_index=0,
                                 record_kv=flags["kv_cache"],
                                 record_dual=flags["dual_sparse"])
            fn = (lambda c=cache, f=flags: runner.sparse_forward(
                x, pe, po, ts, gd, img_ids, txt_ids, c, hard, **f))
        tag_ms, total = _collect(fn, a.iters)
        g = {gname: sum(tag_ms.get(t, 0.0) for t in tags)
             for gname, tags in GROUPS.items()}
        psum = max(sum(g.values()), 1e-9)
        share = {k2: v / psum for k2, v in g.items()}
        raw[name] = dict(groups_ms=g, profiled_sum_ms=psum,
                         shares=share, clean_total_ms=total)
        rows.append(
            f"| {name} | {g['dual_blocks']:.1f} ({share['dual_blocks']:.0%}) "
            f"| {g['single_kv']:.1f} ({share['single_kv']:.0%}) "
            f"| {g['single_other']:.1f} ({share['single_other']:.0%}) "
            f"| {g['overhead']:.2f} ({share['overhead']:.0%}) "
            f"| {g['head']:.2f} | {total:.1f} |")
        print(rows[-1])

    dense_total = raw["dense"]["clean_total_ms"]
    rows += ["", "dense 대비 clean-total 비율: " + ", ".join(
        f"{n} {raw[n]['clean_total_ms'] / dense_total:.3f}x"
        for n, _ in CONFIGS)]
    open(a.out, "w").write("\n".join(rows) + "\n")
    json.dump(raw, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()