"""tools.sd3_latency — 2nd MMDiT의 measured latency vs analytic MAC (리뷰 §2).

512²/1024² x {dense, naive, kv, dualkv} x r{0.15, 0.3}:
median latency, speedup, analytic MAC ratio, realization(pred/meas), peak VRAM.

    python -m tools.sd3_latency --model-id stabilityai/stable-diffusion-3.5-large \
        --resolutions 512 1024 --ratios 0.15 0.3 --iters 20 --out sd3_latency.md
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id",
                    default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.15, 0.3])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from diffusers import StableDiffusion3Pipeline
    from models.sd3_sparse_runner import SD3SparseRunner, SD3AnchorCache
    from tools.mmdit_cost_model import mac, PRESETS

    dev, dtype = "cuda", torch.bfloat16
    pipe = StableDiffusion3Pipeline.from_pretrained(a.model_id,
                                                    torch_dtype=dtype)
    pipe.to(dev)
    runner = SD3SparseRunner(pipe.transformer)
    tcfg = pipe.transformer.config
    arch = dict(n_dual=tcfg.num_layers, n_single=0,
                D=tcfg.num_attention_heads * tcfg.attention_head_dim,
                m=4, T=333, out=tcfg.patch_size ** 2 * tcfg.out_channels,
                n_attn2=len(tuple(getattr(tcfg, "dual_attention_layers",
                                          ()) or ())))
    print(f"arch from config: {arch}")

    pe, _, pooled, _ = pipe.encode_prompt(
        prompt="a photo", prompt_2=None, prompt_3=None, device=dev)
    arch["T"] = pe.shape[1]

    def bench(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    rows = ["| res | variant | r | MAC(analytic) | latency(ms) | speedup "
            "| realization | peak VRAM(GB)¹ |",
            "|---|---|---|---|---|---|---|---|"]
    for res in a.resolutions:
        H = res // pipe.vae_scale_factor
        lat = torch.randn(1, tcfg.in_channels, H, H, device=dev, dtype=dtype)
        t = torch.tensor([500.0], device=dev)

        # dense를 cache 생성 *전*에 측정 — sparse cache가 상주한 상태의
        # peak가 아니라 순수 dense peak을 기록 (VRAM 공정성)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t_dense = bench(lambda: runner.dense_forward(lat, pe, t, pooled))
        vram = torch.cuda.max_memory_allocated() / 2**30
        rows.append(f"| {res} | dense | 1.0 | 1.000 "
                    f"| {t_dense*1e3:.1f} | 1.00x | -- | {vram:.1f} |")

        cache = SD3AnchorCache()
        runner.dense_forward(lat, pe, t, pooled, record=cache)
        N = cache.img_states[0].shape[1]
        rows.append(f"| {res} | naive (paper policy) | any | 1.000 "
                    f"| = dense | 1.00x | -- | -- |"
                    "  <!-- all-dual: sparse-eligible single 블록 없음 -->")
        for r in a.ratios:
            k = max(int(r * N), 1)
            hard = torch.randperm(N, device=dev)[:k].sort().values.unsqueeze(0)
            for lever in ("dual", "dualkv"):
                torch.cuda.reset_peak_memory_stats()
                tm = bench(lambda: runner.sparse_forward(
                    lat, pe, t, pooled, cache, hard, lever))
                vram = torch.cuda.max_memory_allocated() / 2**30
                pred = mac(arch, N, r, lever)
                sp = t_dense / tm
                real = sp * pred        # measured speedup / predicted speedup
                rows.append(f"| {res} | {lever} | {r} | {pred:.3f} "
                            f"| {tm*1e3:.1f} | {sp:.2f}x "
                            f"| {real:.0%} | {vram:.1f} |")
    md = (f"# SD3 latency vs analytic MAC ({a.model_id}, "
          f"{a.iters} iters median)\n\n" + "\n".join(rows) + "\n\n"
          "¹ dense 행은 anchor cache가 없는 상태의 peak; sparse 행은 cache 상주"
          " 상태의 peak (실사용 조건).\n"
          "realization = measured speedup x analytic MAC ratio "
          "(1.0이면 예측대로 실현).\n")
    Path(a.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()