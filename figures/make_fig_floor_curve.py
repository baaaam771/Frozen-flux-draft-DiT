"""figures/make_fig_floor_curve.py — 실측 floor curve + analytic 오버레이.

x = refresh ratio r, y = latency / dense. 4 lever의 실측 점(median, p10-p90
에러바)과 analytic MAC 곡선(점선), naive/dualkv floor 가로 주석.

  python figures/make_fig_floor_curve.py --data floor_curve.json \
      --out fig_floor_curve.pdf
"""
import argparse
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from models.flux_sparse_transformer import estimate_transformer_macs

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "mathtext.fontset": "cm"})

STYLE = {
    "naive":  dict(color="#888888", marker="o", label="naive sparse"),
    "kv":     dict(color="#2b8cbe", marker="s", label="+K/V"),
    "dual":   dict(color="#e6873c", marker="^", label="+dual"),
    "dualkv": dict(color="#c23b22", marker="D", label="+dual+K/V"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--n-dual", type=int, default=19)
    ap.add_argument("--n-single", type=int, default=38)
    ap.add_argument("--dim", type=int, default=3072)
    ap.add_argument("--out", default="fig_floor_curve.pdf")
    a = ap.parse_args()

    d = json.load(open(a.data))
    N = d["N"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    rs_an = np.linspace(1.0 / N, 1.0, 200)
    for lever, st in STYLE.items():
        flags = dict(kv_cached="kv" in lever and lever != "dual",
                     dual_sparse="dual" in lever)
        an = [estimate_transformer_macs(
            a.text_len, N, max(1, int(r * N)), a.n_dual, a.n_single,
            a.dim, **flags)["mac_ratio"] for r in rs_an]
        ax.plot(rs_an, an, ls="--", lw=0.8, color=st["color"], alpha=0.55)

        pts = d["curves"][lever]
        r = [p["r_actual"] for p in pts]
        y = [p["ratio_vs_dense"] for p in pts]
        ylo = [p["p10_ms"] / d["dense_ms"] for p in pts]
        yhi = [p["p90_ms"] / d["dense_ms"] for p in pts]
        ax.errorbar(r, y, yerr=[np.array(y) - ylo, np.array(yhi) - y],
                    fmt=st["marker"] + "-", ms=3, lw=1.1, capsize=1.5,
                    color=st["color"], label=st["label"])

    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    # floor 주석 (r→0 실측값)
    f_naive = d["curves"]["naive"][0]["ratio_vs_dense"]
    f_dkv = d["curves"]["dualkv"][0]["ratio_vs_dense"]
    ax.annotate(f"naive floor ${f_naive:.2f}\\times$",
                xy=(0.02, f_naive), xytext=(0.12, f_naive + 0.1),
                fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.annotate(f"dual+K/V floor ${f_dkv:.2f}\\times$",
                xy=(0.02, f_dkv), xytext=(0.12, max(f_dkv - 0.13, 0.02)),
                fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6))

    ax.set_xscale("log")
    ax.set_xlabel("refresh ratio $r$ (log; leftmost point: "
                  "$r\\to 0$, measured at $k{=}1$)")
    ax.set_ylabel("latency / dense")
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)

    pad = 0.06
    for _ in range(4):                       # 가장자리 잉크 자동검사
        fig.savefig(a.out, bbox_inches="tight", pad_inches=pad)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                    pad_inches=pad)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("L"))
        ink = arr < 128
        if not (ink[:, :3].any() or ink[:, -3:].any()
                or ink[:3, :].any() or ink[-3:, :].any()):
            print(f"saved {a.out} (pad={pad}, edges clean)")
            break
        pad += 0.05
    else:
        print(f"saved {a.out} (pad={pad}) — WARNING: edge ink persists")


if __name__ == "__main__":
    main()