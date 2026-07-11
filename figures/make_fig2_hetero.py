"""figures/make_fig2_hetero.py — Fig.2: time-resolved two-factor test.

위: in-mask/out-mask 변화비 (log y) — regime과 말기 붕괴.
아래: top-30% share + E_rel — 붕괴 시점에 energy 스파이크.
dense-tail 구간(마지막 4 step) 음영.

    python figures/make_fig2_hetero.py \
        --report /mnt/HDD_12TB/bam_ki/flux_fill/out_stage3/hetero_report.json \
        --tail 4 --out fig2_hetero.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--tail", type=int, default=4)
    ap.add_argument("--out", default="fig2_hetero.pdf")
    a = ap.parse_args()

    per = json.load(open(a.report))["per_step"]
    steps = [r["step"] for r in per]
    io = [r["in_out_ratio"] for r in per]
    top = [r["top30_share"] for r in per]
    er = [r["energy_ratio"] for r in per]
    n = max(steps) + 1

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "font.family": "serif", "mathtext.fontset": "cm"})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.0), dpi=300,
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})

    for ax in (ax1, ax2):
        ax.axvspan(n - a.tail - 0.5, n - 0.5, color="#d62728", alpha=0.10, lw=0)
        ax.grid(alpha=0.25, lw=0.4)

    ax1.plot(steps, io, color="#1f77b4", lw=1.2)
    ax1.axhline(1.0, color="k", lw=0.6, ls=":")
    ax1.set_yscale("log")
    ax1.set_ylabel("in/out-mask change ratio")
    ax1.text(n - a.tail - 1, max(io) * 0.8, "dense tail",
             fontsize=6.5, color="#d62728", ha="right")
    ax1.annotate("regime collapse", xy=(steps[-1], io[-1]),
                 xytext=(-58, 14), textcoords="offset points", fontsize=6.5,
                 arrowprops=dict(arrowstyle="->", lw=0.6))

    ax2.plot(steps, top, color="#2ca02c", lw=1.2, label="top-30\\% share")
    ax2.set_ylabel("top-30\\% share", color="#2ca02c")
    ax2.set_ylim(0, 1)
    ax2b = ax2.twinx()
    ax2b.plot(steps, er, color="#9467bd", lw=1.2, ls="--",
              label="$E_{\\mathrm{rel}}$")
    ax2b.set_ylabel("$E_{\\mathrm{rel}}$", color="#9467bd")
    ax2.set_xlabel("denoising step")
    fig.align_ylabels()
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
