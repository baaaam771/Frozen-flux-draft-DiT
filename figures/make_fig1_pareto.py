"""figures/make_fig1_pareto.py — Fig.1: budget-tiered frontier (1024², 3 seeds).

데이터는 Stage 5 FINAL table_final.md의 3-seed mean±std를 내장 (assemble CSV에는
tail/kv/dual 구분이 없어 표가 원천). 재현: 표 갱신 시 아래 딕셔너리만 교체.

    python figures/make_fig1_pareto.py --out fig1_pareto.pdf [--draft]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (wall_s, mask_lpips_mean, std)
DENSE = [(4.97, .1226, .0020), (6.61, .0846, .0032), (8.24, .0616, .0040),
         (9.86, .0442, .0021), (13.12, .0328, .0033)]
REUSE = {"reuse $c{=}3$+tail": (6.61, .0730, .0031),
         "reuse $c{=}2$":      (8.24, .0585, .0018),
         "reuse $c{=}2$+tail": (8.90, .0469, .0017)}
DUALKV = {"$c3\\,r.3$":  (10.69, .0449, .0026),
          "$c2\\,r.15$": (11.33, .0357, .0015),
          "$c2\\,r.3$":  (12.12, .0300, .0031),
          "$c2\\,r.5$":  (13.44, .0251, .0045)}
KVONLY = {"$c2\\,r.3$ KV": (13.47, .0269, .0016)}
DRAFT = {"$c2\\,r.3$ +router": (12.11, .0288, .0003)}

C_DENSE, C_REUSE, C_DUAL, C_KV, C_DRAFT = \
    "#8c8c8c", "#1f77b4", "#d62728", "#9467bd", "#e07b00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig1_pareto.pdf")
    ap.add_argument("--draft", action="store_true",
                    help="learned-router 점 포함")
    a = ap.parse_args()

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)

    xs, ys, es = zip(*DENSE)
    ax.errorbar(xs, ys, yerr=es, color=C_DENSE, marker="o", ms=3.5, lw=1.2,
                capsize=1.5, elinewidth=0.7, zorder=2,
                label="dense (uniform step reduction)")
    for x, y, s in zip(xs, ys, (15, 20, 25, 30, 40)):
        ax.annotate(f"{s}", (x, y), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.5, color=C_DENSE)

    def scatter(d, color, marker, label):
        first = True
        for name, (x, y, e) in d.items():
            ax.errorbar(x, y, yerr=e, color=color, marker=marker, ms=5,
                        capsize=1.5, elinewidth=0.7, lw=0, zorder=3,
                        label=label if first else None)
            first = False
        return d

    scatter(REUSE, C_REUSE, "s", "anchored reuse ($r{=}0$)")
    scatter(DUALKV, C_DUAL, "^", "selective refresh, dual+KV")
    scatter(KVONLY, C_KV, "D", "selective refresh, KV-only")
    if a.draft:
        scatter(DRAFT, C_DRAFT, "*", "+ learned router")

    # 라벨 (수동 오프셋으로 겹침 방지)
    offs = {"reuse $c{=}3$+tail": (-2, -9), "reuse $c{=}2$": (4, 3),
            "reuse $c{=}2$+tail": (4, 2), "$c3\\,r.3$": (4, 3),
            "$c2\\,r.15$": (3, 4), "$c2\\,r.3$": (-6, -10),
            "$c2\\,r.5$": (-14, 6), "$c2\\,r.3$ KV": (4, -3),
            "$c2\\,r.3$ +router": (5, 4)}
    for d in ([REUSE, DUALKV, KVONLY] + ([DRAFT] if a.draft else [])):
        for name, (x, y, e) in d.items():
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=offs.get(name, (4, 3)), fontsize=6.5)

    # headline 강조
    hx, hy, _ = DUALKV["$c2\\,r.3$"]
    dx, dy, _ = DENSE[-1]
    ax.annotate("", xy=(hx, hy), xytext=(dx, dy),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_DUAL, ls=":"))
    ax.text(0.5 * (hx + dx) + 0.15, 0.5 * (hy + dy) + 0.0005,
            "faster & better\nthan dense-40", fontsize=6, color=C_DUAL)

    ax.set_xlabel("wall-clock per image (s)")
    ax.set_ylabel("mask-region LPIPS $\\to$ ref  ($\\downarrow$)")
    ax.set_xlim(4.4, 14.6)
    ax.set_ylim(0.018, 0.132)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right",
              handletextpad=0.4, borderaxespad=0.2)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
