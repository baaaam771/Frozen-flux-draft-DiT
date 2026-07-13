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
DUALKV = {"$r{=}.15$": (11.33, .0357, .0015),
          "$r{=}.3$":  (12.12, .0300, .0031),
          "$r{=}.5$":  (13.44, .0251, .0045),
          "$c3\\,r.3$": (10.69, .0449, .0026)}
KVONLY = {"KV": (13.47, .0269, .0016)}
DRAFT = {"router": (12.11, .0288, .0003)}

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
    fig, ax = plt.subplots(figsize=(3.4, 2.9), dpi=300)

    xs, ys, es = zip(*DENSE)
    ax.errorbar(xs, ys, yerr=es, color=C_DENSE, marker="o", ms=3.5, lw=1.2,
                capsize=1.5, elinewidth=0.7, zorder=2,
                label="dense (uniform step red.)")
    dense_offs = {15: (-11, -9), 20: (-11, -9), 25: (-11, -9), 30: (-11, -9),
                  40: (-13, 6)}                        # 곡선 왼쪽으로 통일
    for x, y, s in zip(xs, ys, (15, 20, 25, 30, 40)):
        ax.annotate(f"{s}", (x, y), textcoords="offset points",
                    xytext=dense_offs[s], fontsize=6.5, color=C_DENSE)

    def scatter(d, color, marker, label):
        first = True
        for name, (x, y, e) in d.items():
            ax.errorbar(x, y, yerr=e, color=color, marker=marker, ms=5,
                        capsize=1.5, elinewidth=0.7, lw=0, zorder=3,
                        label=label if first else None)
            first = False
        return d

    scatter(REUSE, C_REUSE, "s", "anchored reuse ($r{=}0$)")
    scatter(DUALKV, C_DUAL, "^", "sel.\\ refresh, dual+KV")
    scatter(KVONLY, C_KV, "D", "sel.\\ refresh, KV-only")
    if a.draft:
        scatter(DRAFT, C_DRAFT, "*", "+ learned router")

    # 라벨: 밀집 구간(11.3~13.5s)은 leader line으로 빈 공간에 분산 배치.
    # (name, 라벨텍스트, 오프셋(pt), leader 여부)
    LABELS = [
        ("reuse $c{=}3$+tail", "reuse $c{=}3$+tail", (5, -2),  False),
        ("reuse $c{=}2$",      "reuse $c{=}2$",      (5, 2),   False),
        ("reuse $c{=}2$+tail", "reuse $c{=}2$+tail", (3, 7),   False),
        # 밀집 구간은 핵심 3점만 라벨 (r=.15/.5는 legend+caption이 설명)
        # router(12.11)는 ours(12.12)와 동일 지점 — 별 마커+legend로만 표시
        ("$r{=}.3$",  "ours ($r{=}.3$)",  (-56, -14), True),
        ("KV",        "KV-only",          (6, 4),    True),
    ]
    ALL = {**REUSE, **DUALKV, **KVONLY, **(DRAFT if a.draft else {})}
    for name, lbl, (ox, oy), leader in LABELS:
        if name not in ALL:
            continue
        x, y, _ = ALL[name]
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=5.8, zorder=4,
                    arrowprops=(dict(arrowstyle="-", lw=0.45, color="0.55",
                                     shrinkA=0, shrinkB=2) if leader else None))

    # headline 강조
    hx, hy, _ = DUALKV["$r{=}.3$"]
    dx, dy, _ = DENSE[-1]
    ax.annotate("", xy=(hx, hy), xytext=(dx, dy),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_DUAL, ls=":"))
    ax.text(14.25, 0.0355, "faster &\nbetter than\ndense-40",
            fontsize=6, color=C_DUAL, ha="center")

    ax.set_xlabel("wall-clock per image (s)")
    ax.set_ylabel("mask-region LPIPS $\\to$ ref  ($\\downarrow$)")
    ax.set_xlim(4.4, 15.3)
    ax.set_ylim(0.016, 0.132)
    ax.legend(frameon=False, fontsize=5.8, loc="lower left", ncol=2,
              bbox_to_anchor=(-0.02, 1.01), handletextpad=0.35,
              columnspacing=0.8, borderaxespad=0.0)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
