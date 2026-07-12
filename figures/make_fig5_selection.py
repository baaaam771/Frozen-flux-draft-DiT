"""figures/make_fig5_selection.py — Fig.5: 시간에 따른 refresh 선택 맵.

한 샘플의 [input+mask | 초기/중기/말기 step의 선택 빈도 히트맵 | 전 구간 누적].
selector가 초기엔 mask 내부, 이후 boundary/Δ로 이동하는 양상을 보인다.

    python figures/make_fig5_selection.py \
        --run .../seed0/mbd_c2_r03_t4_dualkv --sample 000000XXXXXX \
        --manifest data/coco_manifest_1024.json --out fig5_selection.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _freq_map(sel, hw, lo, hi):
    hp, wp = hw
    m = torch.zeros(hp * wp)
    steps = [r for r in sel if lo <= r["step"] < hi]
    for r in steps:
        m[r["hard_idx"]] += 1
    return (m / max(len(steps), 1)).view(hp, wp).numpy()


def _heat(freq, size):
    import matplotlib.cm as cm
    rgba = plt.get_cmap("inferno")(np.clip(freq, 0, 1))
    img = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    return np.array(img.resize((size, size), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sample", required=True, help="sample stem")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--bands", type=int, nargs="+", default=[0, 17, 34, 46],
                    help="구간 경계 (예: 초기/중기/말기/tail)")
    ap.add_argument("--out", default="fig5_selection.png")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, ".")
    from data.dataset import FluxFillBenchmark
    ds = FluxFillBenchmark(a.manifest)
    idx = next(i for i in range(len(ds))
               if Path(ds[i]["sample_id"]).stem == a.sample)
    s = ds[idx]

    pack = torch.load(Path(a.run) / f"{a.sample}_selection.pt")
    sel, hw = pack["selection"], pack["token_hw"]
    n_steps = max(r["step"] for r in sel) + 1
    bounds = a.bands + [n_steps]

    C = a.cell
    cols = 1 + (len(bounds) - 1) + 1
    canvas = Image.new("RGB", (C * cols, C + 24), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    img = (s["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    m = s["mask"][0].numpy()[..., None]
    over = (img * (1 - 0.45 * m) + np.array([255, 40, 40]) * 0.45 * m).astype(np.uint8)
    canvas.paste(Image.fromarray(over).resize((C, C), Image.LANCZOS), (0, 24))
    draw.text((6, 4), "input+mask", fill="black", font=font)

    for j in range(len(bounds) - 1):
        lo, hi = bounds[j], bounds[j + 1]
        canvas.paste(Image.fromarray(_heat(_freq_map(sel, hw, lo, hi), C)),
                     ((1 + j) * C, 24))
        draw.text(((1 + j) * C + 6, 4), f"steps {lo}\u2013{hi - 1}",
                  fill="black", font=font)
    canvas.paste(Image.fromarray(_heat(_freq_map(sel, hw, 0, n_steps), C)),
                 ((cols - 1) * C, 24))
    draw.text(((cols - 1) * C + 6, 4), "all sparse steps", fill="black", font=font)
    canvas.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
