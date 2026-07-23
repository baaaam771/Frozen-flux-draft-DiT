"""figures/make_fig4_qualitative.py — Fig.4: qualitative grid.

행 = 샘플, 열 = [input+mask overlay | dense-50 (ref) | 지정한 method들].
각 method 열 헤더에 wall(s)과 per-image mask-LPIPS를 표기. pasted 출력을 사용
(seam까지 보이도록) — --raw로 raw 출력 전환.

    python figures/make_fig4_qualitative.py \
        --seed-dir /path/to/results/out_final/seed0 \
        --manifest data/coco_manifest_1024.json \
        --methods dense_s20 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_draft_c2_r03_t4_dualkv \
        --indices 0 3 7 12 --out fig4_qual.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _load(p, size):
    return np.array(Image.open(p).convert("RGB").resize((size, size),
                                                         Image.LANCZOS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--out", default="fig4_qual.png")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, ".")
    from data.dataset import FluxFillBenchmark
    ds = FluxFillBenchmark(a.manifest)
    sd = Path(a.seed_dir)
    ref_dir = sd / "dense_s50"
    cols = ["input+mask", "dense-50 (ref)"] + a.methods
    C, R, H = a.cell, len(a.indices), 26          # H = header band

    def method_meta(m):
        run = json.load(open(sd / m / "run.json"))
        rows = [r for r in run["rows"] if not r.get("warmup")]
        wall = sum(r["wall_s"] for r in rows) / max(len(rows), 1)
        met_p = sd / m / "metrics.json"
        per = {}
        if met_p.exists():
            for r in json.load(open(met_p))["rows"]:
                per[r["sample_id"]] = r.get("mask_lpips_to_ref")
        return wall, per

    meta = {m: method_meta(m) for m in a.methods}
    W_img = C * len(cols)
    canvas = Image.new("RGB", (W_img, H + R * C), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        fsmall = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = fsmall = ImageFont.load_default()

    # 열 라벨: 내부 태그 -> 논문 명칭 (리뷰 지적: mbd_draft는 미정의 태그)
    PRETTY = {"dense": "dense", "reuse": "reuse",
              "mbd": "mbd", "mbd_draft": "mbd + learned router"}
    def _pretty(name):
        stem = name.split("_c")[0]
        if stem.startswith("dense_s"):                 # dense_s20 -> dense-20
            return f"dense-{stem.split('_s')[1]}"
        return PRETTY.get(stem, stem)

    for j, cname in enumerate(cols):
        label = _pretty(cname) if cname not in meta else \
            f"{_pretty(cname)}  ({meta[cname][0]:.1f}s)"
        draw.text((j * C + 6, 6), label, fill="black", font=font)

    for i, idx in enumerate(a.indices):
        s = ds[idx]
        stem = Path(s["sample_id"]).stem
        y = H + i * C
        # col 0: input + red mask overlay
        img = (s["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        m = s["mask"][0].numpy()[..., None]
        over = (img * (1 - 0.45 * m) +
                np.array([255, 40, 40]) * 0.45 * m).astype(np.uint8)
        canvas.paste(Image.fromarray(over).resize((C, C), Image.LANCZOS),
                     (0, y))
        # col 1: reference
        canvas.paste(Image.fromarray(_load(ref_dir / f"{stem}.png", C)), (C, y))
        # method cols
        suffix = ".png" if a.raw else "_pasted.png"
        for j, mth in enumerate(a.methods):
            p = sd / mth / f"{stem}{suffix}"
            if not p.exists():
                p = sd / mth / f"{stem}.png"
            canvas.paste(Image.fromarray(_load(p, C)), ((2 + j) * C, y))
            lp = meta[mth][1].get(stem)
            if lp is not None:
                draw.rectangle([(2 + j) * C + 4, y + C - 20,
                                (2 + j) * C + 78, y + C - 4], fill="white")
                draw.text(((2 + j) * C + 7, y + C - 19), f"{lp:.4f}",
                          fill="black", font=fsmall)

    canvas.save(a.out)
    print(f"saved {a.out}  ({W_img}x{H + R * C})")


if __name__ == "__main__":
    main()