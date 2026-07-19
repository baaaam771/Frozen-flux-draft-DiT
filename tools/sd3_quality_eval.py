"""tools/sd3_quality_eval.py — Stage 14 집계: LPIPS(전체 이미지) -> 자기
backbone의 dense_ref, CLIPScore(캡션), median wall.

  python -m tools.sd3_quality_eval --out $OUT \
      --manifest data/coco_manifest_1024.json --limit 100
"""
import argparse
import json
import os
import statistics
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ref-arm", default="dense_ref")
    a = ap.parse_args()

    import lpips
    from PIL import Image
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="vgg").to(dev).eval()

    try:
        import open_clip
        clip_model, _, clip_pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        clip_tok = open_clip.get_tokenizer("ViT-B-32")
        clip_model = clip_model.to(dev).eval()
    except Exception:
        clip_model = None

    items = {
        Path(str(it["sample_id"])).stem: it
        for it in json.load(open(a.manifest))["items"][:a.limit]
    }

    def to_t(p):
        arr = np.asarray(Image.open(p).convert("RGB"), np.float32) / 127.5 - 1
        return torch.from_numpy(arr).permute(2, 0, 1)[None].to(dev)

    arms = [d for d in sorted(os.listdir(a.out))
            if os.path.isdir(os.path.join(a.out, d))
            and os.path.exists(os.path.join(a.out, d, "run.json"))]
    ref_dir = os.path.join(a.out, a.ref_arm)
    rows = ["| arm | steps | LPIPS->dense_ref | CLIPScore | wall(s) | evals a/s | n |",
            "|---|---|---|---|---|---|---|"]
    for arm in arms:
        d = os.path.join(a.out, arm)
        run = json.load(open(os.path.join(d, "run.json")))
        lps, clips = [], []
        n = 0
        with torch.inference_mode():
            for sid, it in items.items():
                fp = os.path.join(d, f"{sid}.png")
                rp = os.path.join(ref_dir, f"{sid}.png")
                if not (os.path.exists(fp) and os.path.exists(rp)):
                    continue
                x = to_t(fp)
                if arm != a.ref_arm:
                    lps.append(lp(x, to_t(rp)).item())
                if clip_model is not None:
                    im = clip_pre(Image.open(fp).convert("RGB"))[None].to(dev)
                    tx = clip_tok([it["prompt"]]).to(dev)
                    fi = clip_model.encode_image(im)
                    ft = clip_model.encode_text(tx)
                    fi = fi / fi.norm(dim=-1, keepdim=True)
                    ft = ft / ft.norm(dim=-1, keepdim=True)
                    clips.append(100.0 * (fi * ft).sum().item())
                n += 1
        wall = statistics.median(r["wall_s"] for r in run["rows"]) \
            if run["rows"] else 0.0
        ev = (f"{run['rows'][0].get('anchor', '-')}"
              f"/{run['rows'][0].get('sparse', '-')}" if run["rows"] else "-")
        lp_s = f"{statistics.mean(lps):.4f}" if lps else "0 (self)"
        cl_s = f"{statistics.mean(clips):.2f}" if clips else "-"
        rows.append(f"| {arm} | {run['config']['steps']} | {lp_s} | {cl_s} "
                    f"| {wall:.2f} | {ev} | {n} |")
    md = "\n".join(rows) + "\n"
    open(os.path.join(a.out, "table_sd3_quality.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
