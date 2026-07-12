"""eval.clipscore — text-image alignment (batched) + paired Δ vs dense-50.

full-image CLIPScore는 known region이 대부분이라 둔감할 수 있으므로,
dense-50 대비 PAIRED ΔCLIP(이미지 난이도 통제)을 함께 보고. bootstrap 95% CI.

    python -m eval.clipscore --run .../mbd --manifest .../coco_manifest_1024.json \
        --dense-ref .../dense_s50 --out clip.json
"""
import argparse, json, statistics, random
from pathlib import Path

import torch
from PIL import Image


def _embed_dir(run, stems, prompts, preproc, tok, model, dev, pasted, bs=64):
    suffix = "_pasted.png" if pasted else ".png"
    out = {}
    keys = [s for s in stems if (run / f"{s}{suffix}").exists()]
    for i in range(0, len(keys), bs):
        chunk = keys[i:i + bs]
        imgs = torch.stack([preproc(Image.open(run / f"{s}{suffix}").convert("RGB"))
                            for s in chunk]).to(dev)
        txts = tok([prompts[s] for s in chunk]).to(dev)
        with torch.no_grad():
            fi = model.encode_image(imgs); ft = model.encode_text(txts)
            fi = fi / fi.norm(dim=-1, keepdim=True)
            ft = ft / ft.norm(dim=-1, keepdim=True)
            cs = (100 * (fi * ft).sum(-1)).cpu().tolist()
        out.update(dict(zip(chunk, cs)))
    return out


def _boot_ci(vals, n=2000, seed=0):
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(n)]
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dense-ref", default=None)
    ap.add_argument("--pasted", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import open_clip
    model, _, preproc = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()

    prompts = {Path(it["sample_id"]).stem: it["prompt"]
               for it in json.load(open(a.manifest))["items"]}
    run = Path(a.run)
    stems = [p.stem for p in run.glob("*.png") if not p.stem.endswith("_pasted")]
    cs = _embed_dir(run, stems, prompts, preproc, tok, model, dev, a.pasted)
    vals = list(cs.values())
    res = {"encoder": "open_clip ViT-B-32 laion2b_s34b_b79k",
           "variant": "composited" if a.pasted else "raw",
           "clipscore_mean": statistics.mean(vals),
           "clipscore_std": statistics.pstdev(vals), "n": len(vals)}

    if a.dense_ref:
        cd = _embed_dir(Path(a.dense_ref), list(cs.keys()), prompts,
                        preproc, tok, model, dev, a.pasted)
        common = [s for s in cs if s in cd]
        deltas = [cs[s] - cd[s] for s in common]
        lo, hi = _boot_ci(deltas)
        res.update(delta_clip_vs_dense50_mean=statistics.mean(deltas),
                   delta_clip_ci95=[lo, hi], n_paired=len(deltas))

    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
