"""eval.clipscore — text-image alignment (P0 품질평가).
raw output(마스크 채워진 전체 이미지)과 prompt 간 CLIP cosine ×100.
open_clip ViT-B/32 사용. composited 아닌 raw로 생성 정합성 측정.

    python -m eval.clipscore --run .../mbd --manifest .../coco_manifest_1024.json --out clip.json
"""
import argparse, json
from pathlib import Path
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pasted", action="store_true", help="composited 이미지 사용")
    a = ap.parse_args()

    import open_clip
    model, _, preproc = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()

    items = {Path(it["sample_id"]).stem: it["prompt"]
             for it in json.load(open(a.manifest))["items"]}
    run = Path(a.run)
    suffix = "_pasted.png" if a.pasted else ".png"
    scores = []
    with torch.no_grad():
        for stem, prompt in items.items():
            p = run / f"{stem}{suffix}"
            if not p.exists():
                continue
            img = preproc(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            txt = tok([prompt]).to(dev)
            fi = model.encode_image(img); ft = model.encode_text(txt)
            fi = fi / fi.norm(dim=-1, keepdim=True)
            ft = ft / ft.norm(dim=-1, keepdim=True)
            scores.append(100 * (fi * ft).sum().item())
    import statistics
    res = {"clipscore_mean": statistics.mean(scores),
           "clipscore_std": statistics.pstdev(scores), "n": len(scores)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
