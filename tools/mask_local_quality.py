"""tools/mask_local_quality.py — 패키지 C (1순위, GPU generation 0): 기존
5k 이미지로 mask-local real-quality 평가.

리뷰 지적 3.3 대응: full-image FID는 known region이 통계를 지배할 수 있음.
mask 영역만의 분포/유사도를 직접 평가한다.

지표:
  crop-FID / crop-KID : mask tight bbox + margin, 정사각 padding,
                        최소 crop 크기 강제, size-bucket(s/m/l) 분리
  masked-DINO         : DINOv2 feature map을 mask로 pooling한 뒤 real 원본
                        대비 cosine distance (crop resize 왜곡 없음)
  mask-LPIPS→real     : 보조 지표 (multimodal completion 주의 하에)

  python -m tools.mask_local_quality \
      --manifest data/coco_manifest_5k.json \
      --runs $B/dense_s50 $B/reuse_c2_t4 $B/mbd_c2_r03_t4_dualkv \
             $B/mbd_c2_r03_t4_kv \
      --limit 5000 --out $OUT/mask_local_quality.md
"""
import argparse
import json
import os
import statistics
import tempfile

import numpy as np
import torch
from PIL import Image

from data.dataset import load_image_rgb
from data.masks import MaskSpec, make_mask

MIN_CROP = 128          # 1024² 기준 최소 crop (리뷰 권고 128–192)
MARGIN = 32
BUCKET_NAMES = ("small", "medium", "large")   # manifest 규약 그대로 사용
# (재분류 금지 — 기존 논문의 mask-condition breakdown과 동일해야 함)


def _bbox_square(mask_np, H, W):
    ys, xs = np.where(mask_np > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0) + 2 * MARGIN
    side = int(min(max(side, MIN_CROP), H, W))   # 이미지 초과 clamp (검은 padding 방지)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0c = max(0, min(cy - side // 2, H - side))
    x0c = max(0, min(cx - side // 2, W - side))
    y1c, x1c = y0c + side, x0c + side
    assert 0 <= y0c < y1c <= H and 0 <= x0c < x1c <= W, (y0c, y1c, x0c, x1c)
    return y0c, y1c, x0c, x1c


def _load_dino(dev):
    """DINOv2 (timm) — masked pooling은 spatial token이 필수라 fallback 없음.
    실패 시 명확히 죽는다 (pooled-feature 대체는 'masked' 지표가 아님)."""
    import timm
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                          pretrained=True, num_classes=0)
    cfg = timm.data.resolve_model_data_config(m)
    tf = timm.data.create_transform(**cfg, is_training=False)
    return m.to(dev).eval(), tf, "dinov2-vits14", 14


@torch.inference_mode()
def _masked_feat(model, tf, patch, img, mask_np, dev, kind):
    x = tf(img)[None].to(dev)
    toks = model.forward_features(x)              # [1, 1+reg+N, D] (timm)
    n_pre = getattr(model, "num_prefix_tokens", 1)
    toks = toks[:, n_pre:]
    side = int(toks.shape[1] ** 0.5)
    m = torch.from_numpy(mask_np).float()[None, None]
    m = torch.nn.functional.interpolate(m, (side, side), mode="area")
    m = m.reshape(1, -1, 1).to(dev)
    f = (toks * m).sum(1) / m.sum().clamp_min(1e-6)
    return f / f.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", default="mask_local_quality.md")
    ap.add_argument("--skip-fid", action="store_true",
                    help="crop-FID/KID 생략 (DINO/LPIPS만)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = json.load(open(a.manifest))["items"][: a.limit]

    import lpips
    lp = lpips.LPIPS(net="vgg", spatial=True).to(dev).eval()
    dino, dino_tf, dino_kind, patch = _load_dino(dev)
    print(f"feature model: {dino_kind}")

    # real crop 사전 절단 (모든 run이 공유) — mask는 manifest spec에서
    # 런타임 재생성 (생성 시와 동일한 make_mask + SHA-seed 규약)
    R = json.load(open(a.manifest)).get("resolution", 1024)
    tmp_real = tempfile.mkdtemp(prefix="crop_real_")
    crops = {}          # sid -> (box, bucket, mask_np, real_img)
    for it in items:
        sid = os.path.splitext(os.path.basename(str(it["sample_id"])))[0]
        res = it.get("resolution", R)
        real = load_image_rgb(it["image"], res)
        spec = MaskSpec(it["sample_id"], it["mask_type"], it["bucket"],
                        it["mask_seed"])
        mask = make_mask(res, res, spec)[0].numpy().astype(np.float32)
        H, W = mask.shape
        box = _bbox_square(mask, H, W)
        if box is None:
            continue
        bucket = it["bucket"]                     # manifest 규약 그대로
        assert bucket in BUCKET_NAMES, bucket
        crops[sid] = (box, bucket, mask, real)
        y0, y1, x0, x1 = box
        real.crop((x0, y0, x1, y1)).resize((299, 299)).save(
            os.path.join(tmp_real, f"{sid}.png"))
    print(f"prepared {len(crops)} real crops -> {tmp_real}")

    real_feat = {}                                # sid -> masked real feature
    rows = ["| run | bucket | n | crop-FID | crop-KID | masked-DINO dist "
            "| mask-LPIPS→real |",
            "|---|---|---|---|---|---|---|"]
    for run in a.runs:
        tag = os.path.basename(run.rstrip("/"))
        per_bucket = {b: dict(dino=[], lpips=[]) for b in BUCKET_NAMES}
        tmp_gen = tempfile.mkdtemp(prefix=f"crop_{tag}_")
        gen_buckets = {b: [] for b in BUCKET_NAMES}
        for sid, (box, bucket, mask, real) in crops.items():
            fp = os.path.join(run, f"{sid}.png")
            if not os.path.exists(fp):
                continue
            gen = Image.open(fp).convert("RGB")
            y0, y1, x0, x1 = box
            gen.crop((x0, y0, x1, y1)).resize((299, 299)).save(
                os.path.join(tmp_gen, f"{sid}.png"))
            gen_buckets[bucket].append(sid)
            fg = _masked_feat(dino, dino_tf, patch, gen, mask, dev, dino_kind)
            if sid not in real_feat:                 # real은 run 간 공유·캐시
                real_feat[sid] = _masked_feat(dino, dino_tf, patch, real,
                                              mask, dev, dino_kind)
            fr = real_feat[sid]
            per_bucket[bucket]["dino"].append(1.0 - (fg * fr).sum().item())
            with torch.inference_mode():
                tg = torch.from_numpy(
                    np.asarray(gen, np.float32) / 127.5 - 1
                ).permute(2, 0, 1)[None].to(dev)
                tr = torch.from_numpy(
                    np.asarray(real.resize(gen.size), np.float32) / 127.5 - 1
                ).permute(2, 0, 1)[None].to(dev)
                m_t = torch.from_numpy(mask)[None, None].to(dev)
                dmap = lp(tg, tr)
                if dmap.shape[-2:] == (1, 1):
                    raise RuntimeError(
                        "LPIPS returned a scalar map; spatial=True required "
                        "for mask-region evaluation")
                m_r = torch.nn.functional.interpolate(
                    m_t, dmap.shape[-2:], mode="area")
                d = (dmap * m_r).sum() / m_r.sum().clamp_min(1e-6)
                per_bucket[bucket]["lpips"].append(d.item())

        fid_kid = {}
        if not a.skip_fid:
            from cleanfid import fid as cf
            for b in BUCKET_NAMES:
                if len(gen_buckets[b]) < 20:
                    fid_kid[b] = ("n/a", "n/a")
                    continue
                gb = tempfile.mkdtemp(); rb = tempfile.mkdtemp()
                for sid in gen_buckets[b]:
                    os.link(os.path.join(tmp_gen, f"{sid}.png"),
                            os.path.join(gb, f"{sid}.png"))
                    os.link(os.path.join(tmp_real, f"{sid}.png"),
                            os.path.join(rb, f"{sid}.png"))
                fid_kid[b] = (f"{cf.compute_fid(gb, rb):.2f}",
                              f"{cf.compute_kid(gb, rb):.6f}")
        for b in BUCKET_NAMES:
            dd, ll = per_bucket[b]["dino"], per_bucket[b]["lpips"]
            if not dd:
                continue
            f_s, k_s = fid_kid.get(b, ("-", "-"))
            rows.append(f"| {tag} | {b} | {len(dd)} | {f_s} | {k_s} "
                        f"| {statistics.mean(dd):.4f} "
                        f"| {statistics.mean(ll):.4f} |")
            print(rows[-1])
    open(a.out, "w").write("\n".join(rows) + "\n")
    print("->", a.out)


if __name__ == "__main__":
    main()