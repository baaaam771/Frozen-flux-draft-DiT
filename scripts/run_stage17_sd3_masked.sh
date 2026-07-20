#!/usr/bin/env bash
# Stage 17 (종합평가2 패키지 A): SD3.5-Large controlled masked-generation
# transfer. sanity(N=50, 1 seed) -> 확정(N=100, +1 seed).
# arms: dense_ref(28) / dense_matched / reuse / refresh_r015 / refresh_r03 /
#       refresh_kv_r03 — selector·(c,r)·tail 규칙 모두 FLUX에서 재튜닝 없이.
# eval: mask-LPIPS→dense_ref (mask 영역), known PSNR, CLIPScore, wall
#       (sd3_quality_eval 재사용 + region 지표는 아래 eval 블록에서 계산).
#
#   OUT=/mnt/HDD_12TB/bam_ki/flux_fill/stage17_sd3m N=50 \
#   MAN=$PWD/data/coco_manifest_1024.json \
#     bash scripts/run_stage17_sd3_masked.sh
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
OUT=${OUT:?}; MAN=${MAN:-data/coco_manifest_1024.json}
N=${N:-50}; STEPS=${STEPS:-28}; SEEDOFF=${SEEDOFF:-0}

A=(--manifest $MAN --out $OUT --limit $N --steps $STEPS \
   --seed-offset $SEEDOFF)
python -m tools.sd3_masked_transfer "${A[@]}" --arms dense_ref
python -m tools.sd3_masked_transfer "${A[@]}" --arms refresh_r03
python -m tools.sd3_masked_transfer "${A[@]}" --arms refresh_kv_r03
python -m tools.sd3_masked_transfer "${A[@]}" --arms refresh_r015
python -m tools.sd3_masked_transfer "${A[@]}" --arms reuse
python -m tools.sd3_masked_transfer "${A[@]}" --arms dense_matched

# ---- region eval: mask-LPIPS→dense_ref + known PSNR ----
python - << 'PY'
import json, os, statistics
import numpy as np, torch
from PIL import Image
from data.dataset import load_image_rgb
from data.masks import MaskSpec, make_mask
import lpips

out = os.environ["OUT"]; man = os.environ["MAN"]
N = int(os.environ.get("N", 50))
dev = "cuda" if torch.cuda.is_available() else "cpu"
lp = lpips.LPIPS(net="vgg", spatial=True).to(dev).eval()
items = json.load(open(man))["items"][:N]
R = json.load(open(man)).get("resolution", 1024)

def to_t(img):
    return torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1
                            ).permute(2, 0, 1)[None].to(dev)

arms = [d for d in sorted(os.listdir(out))
        if os.path.exists(os.path.join(out, d, "run.json"))]
rows = ["| arm | steps | mask-LPIPS→dense_ref | known-PSNR→input | wall(s) | n |",
        "|---|---|---|---|---|---|"]
for arm in arms:
    d = os.path.join(out, arm)
    run = json.load(open(os.path.join(d, "run.json")))
    mls, psnrs, n = [], [], 0
    with torch.inference_mode():
        for it in items:
            sid = os.path.splitext(os.path.basename(str(it["sample_id"])))[0]
            fp = os.path.join(d, f"{sid}.png")
            rp = os.path.join(out, "dense_ref", f"{sid}.png")
            if not (os.path.exists(fp) and os.path.exists(rp)):
                continue
            gen = to_t(Image.open(fp).convert("RGB"))
            spec = MaskSpec(it["sample_id"], it["mask_type"], it["bucket"],
                            it["mask_seed"])
            m = make_mask(R, R, spec)[0].numpy().astype(np.float32)
            m_t = torch.from_numpy(m)[None, None].to(dev)
            if arm != "dense_ref":
                ref = to_t(Image.open(rp).convert("RGB"))
                dmap = lp(gen, ref)
                assert dmap.shape[-2:] != (1, 1), \
                    "Spatial LPIPS required for mask-region evaluation"
                mr = torch.nn.functional.interpolate(m_t, dmap.shape[-2:],
                                                     mode="area")
                mls.append(((dmap * mr).sum()
                            / mr.sum().clamp_min(1e-6)).item())
            inp = to_t(load_image_rgb(it["image"], R))
            known = (1 - m_t)
            mse = (((gen - inp) ** 2) * known).sum() / known.sum() / 3
            psnrs.append(10 * torch.log10(4.0 / mse.clamp_min(1e-10)).item())
            n += 1
    wall = statistics.median(r["wall_s"] for r in run["rows"]) \
        if run["rows"] else 0.0
    ml = f"{statistics.mean(mls):.4f}" if mls else "0 (self)"
    rows.append(f"| {arm} | {run['config']['steps']} | {ml} "
                f"| {statistics.mean(psnrs):.2f} | {wall:.2f} | {n} |")
    print(rows[-1])
open(os.path.join(out, "table_sd3_masked.md"), "w").write(
    "\n".join(rows) + "\n")
PY
python - << 'PY'
import json, os, statistics
out = os.environ["OUT"]
def med(arm):
    p = f"{out}/{arm}/run.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p)).get("rows", [])
    return statistics.median(x["wall_s"] for x in r) if r else None
dm, target = med("dense_matched"), (med("refresh_kv_r03")
                                    or med("refresh_r03"))
if dm and target and abs(dm - target) / target > 0.10:
    print(f"[warn] dense_matched wall {dm:.2f}s vs target {target:.2f}s "
          f"(>10% 어긋남) — dense step curve 실측 기반 재매칭 권장: "
          f"근처 step 2-3개를 --matched-steps로 N=10 측정 후 최근접 선택")
PY
echo "done -> $OUT/table_sd3_masked.md"