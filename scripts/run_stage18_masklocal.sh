#!/usr/bin/env bash
# Stage 18 (종합평가2 패키지 C, 1순위 — GPU generation 0): 기존 5k 이미지로
# mask-local real-quality (crop FID/KID + masked-DINO + mask-LPIPS→real).
#
#   B=/mnt/HDD_12TB/bam_ki/flux_fill/stage9_5k_final \
#   MAN=/path/to/coco_manifest_5k.json \
#   OUT=/mnt/HDD_12TB/bam_ki/flux_fill/stage18_masklocal \
#     bash scripts/run_stage18_masklocal.sh
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
B=${B:?5k run 상위 디렉터리}; MAN=${MAN:?5k manifest}
OUT=${OUT:?}; mkdir -p "$OUT"
LIMIT=${LIMIT:-5000}   # 소규모 smoke: LIMIT=20 (path/mask/DINO/LPIPS 동작 확인용)
pip show timm > /dev/null 2>&1 || pip install timm --quiet || true
python -m tools.mask_local_quality --manifest $MAN --limit $LIMIT \
  --out "$OUT/mask_local_quality.md" --runs \
  "$B/dense_s50" "$B/dense_s30" "$B/reuse_c2_t4" \
  "$B/mbd_c2_r03_t4_dualkv" "$B/mbd_c2_r03_t4_kv"
echo "done -> $OUT/mask_local_quality.md"