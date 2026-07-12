#!/usr/bin/env bash
# Stage 9b: 5k FID/KID (P0). 별도 5k manifest 필요 (~5.5h/arm dense50 기준 大).
#  주의: dense-50 5k만 ~15h. 핵심 arm만. overnight×2 규모.
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
OUT=${OUT:?}; PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
# 5k manifest (COCO val 5000장)
test -f data/coco_manifest_5k.json || python - << 'PY'
from data.dataset import build_manifest
build_manifest("/mnt/HDD_12TB/bam_ki/datasets/coco2017", "data/coco_manifest_5k.json",
               n=5000, resolution=1024)
PY
MAN=data/coco_manifest_5k.json
r5() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit 5000 \
       --steps 50 "${PCARG[@]}" "$@"; }
r5 --method dense --tag dense_s50
r5 --method dense --steps 30 --tag dense_s30
r5 --method reuse --cache-period 2 --dense-tail 4 --tag reuse_c2_t4
r5 --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
   --dual-sparse --kv-cache --tag mbd_c2_r03_t4_dualkv
r5 --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
   --kv-cache --tag mbd_c2_r03_t4_kv
for D in dense_s30 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
  python -m eval.kid --run $OUT/$D --ref $OUT/dense_s50 --out $OUT/$D/fidkid.json
  python -m eval.clipscore --run $OUT/$D --manifest $MAN --out $OUT/$D/clip.json
done
python -m eval.clipscore --run $OUT/dense_s50 --manifest $MAN --out $OUT/dense_s50/clip.json
