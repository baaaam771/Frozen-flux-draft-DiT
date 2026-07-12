#!/usr/bin/env bash
# Stage 9b: 5k real-COCO FID/KID (P0). Stage 9 N=500 sanity 통과 후에만 실행.
# COCO val2017 전체 5000장, 첫 caption, deterministic mask (spec_for_index).
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
OUT=${OUT:?}; RES=${RES:-1024}; PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
test -f data/coco_manifest_5k.json || python - << 'PY'
from data.dataset import build_manifest
m = build_manifest("/mnt/HDD_12TB/bam_ki/datasets/coco2017",
                   "data/coco_manifest_5k.json", n=5000, resolution=1024)
# mask type/bucket 균형 집계 (논문 보고용)
from collections import Counter
c = Counter((it["bucket"], it["mask_type"]) for it in m["items"])
print("5k manifest balance:", dict(c))
PY
MAN=data/coco_manifest_5k.json; DREF=$OUT/dense_s50
r5() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit 5000 \
       --steps 50 "${PCARG[@]}" "$@"; }
r5 --method dense --tag dense_s50
r5 --method dense --steps 30 --tag dense_s30
r5 --method reuse --cache-period 2 --dense-tail 4 --tag reuse_c2_t4
r5 --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
   --dual-sparse --kv-cache --tag mbd_c2_r03_t4_dualkv
r5 --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
   --kv-cache --tag mbd_c2_r03_t4_kv
for D in dense_s50 dense_s30 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
  python -m eval.kid --run $OUT/$D --manifest $MAN --resolution $RES --dense-ref $DREF \
    --out $OUT/$D/fidkid.json
  python -m eval.clipscore --run $OUT/$D --manifest $MAN --dense-ref $DREF \
    --out $OUT/$D/clip.json
done
python -m eval.assemble --runs $OUT/dense_s30 $OUT/reuse_c2_t4 \
  $OUT/mbd_c2_r03_t4_dualkv $OUT/mbd_c2_r03_t4_kv --out $OUT/table_5k.md
