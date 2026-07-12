#!/usr/bin/env bash
# Stage 8: N=500 sanity (frontier 핵심 arm만) + KID @ N=500
# 목적: (1) N=100 frontier의 표본 안정성 방어, (2) 분포 metric(KID) 확보
# 예상: dense_s50 ref 2.3h + 5 arms ~8h + metrics ~0.5h ≈ 11h (overnight)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?set OUT (예: .../out_n500)}
N=${N:-500}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")

run() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
        --steps 50 "${PCARG[@]}" "$@"; }

# reference + dense frontier 2점
run --method dense --tag dense_s50
run --method dense --steps 40 --tag dense_s40
run --method dense --steps 30 --tag dense_s30
# frontier 핵심 arm 3개 (저/중/고 예산 tier 대표)
run --method reuse --cache-period 2 --dense-tail 4 --tag reuse_c2_t4
run --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
    --dual-sparse --kv-cache --tag mbd_c2_r03_t4_dualkv
run --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
    --kv-cache --tag mbd_c2_r03_t4_kv

for D in $OUT/dense_s40 $OUT/dense_s30 $OUT/reuse_c2_t4 \
         $OUT/mbd_c2_r03_t4_dualkv $OUT/mbd_c2_r03_t4_kv; do
  python -m eval.region_metrics --run $D --ref $OUT/dense_s50 --manifest $MAN --out $D/metrics.json
done
python -m eval.assemble --runs $OUT/dense_s40 $OUT/dense_s30 $OUT/reuse_c2_t4 \
  $OUT/mbd_c2_r03_t4_dualkv $OUT/mbd_c2_r03_t4_kv \
  --out $OUT/table_n500.md --csv $OUT/pareto_n500.csv

# KID @ N=500 (unbiased; 각 arm vs dense-50 ref) — clean-fid 필요
for D in dense_s30 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
  python -m eval.kid --run $OUT/$D --ref $OUT/dense_s50 --out $OUT/$D/kid.json \
    || { echo "KID skipped (pip install clean-fid --break-system-packages)"; break; }
done
