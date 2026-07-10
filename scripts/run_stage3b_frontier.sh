#!/usr/bin/env bash
# Stage 3b: 운영점 frontier 탐색 (Stage 3 결과 반영)
#   근거: (1) hetero 곡선상 step ~46부터 regime 붕괴 -> dense tail
#         (2) c=3/r=0.3 단일점은 dense_s30에 Pareto 열세 -> c/r/tail sweep으로
#             frontier 자체를 그려 비교해야 함
# 전제: Stage 3와 동일한 $OUT (dense_s50 reference 재사용)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest.json}; OUT=${OUT:?set OUT to the stage3 dir}; N=${N:-100}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
test -d "$OUT/dense_s50" || { echo "Missing $OUT/dense_s50: run Stage 3 first"; exit 1; }

run() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
        --method cache_sparse --selector mbd --steps 50 "${PCARG[@]}" "$@"; }

# 1) dense tail의 효과 격리 (c=3, r=0.3 고정)
for TAIL in 2 4 6; do
  run --cache-period 3 --ratio 0.3 --dense-tail $TAIL --tag mbd_c3_r03_t$TAIL
done
# 2) tail=4 고정, c x r frontier
for C in 2 3 4; do
  for R in 0.15 0.3 0.5; do
    run --cache-period $C --ratio $R --dense-tail 4 --tag mbd_c${C}_r${R/0./0}_t4
  done
done
# 3) r=0 reuse + tail (가장 싼 arm의 방어선)
python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
  --method reuse --steps 50 --cache-period 3 --dense-tail 4 --tag reuse_c3_t4 "${PCARG[@]}"

for D in $OUT/mbd_c*_t* $OUT/reuse_c3_t4; do
  python -m eval.region_metrics --run $D --ref $OUT/dense_s50 --manifest $MAN --out $D/metrics.json
done
python -m eval.assemble --runs $OUT/dense_s30 $OUT/dense_s25 $OUT/dense_s20 \
  $OUT/reuse_c3 $OUT/reuse_c3_t4 $OUT/mbd_c3_r03 $OUT/mbd_c*_t* \
  --out $OUT/table_frontier.md --csv $OUT/pareto_frontier.csv
