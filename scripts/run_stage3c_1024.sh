#!/usr/bin/env bash
# Stage 3c: 1024² (FLUX native res) 품질 suite — latency 실측에서 sparse 절감이
# 1024에서 실현됨(1.45~1.58x, 이론 대비 91%)을 확인했으므로 headline 해상도 이동.
# 예상: sample당 dense ~16s, 전체 suite ~5.5h (N=100)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?set OUT}; N=${N:-100}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
test -f "$MAN" || { echo "Missing $MAN — build_manifest(resolution=1024) 먼저"; exit 1; }

run() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
        --steps 50 "${PCARG[@]}" "$@"; }

# dense frontier (s50 = reference)
for S in 50 40 30 25 20; do
  python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
    --method dense --steps $S --tag dense_s$S "${PCARG[@]}"
done
# r=0 reuse arms
run --method reuse --cache-period 2 --tag reuse_c2
run --method reuse --cache-period 3 --tag reuse_c3
run --method reuse --cache-period 3 --dense-tail 4 --tag reuse_c3_t4
# selective refresh arms (mbd) — 512에서 확정된 운영점 중심
for CFG in "2 0.15" "2 0.3" "3 0.15" "3 0.3"; do
  set -- $CFG
  run --method cache_sparse --selector mbd --cache-period $1 --ratio $2 \
      --dense-tail 4 --tag mbd_c$1_r${2/0./0}_t4
done
# 대조군 (한 지점)
run --method cache_sparse --selector random --cache-period 3 --ratio 0.3 --dense-tail 4 --tag random_c3_r03_t4
run --method cache_sparse --selector oracle --cache-period 3 --ratio 0.3 --dense-tail 4 --tag oracle_c3_r03_t4

for D in $OUT/dense_s40 $OUT/dense_s30 $OUT/dense_s25 $OUT/dense_s20 \
         $OUT/reuse_c* $OUT/mbd_c* $OUT/random_c3_r03_t4 $OUT/oracle_c3_r03_t4; do
  python -m eval.region_metrics --run $D --ref $OUT/dense_s50 --manifest $MAN --out $D/metrics.json
done
python -m eval.assemble --runs $OUT/dense_s40 $OUT/dense_s30 $OUT/dense_s25 $OUT/dense_s20 \
  $OUT/reuse_c2 $OUT/reuse_c3 $OUT/reuse_c3_t4 \
  $OUT/mbd_c2_r015_t4 $OUT/mbd_c2_r03_t4 $OUT/mbd_c3_r015_t4 $OUT/mbd_c3_r03_t4 \
  $OUT/random_c3_r03_t4 $OUT/oracle_c3_r03_t4 \
  --out $OUT/table_1024.md --csv $OUT/pareto_1024.csv
