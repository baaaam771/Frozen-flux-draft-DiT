#!/usr/bin/env bash
# Stage 3e (Lever A): dual-stream sparsification — gate -> latency -> 품질 frontier
# 전제: Stage 3c와 동일 $OUT (1024², dense_s50 reference 재사용)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?set OUT}; N=${N:-100}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
IMG=${IMG:-sample.png}; MSK=${MSK:-sample_mask.png}; PROMPT=${PROMPT:-"a photo"}
test -d "$OUT/dense_s50" || { echo "Missing $OUT/dense_s50: run Stage 3c first"; exit 1; }

# Gate ladder: B0-dual -> B2(dual_sparse, dual_sparse+kv 모두 exact 기준)
python tests/test_dual_block_equivalence.py --resolution 512
python tests/test_dual_block_equivalence.py --resolution 512 --fp32
python tests/test_cache_exactness.py --image "$IMG" --mask "$MSK" --prompt "$PROMPT" \
  --step-index 0 --ratios 0.3 --dual-sparse
python tests/test_cache_exactness.py --image "$IMG" --mask "$MSK" --prompt "$PROMPT" \
  --step-index 0 --ratios 0.3 --dual-sparse --kv-cache
python tests/test_cache_exactness.py --image "$IMG" --mask "$MSK" --prompt "$PROMPT" \
  --step-index 0 --ratios 0.3 --dual-sparse --kv-cache --fp32

# latency: dual / dual+kv (est MAC r0.15 = 0.345 / 0.244)
python -m eval.latency --resolution 1024 --ratios 0.15 0.3 --dual-sparse \
  --out $OUT/latency_1024_dual.json
python -m eval.latency --resolution 1024 --ratios 0.15 0.3 --dual-sparse --kv-cache \
  --out $OUT/latency_1024_dual_kv.json

# 품질: 핵심 운영점 (dual+kv) — 예상 wall: c3_r015 ~8.4s, c5_r03 ~7.5s
run() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
        --steps 50 --method cache_sparse --selector mbd --dense-tail 4 \
        --dual-sparse --kv-cache "${PCARG[@]}" "$@"; }
run --cache-period 3 --ratio 0.15 --tag mbd_c3_r015_t4_dualkv
run --cache-period 3 --ratio 0.3  --tag mbd_c3_r03_t4_dualkv
run --cache-period 2 --ratio 0.3  --tag mbd_c2_r03_t4_dualkv
run --cache-period 5 --ratio 0.3  --tag mbd_c5_r03_t4_dualkv     # 깊은 c + refresh
run --cache-period 5 --ratio 0.5  --tag mbd_c5_r05_t4_dualkv

for D in $OUT/mbd_c*_dualkv; do
  python -m eval.region_metrics --run $D --ref $OUT/dense_s50 --manifest $MAN --out $D/metrics.json
done
python -m eval.assemble --runs $OUT/dense_s40 $OUT/dense_s30 $OUT/dense_s25 $OUT/dense_s20 \
  $OUT/reuse_c2 $OUT/reuse_c3_t4 \
  $OUT/mbd_c3_r015_t4_kv $OUT/mbd_c3_r03_t4_kv $OUT/mbd_c2_r03_t4_kv \
  $OUT/mbd_c3_r015_t4_dualkv $OUT/mbd_c3_r03_t4_dualkv $OUT/mbd_c2_r03_t4_dualkv \
  $OUT/mbd_c5_r03_t4_dualkv $OUT/mbd_c5_r05_t4_dualkv \
  --out $OUT/table_dual.md --csv $OUT/pareto_dual.csv
