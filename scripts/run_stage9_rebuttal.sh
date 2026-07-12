#!/usr/bin/env bash
# Stage 9: 리뷰 대응 (P0 품질평가 + P0 baseline). N=500 재사용 + 5k 생성.
#  A) 기존 acceleration baseline (mask-blind, latency-matched): fora / blockcache / uniform
#     + mask-only / random (이미 있음) — 전부 headline과 같은 wall에서 비교
#  B) FID/KID @ N=500 (이미 out_n500) + CLIPScore (raw)
#  C) [대용량] 5k FID/KID: dense-50 ref vs {dense-30, reuse_c2_t4, headline, kv-only}
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?set OUT}; N=${N:-500}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")

# ---------- A) baseline matrix (같은 c/r/tail, selector/method만 교체) ----------
bl() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
       --steps 50 --cache-period 2 --dense-tail 4 --dual-sparse --kv-cache \
       "${PCARG[@]}" "$@"; }
bl --method fora       --ratio 0.3 --tag base_fora_c2_r03
bl --method blockcache --ratio 0.3 --tag base_blockcache_c2_r03
bl --method cache_sparse --selector random --ratio 0.3 --tag base_random_c2_r03
bl --method cache_sparse --selector mask   --ratio 0.3 --tag base_maskonly_c2_r03
# teacache는 예산이 threshold로 정해지므로 별도 (latency는 forced-dense 수로 결정)
python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N --steps 50 \
  --method teacache --cache-period 2 --dense-tail 4 --teacache-thresh 0.15 \
  "${PCARG[@]}" --tag base_teacache_t015
for D in base_fora_c2_r03 base_blockcache_c2_r03 base_random_c2_r03 \
         base_maskonly_c2_r03 base_teacache_t015; do
  python -m eval.region_metrics --run $OUT/$D --ref $OUT/dense_s50 --manifest $MAN \
    --out $OUT/$D/metrics.json
done
python -m eval.assemble --runs $OUT/mbd_c2_r03_t4_dualkv \
  $OUT/base_fora_c2_r03 $OUT/base_blockcache_c2_r03 $OUT/base_random_c2_r03 \
  $OUT/base_maskonly_c2_r03 $OUT/base_teacache_t015 \
  --out $OUT/table_baselines.md --csv $OUT/pareto_baselines.csv

# ---------- B) CLIPScore (raw) + FID/KID @ N=500 ----------
for D in dense_s50 dense_s30 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
  python -m eval.clipscore --run $OUT/$D --manifest $MAN --out $OUT/$D/clip.json || \
    { echo "clipscore skipped (pip install open_clip_torch)"; break; }
done
for D in dense_s30 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
  python -m eval.kid --run $OUT/$D --ref $OUT/dense_s50 --out $OUT/$D/fidkid.json || break
done
