#!/usr/bin/env bash
# Stage 11: acceptance-polish (P1-4 budget, P1-2 tail+adaptive, P1-3 memory,
#           P2-1 router transfer 3점). 5K 종료 후 단독 실행 (~13h).
# 기본 OUT: stage9_n100 재사용 (dense_s50 ref 공유, provenance patch 완료본).
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?예: .../stage9_n100}
N=${N:-100}; PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
DREF=$OUT/dense_s50
DRAFT_CKPT=${DRAFT_CKPT:-}          # P2-1용 (비우면 router transfer 생략)
FFHQ_REF=${FFHQ_REF:-}              # 예: .../out_final/ffhq (dense_s50 포함 dir)
G10_REF=${G10_REF:-}                # 예: .../out_final/dense_s50_g10

run() { local tag=$1; shift
  test -f "$OUT/$tag/run.json" || \
    python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
      --steps 50 "${PCARG[@]}" "$@" --tag "$tag"; }
met() { test -f "$OUT/$1/metrics.json" || \
  python -m eval.region_metrics --run $OUT/$1 --ref ${2:-$DREF} --manifest ${3:-$MAN} \
    --out $OUT/$1/metrics.json; }

# ---------- A) P1-4: budget calibration r-sweep (c2, t4, dual+KV) ----------
for R in 0.05 0.1 0.2 0.4 0.5; do
  T=pol_r${R/./}
  run $T --method cache_sparse --selector mbd --cache-period 2 --ratio $R \
      --dense-tail 4 --dual-sparse --kv-cache
  met $T
done

# ---------- B) P1-2: tail sweep K∈{0..8}\{4} + adaptive 2점 ----------
for K in 0 1 2 3 5 6 8; do
  T=pol_tail${K}
  run $T --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 \
      --dense-tail $K --dual-sparse --kv-cache
  met $T
done
for A in 0.02 0.05; do
  T=pol_adapt${A/./}
  run $T --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 \
      --dense-tail 0 --adaptive-tail $A --dual-sparse --kv-cache
  met $T
done

python -m eval.assemble --runs $OUT/pol_r* $OUT/pol_tail* $OUT/pol_adapt* \
  $OUT/mbd_c2_r03_t4_dualkv \
  --out $OUT/table_polish.md --csv $OUT/pareto_polish.csv

# ---------- C) P1-3: memory 표 (별도 프로세스, GPU 단독) ----------
python -m tools.memory_table --resolutions 768 1024 1536 --ratio 0.15 \
  --out $OUT/memory_table.md

# ---------- D) P2-1: router transfer 3점 (ckpt 있을 때만) ----------
if [[ -n "$DRAFT_CKPT" ]]; then
  # (1) 같은 세트, r=0.15
  run pol_router_r015 --method cache_sparse --selector mbd_draft \
      --draft-ckpt $DRAFT_CKPT --cache-period 2 --ratio 0.15 --dense-tail 4 \
      --dual-sparse --kv-cache
  met pol_router_r015
  # 비교짝: 같은 r의 training-free
  run pol_mbd_r015 --method cache_sparse --selector mbd --cache-period 2 \
      --ratio 0.15 --dense-tail 4 --dual-sparse --kv-cache
  met pol_mbd_r015
  # (2) guidance 10 (자체 ref 필요)
  if [[ -n "$G10_REF" ]]; then
    run pol_router_g10 --method cache_sparse --selector mbd_draft \
        --draft-ckpt $DRAFT_CKPT --cache-period 2 --ratio 0.3 --dense-tail 4 \
        --dual-sparse --kv-cache --guidance 10
    met pol_router_g10 "$G10_REF"
  fi
  # (3) FFHQ (도메인 전이; 자체 manifest+ref)
  if [[ -n "$FFHQ_REF" ]]; then
    python -m samplers.cached_flux_fill --manifest data/ffhq_manifest_1024.json \
      --out $OUT --limit $N --steps 50 --method cache_sparse \
      --selector mbd_draft --draft-ckpt $DRAFT_CKPT --cache-period 2 \
      --ratio 0.3 --dense-tail 4 --dual-sparse --kv-cache --tag pol_router_ffhq
    met pol_router_ffhq "$FFHQ_REF/dense_s50" data/ffhq_manifest_1024.json
  fi
  python -m eval.assemble --runs $OUT/pol_router* $OUT/pol_mbd_r015 \
    --out $OUT/table_router_transfer.md
fi
