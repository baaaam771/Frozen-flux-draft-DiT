#!/usr/bin/env bash
# Stage 10: faithful TeaCache baseline (P0-1).
# 공식 TeaCache4FLUX 정책의 이식 — 판정신호/계수/누적/강제dense/residual 전부 보존.
# adaptation은 Fill ckpt + guidance temb + 계수 transfer(repo 권장 관행)뿐.
# 논문 표기: "TeaCache (adapted to FLUX Fill)".
#
# 순서: N=50 검증 -> (통과 시) N=500 sweep -> wallmatch로 headline 12.12s 대응점 선택.
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?set OUT}; N=${N:-500}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
DREF=$OUT/dense_s50
THRESHES=${THRESHES:-"0.25 0.4 0.6 0.8"}   # 공식 운영점 (1.5/1.8/2.0/2.25x)

# fix_t #2: 같은 OUT/같은 N의 ref와 MBD 비교점을 없으면 자동 생성
# (다른 N의 run과 한 표에 섞는 것은 불공정 — 반드시 동일 N/manifest/ref)
gen_if_missing() {
  local tag=$1; shift
  if [[ ! -f "$OUT/$tag/run.json" ]]; then
    python -m samplers.cached_flux_fill --manifest "$MAN" --out "$OUT" \
      --limit "$N" "${PCARG[@]}" "$@" --tag "$tag"
  fi
}
gen_if_missing dense_s50 --method dense --steps 50
gen_if_missing mbd_c2_r03_t4_dualkv --method cache_sparse --steps 50 \
  --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
  --dual-sparse --kv-cache
test -f $OUT/mbd_c2_r03_t4_dualkv/metrics.json || \
  python -m eval.region_metrics --run $OUT/mbd_c2_r03_t4_dualkv --ref $DREF \
    --manifest $MAN --out $OUT/mbd_c2_r03_t4_dualkv/metrics.json

for TH in $THRESHES; do
  TAG=teacache_l1_${TH/./}
  python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N --steps 50 \
    --method teacache --teacache-rel-l1 $TH "${PCARG[@]}" --tag $TAG
  python -m eval.region_metrics --run $OUT/$TAG --ref $DREF --manifest $MAN \
    --out $OUT/$TAG/metrics.json
done

# 실행 패턴 확인 (thresh별 calc/skip 수가 실제로 달라야) + wall-match
python -m eval.wallmatch --runs $OUT/teacache_l1_* --target-wall ${TARGET_WALL:-12.12} \
  --out $OUT/table_teacache_sweep.md
python -m eval.assemble --runs $OUT/teacache_l1_* $OUT/mbd_c2_r03_t4_dualkv \
  --out $OUT/table_teacache.md --csv $OUT/pareto_teacache.csv