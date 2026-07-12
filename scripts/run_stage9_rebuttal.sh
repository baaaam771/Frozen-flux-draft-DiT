#!/usr/bin/env bash
# Stage 9 (수정판): selector-control ablation + adapted temporal baseline + real-ref 품질.
# 두 그룹을 분리 (리뷰 4번):
#  A) CONTROL (동일 anchored backend, selector만 교체): mask-aware 이득 격리
#  B) ADAPTED temporal-threshold (faithful TeaCache 아님, 명시) + wall-matched sweep
#  C) real-COCO FID/KID (raw+composited) + dense-50 fidelity + batched CLIPScore
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?set OUT}; N=${N:-500}; RES=${RES:-1024}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
DREF=$OUT/dense_s50

# fix3-A: 새 OUT에서 한 명령으로 smoke 가능하게 core arm 자동 생성
BOOTSTRAP_CORE=${BOOTSTRAP_CORE:-0}
if [[ "$BOOTSTRAP_CORE" == "1" && ! -d "$DREF" ]]; then
  core() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT \
           --limit $N --steps 50 "${PCARG[@]}" "$@"; }
  core --method dense --tag dense_s50
  core --method dense --steps 30 --tag dense_s30
  core --method reuse --cache-period 2 --dense-tail 4 --tag reuse_c2_t4
  core --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 \
       --dense-tail 4 --dual-sparse --kv-cache --tag mbd_c2_r03_t4_dualkv
  core --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 \
       --dense-tail 4 --kv-cache --tag mbd_c2_r03_t4_kv
  for D in dense_s30 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
    python -m eval.region_metrics --run $OUT/$D --ref $DREF --manifest $MAN \
      --out $OUT/$D/metrics.json
  done
fi

# #9: 재사용할 기존 run들이 같은 config인지 먼저 검증 (다르면 비교 무효)
python -m tools.validate_run_compat --manifest "$MAN" --limit "$N" --guidance 30 \
  $DREF $OUT/dense_s30 $OUT/reuse_c2_t4 $OUT/mbd_c2_r03_t4_dualkv $OUT/mbd_c2_r03_t4_kv

# ---------- A) CONTROL: selector ablation (동일 c/r/tail/backend) ----------
ctrl() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
         --steps 50 --cache-period 2 --ratio 0.3 --dense-tail 4 --dual-sparse --kv-cache \
         "${PCARG[@]}" "$@"; }
ctrl --method uniform_grid      --tag ctrl_uniform_grid_c2_r03
ctrl --method contiguous_block  --tag ctrl_contiguous_block_c2_r03
ctrl --method cache_sparse --selector random --tag ctrl_random_c2_r03
ctrl --method cache_sparse --selector mask   --tag ctrl_maskonly_c2_r03
# (mbd/oracle은 Stage 5에 이미 있음 → 표에서 재사용)

# ---------- B) ADAPTED temporal-threshold: wall-matched sweep ----------
for TH in 0.03 0.05 0.08 0.10 0.15 0.20 0.30; do
  python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N --steps 50 \
    --method temporal_thresh --cache-period 2 --dense-tail 4 --teacache-thresh $TH \
    "${PCARG[@]}" --tag adapt_thresh_t${TH/./}
  python -m eval.region_metrics --run $OUT/adapt_thresh_t${TH/./} --ref $DREF \
    --manifest $MAN --out $OUT/adapt_thresh_t${TH/./}/metrics.json
done

for D in ctrl_uniform_grid_c2_r03 ctrl_contiguous_block_c2_r03 \
         ctrl_random_c2_r03 ctrl_maskonly_c2_r03; do
  python -m eval.region_metrics --run $OUT/$D --ref $DREF --manifest $MAN --out $OUT/$D/metrics.json
done
# #5: threshold sweep에서 headline(12.12s)에 wall-match되는 점 자동 선택 + 중복 제거
python -m eval.wallmatch --runs $OUT/adapt_thresh_t* --target-wall 12.12 \
  --out $OUT/table_thresh_sweep.md

python -m eval.assemble --runs $OUT/ctrl_uniform_grid_c2_r03 \
  $OUT/ctrl_contiguous_block_c2_r03 $OUT/ctrl_random_c2_r03 $OUT/ctrl_maskonly_c2_r03 \
  $OUT/mbd_c2_r03_t4_dualkv \
  --out $OUT/table_control.md --csv $OUT/pareto_control.csv

# ---------- #8 SANITY: eval 시작 전 필수 검사 ----------
python -m tools.sanity_eval_sets --runs $DREF $OUT/dense_s30 $OUT/reuse_c2_t4 \
  $OUT/mbd_c2_r03_t4_dualkv $OUT/mbd_c2_r03_t4_kv \
  --manifest $MAN --resolution $RES --dense-ref $DREF

# ---------- C) real-COCO FID/KID + CLIPScore (N=500 arms) ----------
for D in dense_s50 dense_s30 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
  python -m eval.kid --run $OUT/$D --manifest $MAN --resolution $RES \
    --dense-ref $DREF --out $OUT/$D/fidkid.json || \
    { echo "FID skipped (pip install clean-fid)"; break; }
  python -m eval.clipscore --run $OUT/$D --manifest $MAN --dense-ref $DREF \
    --out $OUT/$D/clip.json || { echo "CLIP skipped (pip install open_clip_torch)"; break; }
done



# fix3-B: 모든 arm의 FID eval set이 정확히 동일했는지 최종 assert
python - << 'PY'
import json, glob, sys, os
out = os.environ["OUT"]
hs = {}
for p in glob.glob(f"{out}/*/fidkid.json"):
    d = json.load(open(p))
    hs[p.split("/")[-2]] = (d.get("eval_set_hash"), d.get("eval_set_size"))
if len(set(hs.values())) > 1:
    print("EVAL-SET MISMATCH:", hs); sys.exit(1)
print(f"eval-set identity OK: {len(hs)} arms, "
      f"hash={next(iter(hs.values()))[0]} size={next(iter(hs.values()))[1]}")
PY
