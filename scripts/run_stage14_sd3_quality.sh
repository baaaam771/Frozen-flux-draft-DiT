#!/usr/bin/env bash
# Stage 14 (총평 1순위, 고비용): SD3.5-Large 실제 품질-속도 4-arm.
# 인과관계 재현이 목적: cost-floor 예측(all-dual에서 naive=dense) →
# lever 적용(dualkv) → frontier 개선. selector transfer는 주장하지 않음
# (mask 없는 t2i — delta selector만; 논문 claim 범위와 일치).
#
# 실행 순서가 중요: dualkv를 먼저 돌려 wall을 얻고, dense_matched의
# step 수를 그 wall에 자동 매칭한다 (tools/sd3_quality.py 내장 로직).
# 예상 소요 (N=100, 1024², CFG 2-batch): dense 28step ~0.3s/step ->
# arm당 15~25분, 총 ~1.5h + eval ~10분.
#
#   OUT=/mnt/HDD_12TB/bam_ki/flux_fill/stage14_sd3q \
#   MAN=$PWD/data/coco_manifest_1024.json N=100 \
#     bash scripts/run_stage14_sd3_quality.sh
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
OUT=${OUT:?예: .../stage14_sd3q}; MAN=${MAN:-data/coco_manifest_1024.json}
N=${N:-100}; RES=${RES:-1024}; STEPS=${STEPS:-28}
mkdir -p "$OUT"

# arm 순서: ref -> dualkv(핵심) -> reuse -> matched(자동 step)
python -m tools.sd3_quality --manifest $MAN --out $OUT --limit $N \
  --steps $STEPS --resolution $RES --arms dense_ref
python -m tools.sd3_quality --manifest $MAN --out $OUT --limit $N \
  --steps $STEPS --resolution $RES --arms dualkv
python -m tools.sd3_quality --manifest $MAN --out $OUT --limit $N \
  --steps $STEPS --resolution $RES --arms reuse
# dense_matched: dualkv/dense_ref의 run.json에서 wall을 읽어 step 산출
MATCH=$(python - << 'PY'
import json, os, statistics
out = os.environ["OUT"]
def med(arm):
    r = json.load(open(f"{out}/{arm}/run.json"))["rows"]
    if not r:
        raise SystemExit(f"FATAL: {arm}/run.json has no wall rows — "
                         f"rm -rf {out}/{arm} 후 해당 arm 재생성 필요")
    return statistics.median(x["wall_s"] for x in r)
steps = int(os.environ.get("STEPS", 28))
m = max(4, round(steps * med("dualkv") / med("dense_ref")))
print(m)
PY
)
echo "[dense_matched] steps=$MATCH"
python -m tools.sd3_quality --manifest $MAN --out $OUT --limit $N \
  --steps $STEPS --matched-steps $MATCH --resolution $RES \
  --arms dense_matched

# 집계
pip show open_clip_torch > /dev/null 2>&1 || \
  pip install open_clip_torch --quiet || true
python -m tools.sd3_quality_eval --out $OUT --manifest $MAN --limit $N
echo "done -> $OUT/table_sd3_quality.md"