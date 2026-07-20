#!/usr/bin/env bash
# Stage 15 (신규 우선순위 A/B/C/D): cost floor를 실측으로 못 박는 패키지.
# GPU 추가 불필요 — 현 서버에서 총 ~1.5h.
#  A) floor curve   : r→0..1 × 4 lever transformer latency (+그림)
#  B) runtime dist. : 평가 샘플들에 걸친 e2e runtime 분포 재집계 (GPU 0분)
#  C) breakdown     : FLUX_PROFILE=1 profiler로 4그룹 비용 분해
#  D) selector/router overhead 마이크로벤치
#
#   OUT=/mnt/HDD_12TB/bam_ki/flux_fill/stage15_cost \
#   E2E_BASE=/mnt/HDD_12TB/bam_ki/flux_fill/stage9_n100 \
#   DRAFT_CKPT=/mnt/HDD_12TB/bam_ki/flux_fill/router_ckpt_1024/router_0060000.pt \
#     bash scripts/run_stage15_cost_evidence.sh
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
OUT=${OUT:?예: .../stage15_cost}; mkdir -p "$OUT"
E2E_BASE=${E2E_BASE:-/mnt/HDD_12TB/bam_ki/flux_fill/stage9_n100}
DRAFT_CKPT=${DRAFT_CKPT:-}

# ---------- A) floor curve (~40분: 4 lever x 9 ratios x 50 iters) ----------
test -f "$OUT/floor_curve_1024.json" || \
  python -m tools.floor_curve --resolution 1024 --iters 50 \
    --out "$OUT/floor_curve_1024.json"
python figures/make_fig_floor_curve.py --data "$OUT/floor_curve_1024.json" \
  --out "$OUT/fig_floor_curve.pdf"

# ---------- B) per-sample end-to-end runtime distribution (GPU 불필요) ----------
for d in dense_s50 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
  test -f "$E2E_BASE/$d/run.json" || {
    echo "missing required run: $E2E_BASE/$d/run.json"; exit 1; }
done
RUNS=(
  "$E2E_BASE/dense_s50"
  "$E2E_BASE/reuse_c2_t4"
  "$E2E_BASE/mbd_c2_r03_t4_dualkv"
  "$E2E_BASE/mbd_c2_r03_t4_kv"
)
if [ -d "$E2E_BASE/naive_c2_r03_t4" ]; then
  RUNS+=("$E2E_BASE/naive_c2_r03_t4")
fi
python -m tools.e2e_runtime_distribution \
  --out "$OUT/e2e_runtime_distribution.md" --runs "${RUNS[@]}"

# ---------- C) profiler breakdown (~20분) ----------
test -f "$OUT/breakdown_1024_r015.md" || \
  FLUX_PROFILE=1 python -m tools.latency_breakdown --resolution 1024 \
    --ratio 0.15 --iters 12 --out "$OUT/breakdown_1024_r015.md"
test -f "$OUT/breakdown_1024_r03.md" || \
  FLUX_PROFILE=1 python -m tools.latency_breakdown --resolution 1024 \
    --ratio 0.3 --iters 12 --out "$OUT/breakdown_1024_r03.md"

# ---------- D) selector/router overhead (~5분) ----------
read -r SPARSE_MS DENSE_MS << EOV
$(python - << 'PY'
import json, os
d = json.load(open(os.environ["OUT"] + "/floor_curve_1024.json"))
pts = d["curves"]["dualkv"]
sp = next(p["median_ms"] for p in pts if abs(p["r_requested"] - 0.3) < 1e-6)
print(f"{sp} {d['dense_ms']}")
PY
)
EOV
DARG=(); [ -n "$DRAFT_CKPT" ] && DARG=(--draft-ckpt "$DRAFT_CKPT")
# headline c2/t4/50step 실측 구성: dense(anchor+tail) 27 / sparse 23
python -m tools.selector_overhead --resolution 1024 --ratio 0.3 \
  --sparse-step-ms "$SPARSE_MS" --dense-step-ms "$DENSE_MS" \
  --steps 50 --num-sparse-steps 23 --num-dense-steps 27 \
  "${DARG[@]}" --out "$OUT/selector_overhead.md"

echo "done -> $OUT/{floor_curve_1024.json, fig_floor_curve.pdf, e2e_runtime_distribution.md, breakdown_*.md, selector_overhead.md}"