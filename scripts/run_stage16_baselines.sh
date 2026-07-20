#!/usr/bin/env bash
# Stage 16 (종합평가2 패키지 B, 2·5순위): mechanism-matched prior baselines.
# 동일 backend·checkpoint·mask·seed·prompt-cache·GPU에서 정책만 교체.
# 명칭 주의: 공식 재현이 아니라 mechanism-matched adaptation이다 — 논문에서
# "BlockCache-style threshold reuse" / "FORA-style fixed-period reuse"로만
# 표기하고 원 시스템의 완전 재현이라 주장하지 않는다.
#   blockcache_delta    : BlockCache-style (per-block delta threshold)
#   blockcache_delta_ma : mask-aware variant
#   blockcache_period   : FORA-style (fixed-period full-block reuse)
#   delta_only          : generic dynamic token pruning (mask 미사용)
#   (TeaCache는 기존 stage10 결과 재사용)
# 2단계: (1) N=50 seed0 sweep으로 각 baseline Pareto 지점 파악
#        (2) 선정 arm만 N=300 확정 — 동일 300 sample의 paired comparison
#            (single seed offset; "3-seed"가 아님. 3-seed가 필요하면
#            N=100 x SEED_OFFSET {0,1000,2000}으로 별도 실행할 것)
#
#   OUT=/mnt/HDD_12TB/bam_ki/flux_fill/stage16_baselines N1=50 N2=300 \
#   MAN=$PWD/data/coco_manifest_1024.json \
#   PC=/mnt/HDD_12TB/bam_ki/flux_fill/prompt_cache \
#   DREF=/mnt/HDD_12TB/bam_ki/flux_fill/stage9_n500/dense_s50 \
#     bash scripts/run_stage16_baselines.sh
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
OUT=${OUT:?}; MAN=${MAN:-data/coco_manifest_1024.json}
N1=${N1:-50}; N2=${N2:-300}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
DREF=${DREF:?dense_s50 경로 (N=300 final이면 stem 300개 이상 필요 — n500 권장)}
mkdir -p "$OUT"
# reference가 final N을 커버하는지 사전 검증 (edit_16 §4)
NREF=$(find "$DREF" -maxdepth 1 -name '*.png' | wc -l)
if [ "$NREF" -lt "$N2" ]; then
  echo "FATAL: DREF에 png ${NREF}개 < N2=${N2} — stage9_n500/dense_s50 등"
  echo "       ${N2}개 이상을 포함하는 reference를 지정하세요"; exit 1
fi
python - << 'PY'
import json, os
man = os.environ.get("MAN", "data/coco_manifest_1024.json")
ref = os.environ["DREF"]; n2 = int(os.environ.get("N2", 300))
items = json.load(open(man))["items"][:n2]
exp = {os.path.splitext(os.path.basename(str(x["sample_id"])))[0]
       for x in items}
act = {os.path.splitext(f)[0] for f in os.listdir(ref) if f.endswith(".png")}
missing = sorted(exp - act)
print(f"[ref-check] expected {len(exp)}, available {len(exp & act)}, "
      f"missing {len(missing)}")
assert not missing, f"reference에 없는 stem: {missing[:10]}"
PY

run() { local tag=$1; local n=$2; shift 2
  test -f "$OUT/$tag/run.json" || \
    python -m samplers.cached_flux_fill --manifest $MAN --out $OUT \
      --limit $n "${PCARG[@]}" "$@" --tag "$tag"; }
met() { test -f "$OUT/$1/metrics.json" || \
  python -m eval.region_metrics --run $OUT/$1 --ref $DREF --manifest $MAN \
    --out $OUT/$1/metrics.json; }

# ---------- 1단계: N=50 sweep ----------
# blockcache delta threshold: 낮을수록 재사용 적음(느림·정확) — 5점
for TH in 0.015 0.02 0.04 0.06 0.10 0.15; do
  run bc_delta_th${TH} $N1 --method blockcache \
    --blockcache-policy delta_threshold --blockcache-thresh $TH
  met bc_delta_th${TH}
  run bc_deltaMA_th${TH} $N1 --method blockcache \
    --blockcache-policy delta_threshold --blockcache-thresh $TH \
    --blockcache-mask-aware
  met bc_deltaMA_th${TH}
done
# FORA-style fixed period: 2/3/4
for P in 2 3 4; do
  run bc_period_p${P} $N1 --method blockcache \
    --blockcache-policy fixed_period --blockcache-period $P
  met bc_period_p${P}
done
# generic dynamic pruning: headline과 동일 lever(dual+KV, tail 4)에서
# selector만 delta_only로 교체 — mask prior의 기여를 정확히 분리하는 비교
for R in 0.3 0.5; do
  run delta_only_r${R}_dualkv $N1 --method cache_sparse \
    --selector delta_only --cache-period 2 --ratio $R --dense-tail 4 \
    --kv-cache --dual-sparse
  met delta_only_r${R}_dualkv
done

# 우리 arm 3종을 동일 N/seed/이미지에서 대조로 (N=50 sweep 내 직접 비교용
# — 기존 frontier 수치는 N=100/3-seed라 sweep과 직접 비교 불가)
run ours_mbd_dualkv $N1 --method cache_sparse --selector mbd \
  --cache-period 2 --ratio 0.3 --dense-tail 4 --kv-cache --dual-sparse
met ours_mbd_dualkv
run ours_mbd_kv $N1 --method cache_sparse --selector mbd \
  --cache-period 2 --ratio 0.3 --dense-tail 4 --kv-cache
met ours_mbd_kv
run ours_reuse $N1 --method reuse --cache-period 2 --dense-tail 4
met ours_reuse

# 태그별 개별 표 (assemble은 blockcache 설정 컬럼을 몰라 config를 병합함)
python - << 'PY'
import glob, json, os, statistics
out = os.environ["OUT"]
rows = ["| tag | thresh/period | mask-LPIPS→ref | wall(s) | evals a/s |",
        "|---|---|---|---|---|"]
for d in sorted(glob.glob(f"{out}/bc_*") + glob.glob(f"{out}/delta_only_*")
                + glob.glob(f"{out}/ours_*")):
    mj, rj = f"{d}/metrics.json", f"{d}/run.json"
    if not (os.path.exists(mj) and os.path.exists(rj)):
        continue
    m = json.load(open(mj))["aggregate"].get("mask_lpips_to_ref")
    r = json.load(open(rj))
    c = r["config"]
    knob = (f"th={c.get('blockcache_thresh')}"
            if c.get("blockcache_policy") == "delta_threshold"
            else f"p={c.get('blockcache_period')}"
            if c.get("method") == "blockcache" else f"r={c.get('ratio')}")
    ws = [x["wall_s"] for x in r["rows"] if not x.get("warmup")]
    ev = (f"{r['rows'][0].get('anchor', '-')}"
          f"/{r['rows'][0].get('sparse', '-')}") if r["rows"] else "-"
    rows.append(f"| {os.path.basename(d)} | {knob} | {m:.4f} "
                f"| {statistics.median(ws):.2f} | {ev} |")
open(f"{out}/table_sweep_per_tag.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
PY
python -m eval.assemble --runs $OUT/bc_* $OUT/delta_only_* \
  --out $OUT/table_sweep.md || true
echo "=== sweep 완료: $OUT/table_sweep.md 에서 headline wall(~12.1s)과"
echo "=== 가장 가까운 지점을 골라 아래 2단계 변수로 재실행하세요:"
echo "===   FINAL_ARMS='bc_delta_th0.06 bc_period_p2 delta_only_r0.3' \\"
echo "===   OUT=... N2=300 bash scripts/run_stage16_baselines.sh"

# ---------- 2단계: 선택 arm N=300 확정 ----------
if [ -n "${FINAL_ARMS:-}" ]; then
  for A in $FINAL_ARMS; do
    CFG=$(python - "$OUT/$A/run.json" << 'PY'
import json, sys
c = json.load(open(sys.argv[1]))["config"]
parts = ["--method", c["method"]]
if c["method"] == "blockcache":
    policy = c["blockcache_policy"]
    parts += ["--blockcache-policy", policy]
    if policy == "delta_threshold":
        parts += ["--blockcache-thresh", str(c["blockcache_thresh"])]
    elif policy == "fixed_period":
        parts += ["--blockcache-period", str(c["blockcache_period"])]
    if c.get("blockcache_mask_aware"):
        parts.append("--blockcache-mask-aware")
elif c["method"] == "cache_sparse":
    parts += ["--selector", c["selector"],
              "--cache-period", str(c["cache_period"]),
              "--ratio", str(c["ratio"]),
              "--dense-tail", str(c.get("dense_tail", 0))]
    if c.get("kv_cache"):
        parts.append("--kv-cache")
    if c.get("dual_sparse"):
        parts.append("--dual-sparse")
elif c["method"] == "reuse":
    parts += ["--cache-period", str(c["cache_period"]),
              "--dense-tail", str(c.get("dense_tail", 0))]
else:
    raise SystemExit(f"FATAL: unknown method {c['method']} — replay 미지원")
print(" ".join(parts))
PY
)
    run ${A}_n300 $N2 $CFG
    met ${A}_n300
  done
  python -m eval.assemble --runs $OUT/*_n300 --out $OUT/table_final.md
  echo "done -> $OUT/table_final.md"
fi