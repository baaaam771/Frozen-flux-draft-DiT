#!/usr/bin/env bash
# Stage 13-B (총평 3순위, 중비용): dense step curve를 실측점으로 촘촘히.
# 목적: Dense@wall 선형보간 의존 제거 — 각 sparse arm의 wall 근처에 실제
# dense 측정점을 둔다. sparse wall(6.61/8.24/8.90/11.33/12.12/13.44s)에
# 대응: step {20, 25, 27, 34, 37, 41} (30/40/50은 기존 결과 재사용).
#
# 2-pass: (1) seed0로 6점 curve 탐색 → (2) 핵심 4점만 3-seed 확정.
# scheduler는 공식 set_timesteps(N) 경로 그대로 (truncation 아님) —
# cached_flux_fill --method dense --steps N이 이미 그 경로를 사용.
# 예상 소요: pass1 ~4h + pass2 ~5h.
#
#   OUT=/mnt/HDD_12TB/bam_ki/flux_fill/stage9_n100 N=100 \
#   MAN=$PWD/data/coco_manifest_1024.json \
#   PC=/mnt/HDD_12TB/bam_ki/flux_fill/prompt_cache \
#     bash scripts/run_stage13_dense_curve.sh
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?예: .../stage9_n100}
N=${N:-100}; PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
# seed0 ref: OUT에 dense_s50이 있으면 그것, 없으면 DREF로 명시 지정
# (예: DREF=/mnt/HDD_12TB/bam_ki/flux_fill/stage9_n100/dense_s50)
DREF=${DREF:-$OUT/dense_s50}
test -d "$DREF" || DREF=/mnt/HDD_12TB/bam_ki/flux_fill/stage9_n100/dense_s50
test -d "$DREF" || { echo "FATAL: seed0 ref 없음 — DREF=<dense_s50 경로> 지정"; exit 1; }
echo "seed0 ref: $DREF"
# seed-offset run의 ref는 반드시 같은 seed의 dense_s50 (기존 3-seed 프로토콜)
SEED_REF_BASE=${SEED_REF_BASE:-/mnt/HDD_12TB/bam_ki/flux_fill/out_final}

run() { local tag=$1; shift
  test -f "$OUT/$tag/run.json" || \
    python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
      "${PCARG[@]}" "$@" --tag "$tag"; }
met() {
  if [ -f "$OUT/$1/metrics.json" ]; then
    # 과거 잘못된 ref로 n=0이 저장됐으면 삭제 후 재계산
    python - "$OUT/$1/metrics.json" << 'PY' || rm -f "$OUT/$1/metrics.json"
import json, sys
m = json.load(open(sys.argv[1]))
if not (m.get("n", 0) > 0 and "mask_lpips_to_ref" in m.get("aggregate", {})):
    print(f"[recompute] {sys.argv[1]} (empty metrics from wrong ref)")
    sys.exit(1)
PY
  fi
  test -f "$OUT/$1/metrics.json" || \
    python -m eval.region_metrics --run $OUT/$1 --ref ${2:-$DREF} --manifest $MAN \
      --out $OUT/$1/metrics.json
}

# ---------- pass 1: seed 0, 6점 탐색 ----------
for S in 20 25 27 34 37 41; do
  run dc_s${S} --method dense --steps $S
  met dc_s${S}
done

# ---------- pass 2: sparse arm 최근접 4점 {25, 27, 37, 41}만 3-seed ----------
# (20은 최저 budget 참고점, 34는 35≈34.5 반올림 대안으로 seed0만 유지)
for S in 25 27 37 41; do
  for SEED in 1000 2000; do
    SREF=$SEED_REF_BASE/seed${SEED}/dense_s50
    test -d "$SREF" || { echo "[skip] $SREF 없음 — seed${SEED} 생략"; continue; }
    run dc_s${S}_seed${SEED} --method dense --steps $S --seed-offset $SEED
    met dc_s${S}_seed${SEED} "$SREF"
  done
done

# ---------- 집계 + nearest-dense 직접 비교(ΔQ/Δt) ----------
python -m eval.assemble --runs $OUT/dc_s* --out $OUT/table_dense_curve.md
python - << 'PY'
import json, glob, os, statistics
out = os.environ["OUT"]

SPARSE_BASE = os.environ.get("SPARSE_BASE",
                             "/mnt/HDD_12TB/bam_ki/flux_fill/stage9_n100")

def load(tag, base=None):
    runs = sorted(glob.glob(f"{base or out}/{tag}*/metrics.json"))
    q = []
    w = []
    for p in runs:
        agg = json.load(open(p)).get("aggregate", {})
        if "mask_lpips_to_ref" not in agg:
            print(f"[warn] skip {p} (empty metrics — wrong ref?)")
            continue
        r = json.load(open(os.path.dirname(p) + "/run.json"))
        rows = [x for x in r["rows"] if not x.get("warmup")]
        q.append(agg["mask_lpips_to_ref"])
        w.append(statistics.median(x["wall_s"] for x in rows))
    return q, w

# 실측 dense 점들 (기존 30/40/50 + 신규)
dense = {}
for s in (20, 25, 27, 30, 34, 37, 40, 41, 50):
    tags = {30: "dense_s30", 40: "dense_s40", 50: "dense_s50"}.get(s, f"dc_s{s}")
    qs, ws = load(tags, base=SPARSE_BASE if s in (30, 40, 50) else None)
    if qs:
        dense[s] = (statistics.mean(qs), statistics.mean(ws), len(qs))

# sparse arm (기존 결과에서)
sparse_tags = ["reuse_c2_t4", "mbd_c2_r03_t4_dualkv", "mbd_c2_r03_t4_kv"]
lines = ["| sparse arm | wall(s) | LPIPS | nearest dense | dense wall | dense LPIPS | ΔQ | Δt(s) |",
         "|---|---|---|---|---|---|---|---|"]
for tag in sparse_tags:
    qs, ws = load(tag, base=SPARSE_BASE)
    if not qs:
        continue
    q, w = statistics.mean(qs), statistics.mean(ws)
    s_near = min(dense, key=lambda s: abs(dense[s][1] - w))
    dq, dw, _ = dense[s_near]
    lines.append(f"| {tag} | {w:.2f} | {q:.4f} | dense-{s_near} | {dw:.2f} "
                 f"| {dq:.4f} | {q - dq:+.4f} | {w - dw:+.2f} |")
lines.append("")
lines.append("| dense steps | LPIPS (mean over seeds) | wall(s) | seeds |")
lines.append("|---|---|---|---|")
for s in sorted(dense):
    q, w, n = dense[s]
    lines.append(f"| {s} | {q:.4f} | {w:.2f} | {n} |")
open(f"{out}/table_dense_direct.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
echo "done -> $OUT/table_dense_curve.md, $OUT/table_dense_direct.md"