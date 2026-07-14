#!/usr/bin/env bash
# Stage 9b: 5k real-COCO FID/KID (P0 최종). — 보호장치 Stage 9 동급으로 강화:
#  * arm 단위 resume (run.json 있으면 skip) + 이미지 단위 resume (--skip-existing)
#  * 생성 후 validator (provenance/manifest/limit 상호 검증)
#  * sanity gate (self-FID, stem-set, composited recon)
#  * eval-set hash 동일성 assert
# 권장 OUT: 깨끗한 새 디렉터리 (예: .../stage9_5k_final)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
OUT=${OUT:?새 디렉터리 권장 (예: /mnt/HDD_12TB/bam_ki/flux_fill/stage9_5k_final)}
RES=${RES:-1024}; PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")

test -f data/coco_manifest_5k.json || python - << 'PY'
from data.dataset import build_manifest
build_manifest("/mnt/HDD_12TB/bam_ki/datasets/coco2017",
               "data/coco_manifest_5k.json", n=5000, resolution=1024)
PY
python - << 'PY'
import json
from collections import Counter
m = json.load(open("data/coco_manifest_5k.json"))
c = Counter((it["bucket"], it["mask_type"]) for it in m["items"])
print(f"5k manifest: {len(m['items'])} items")
for k in sorted(c):
    print(f"  {k[0]}/{k[1]}: {c[k]}")
PY
MAN=data/coco_manifest_5k.json; DREF=$OUT/dense_s50

# ---------- 생성 (arm 단위 + 이미지 단위 resume) ----------
r5() { local tag=$1; shift
  if [[ -f "$OUT/$tag/run.json" ]]; then
    echo "skip $tag (완료됨)"; return
  fi
  python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit 5000 \
    --steps 50 --skip-existing "${PCARG[@]}" "$@" --tag "$tag"
}
r5 dense_s50            --method dense
r5 dense_s30            --method dense --steps 30
r5 reuse_c2_t4          --method reuse --cache-period 2 --dense-tail 4
r5 mbd_c2_r03_t4_dualkv --method cache_sparse --selector mbd --cache-period 2 \
                        --ratio 0.3 --dense-tail 4 --dual-sparse --kv-cache
r5 mbd_c2_r03_t4_kv     --method cache_sparse --selector mbd --cache-period 2 \
                        --ratio 0.3 --dense-tail 4 --kv-cache

# ---------- validator (5-arm 상호 검증) ----------
python -m tools.validate_run_compat --manifest "$MAN" --limit 5000 --guidance 30 \
  $DREF $OUT/dense_s30 $OUT/reuse_c2_t4 $OUT/mbd_c2_r03_t4_dualkv $OUT/mbd_c2_r03_t4_kv

# ---------- region metrics ----------
for D in dense_s30 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
  test -f $OUT/$D/metrics.json || \
    python -m eval.region_metrics --run $OUT/$D --ref $DREF --manifest $MAN \
      --out $OUT/$D/metrics.json
done

# ---------- sanity gate (FID/CLIP 전에 반드시) ----------
python -m tools.sanity_eval_sets --runs $DREF $OUT/dense_s30 $OUT/reuse_c2_t4 \
  $OUT/mbd_c2_r03_t4_dualkv $OUT/mbd_c2_r03_t4_kv \
  --manifest $MAN --resolution $RES --dense-ref $DREF

# ---------- FID/KID + CLIP ----------
for D in dense_s50 dense_s30 reuse_c2_t4 mbd_c2_r03_t4_dualkv mbd_c2_r03_t4_kv; do
  test -f $OUT/$D/fidkid.json || \
    python -m eval.kid --run $OUT/$D --manifest $MAN --resolution $RES \
      --dense-ref $DREF --out $OUT/$D/fidkid.json
  test -f $OUT/$D/clip.json || \
    python -m eval.clipscore --run $OUT/$D --manifest $MAN --dense-ref $DREF \
      --out $OUT/$D/clip.json
done
python -m eval.assemble --runs $OUT/dense_s30 $OUT/reuse_c2_t4 \
  $OUT/mbd_c2_r03_t4_dualkv $OUT/mbd_c2_r03_t4_kv --out $OUT/table_5k.md \
  --csv $OUT/pareto_5k.csv

# ---------- eval-set hash 동일성 assert ----------
python - << 'PY'
import json, os, sys
out = os.environ["OUT"]
names = ["dense_s50", "dense_s30", "reuse_c2_t4",
         "mbd_c2_r03_t4_dualkv", "mbd_c2_r03_t4_kv"]   # 핵심 5 arm 명시
hs = {}
for name in names:
    p = os.path.join(out, name, "fidkid.json")
    if not os.path.isfile(p):
        raise SystemExit(f"missing fidkid.json: {name}")
    d = json.load(open(p))
    hs[name] = (d.get("eval_set_hash"), d.get("eval_set_size"))
if len(set(hs.values())) != 1:
    print("EVAL-SET MISMATCH:", hs); sys.exit(1)
v = next(iter(hs.values()))
expected = len(json.load(open("data/coco_manifest_5k.json"))["items"])
if v[1] != expected:
    print(f"EVAL-SET SIZE MISMATCH: {v[1]} != manifest {expected}")
    sys.exit(1)
print(f"eval-set identity OK: {len(hs)} arms, hash={v[0]} "
      f"size={v[1]}/{expected}")
PY