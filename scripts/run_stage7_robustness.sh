#!/usr/bin/env bash
# Stage 7: robustness & analysis suite (Stage 6 이후)
#  A) [무료] bucket/type 분해 + refresh-vs-reuse 실패 사례 (Fig.4 row-2 유형 정량화)
#  B) guidance 민감도 {10, 50} @ headline  (30은 기존 결과)
#  C) 28-step 세계의 mini-frontier (schedule 전이 — FLUX 관례 step 수)
#  D) FFHQ 전이 (도메인 일반화; 고정 prompt)
#  E) 해상도 latency ladder 768/1536 (절감-해상도 스케일링 주장 보강)
#  F) [옵션] KID @ headline vs ref (KID=1k 미만에서도 unbiased)
#  G) selection map 덤프 (Fig.5)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?set OUT to out_final}; N=${N:-100}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
FFHQ_DIR=${FFHQ_DIR:-/mnt/HDD_12TB/bam_ki/datasets/ffhq_hf/images}
SD=$OUT/seed0

# ---------- A) 무료 분석 ----------
python -m eval.breakdown \
  --runs $SD/mbd_c2_r03_t4_dualkv $SD/reuse_c2_t4 $SD/dense_s30 $SD/mbd_draft_c2_r03_t4_dualkv \
  --manifest $MAN --out $OUT/breakdown.md \
  --failures $SD/mbd_c2_r03_t4_dualkv:$SD/reuse_c2_t4

# ---------- G) selection maps (5장, headline 재실행 소량) ----------
python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit 5 --steps 50 \
  --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
  --dual-sparse --kv-cache --dump-selection --tag selmap "${PCARG[@]}"

# ---------- B) guidance 민감도 ----------
for G in 10 50; do
  python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N --steps 50 \
    --method dense --guidance $G --tag dense_s50_g$G "${PCARG[@]}"
  python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N --steps 50 \
    --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
    --dual-sparse --kv-cache --guidance $G --tag mbd_headline_g$G "${PCARG[@]}"
  python -m eval.region_metrics --run $OUT/mbd_headline_g$G --ref $OUT/dense_s50_g$G \
    --manifest $MAN --out $OUT/mbd_headline_g$G/metrics.json
done

# ---------- C) 28-step schedule 전이 (자체 ref = dense-28) ----------
S28() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
        --steps 28 "${PCARG[@]}" "$@"; }
S28 --method dense --tag s28_dense28
S28 --method dense --steps 17 --tag s28_dense17
S28 --method reuse --cache-period 2 --dense-tail 3 --tag s28_reuse_c2_t3
S28 --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 3 \
    --dual-sparse --kv-cache --tag s28_mbd_c2_r03_t3
for D in $OUT/s28_dense17 $OUT/s28_reuse_c2_t3 $OUT/s28_mbd_c2_r03_t3; do
  python -m eval.region_metrics --run $D --ref $OUT/s28_dense28 --manifest $MAN --out $D/metrics.json
done

# ---------- D) FFHQ 전이 ----------
python - << 'PY'
from data.dataset import build_manifest_imagedir
import os
d = os.environ.get("FFHQ_DIR", "/mnt/HDD_12TB/bam_ki/datasets/ffhq_hf/images")
build_manifest_imagedir(d, "data/ffhq_manifest_1024.json",
    prompt="a high-quality photograph of a person's face", n=100, resolution=1024)
PY
FM=data/ffhq_manifest_1024.json
FF() { python -m samplers.cached_flux_fill --manifest $FM --out $OUT/ffhq --limit $N \
       --steps 50 "$@"; }   # prompt cache는 COCO용이므로 미사용(고정 prompt 1개, 부담 없음)
FF --method dense --tag dense_s50
FF --method dense --steps 30 --tag dense_s30
FF --method reuse --cache-period 2 --dense-tail 4 --tag reuse_c2_t4
FF --method cache_sparse --selector mbd --cache-period 2 --ratio 0.3 --dense-tail 4 \
   --dual-sparse --kv-cache --tag mbd_headline
for D in $OUT/ffhq/dense_s30 $OUT/ffhq/reuse_c2_t4 $OUT/ffhq/mbd_headline; do
  python -m eval.region_metrics --run $D --ref $OUT/ffhq/dense_s50 --manifest $FM --out $D/metrics.json
done
python -m eval.assemble --runs $OUT/ffhq/dense_s30 $OUT/ffhq/reuse_c2_t4 $OUT/ffhq/mbd_headline \
  --out $OUT/ffhq/table_ffhq.md

# ---------- E) 해상도 latency ladder ----------
for R in 768 1536; do
  python -m eval.latency --resolution $R --ratios 0.15 0.3 --dual-sparse --kv-cache \
    --out $OUT/latency_${R}_dualkv.json
done

# ---------- F) KID (옵션: pip install clean-fid) ----------
python -m eval.kid --run $SD/mbd_c2_r03_t4_dualkv --ref $SD/dense_s50 \
  --out $OUT/kid_headline.json || echo "KID skipped (clean-fid 미설치?)"
