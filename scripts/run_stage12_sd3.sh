#!/usr/bin/env bash
# Stage 12: 2nd-MMDiT cost-transfer (P1-1 축소판).
# 순서: analytic 표(GPU 불필요) -> exactness gate -> latency.
# 요구: SD3.5 HF 접근권한 (hf auth login + 모델 페이지 동의).
set -euo pipefail
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MODEL=${MODEL:-stabilityai/stable-diffusion-3.5-large}
OUT=${OUT:?결과 디렉터리}
mkdir -p "$OUT"

python -m tools.mmdit_cost_model --arch flux-fill sd35-large \
  --resolution 1024 --ratios 0.0 0.15 0.3 --out "$OUT/cost_transfer_1024.md"
python -m tools.mmdit_cost_model --arch flux-fill sd35-large \
  --resolution 512 --ratios 0.0 0.15 0.3 --out "$OUT/cost_transfer_512.md"

python -m tools.sd3_exactness --model-id "$MODEL" --resolution 512 \
  --ratio 0.15 2>&1 | tee "$OUT/sd3_exactness.log"
# (exactness는 실패 시 스스로 nonzero exit; pipefail이 tee 뒤에서도 전파)

python -m tools.sd3_latency --model-id "$MODEL" --resolutions 512 1024 \
  --ratios 0.15 0.3 --iters 20 --out "$OUT/sd3_latency.md"