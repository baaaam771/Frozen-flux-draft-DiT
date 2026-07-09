# flux_fill_sparse — FreqSpec-Cache-FLUX

Frozen FLUX.1 Fill [dev] target + mask/frequency-aware token routing +
depth-aligned target cache + selective image-token refresh + optional draft.

전체 계획·근거·Gate 정의는 **PLAN.md** 참조. 아래는 실행 순서만.

```bash
# 0) 환경
conda create -n fluxspec python=3.11 -y && conda activate fluxspec
pip install -r requirements.txt
hf auth login                       # FLUX.1-Fill-dev 라이선스 동의 필요

# 1) CPU 사전 검증 (모델 불필요 — 지금 바로 실행 가능)
export PYTHONPATH=.
python tests/test_token_mapping.py
python tests/test_selectors.py
python tests/test_sparse_math_mock.py    # sparse 수식 fresh-cache exactness (float64)

# 2) 벤치마크 고정 (COCO val2017)
python - << 'PY'
from data.dataset import build_manifest
build_manifest("/mnt/HDD_12TB/bam_ki/datasets/coco2017", "data/coco_manifest.json",
               n=500, resolution=512)
PY
python -m data.prompt_cache --manifest data/coco_manifest.json \
  --out /mnt/HDD_12TB/bam_ki/flux_fill/prompt_cache

# 3) Stage 0-2 gates (GPU) — Gate ladder: B0 -> B1 -> B2 (실패 시 실험 중단)
bash scripts/run_stage0_smoke.sh sample.png sample_mask.png "a red sports car"
bash scripts/run_stage1_dense.sh sample.png sample_mask.png "a red sports car"   # B1 (gate_a)
bash scripts/run_stage2_cache.sh                                                 # B0 + B2

# 4) Stage 3-5: Q1 heterogeneity -> PoC -> selector ablation -> main table
MAN=data/coco_manifest.json PC=/mnt/HDD_12TB/bam_ki/flux_fill/prompt_cache \
  bash scripts/run_stage3_selectors.sh

# 5) Stage 6: draft (teacher dump -> router 학습 -> mbfd_draft 평가)
MAN=data/coco_manifest.json bash scripts/run_stage4_drafts.sh

# 6) Stage 7-8: 3 seeds x 전체 suite -> mean±std 테이블 + Pareto CSV + latency
# 실행 규모 tier (기본값은 개발용):
#   Development: N=500  (기본)      MAN=data/coco_manifest.json
#   Main:        N=5000            5k manifest 필요
#   Final:       N=10000           10k manifest 필요
MAN=data/coco_manifest.json bash scripts/run_stage5_final.sh          # dev
N=5000  MAN=data/coco_manifest_5k.json  bash scripts/run_stage5_final.sh   # main
N=10000 MAN=data/coco_manifest_10k.json bash scripts/run_stage5_final.sh   # final
```

## 주의사항
- `token_selectors/` 는 원 계획의 `selectors/` — stdlib `selectors` 모듈과의
  import 충돌 때문에 개명 (torch가 subprocess 경유로 stdlib selectors를 import함).
- 모든 clean estimate는 `utils/flow_math.clean_estimate` (x̂0 = z − σ·v)만 사용.
  DDPM 공식 사용 금지 (rectified flow).
- Gate ladder(B0 block-equiv -> B1 full-dense-equiv -> B2 fresh-cache) 중 하나라도
  실패하면 어떤 실험도 진행 금지 (DACE 규칙).
- block>1 실행 시 실제 refresh 비율은 요청값과 다름 — 테이블의 `r(actual)` 사용.
- "30% 토큰 실행" ≠ "30% compute": dual stream 전체 + single K/V가 dense로 남음.
  전체 transformer MAC 추정치(`MACratio(est)`)와 측정 latency를 항상 분리 보고.
- argparse help 문자열에 `%` 금지 (badly formed help string) — 이 repo는 사용 안 함.
- Router teacher dump 용량: 512², 50 steps 기준 약 13 MB/장
  (200장 ≈ 2.6 GB, 5k ≈ 65 GB, 10k ≈ 130 GB). 먼저 200~1,000장으로 router를
  학습해 AUROC를 확인한 뒤에만 확대할 것.
- Gate B0/B2의 fp32 tol은 1e-5 (SDPA kernel 선택에 따른 reduction 오차 여지);
  bf16은 3e-2 / 2e-2. 실제 측정값을 보고 조정.
