# FreqSpec-Cache-FLUX 구현 계획안

**목표 시스템**

```
Frozen FLUX Fill target
+ mask/frequency-aware token routing      (FreqSpec: WACV #415, One-Verifier)
+ depth-aligned target cache              (DACE: AAAI 2027)
+ selective image-token refresh           (본 프로젝트의 신규 execution layer)
+ optional plug-in draft                  (selector / temporal correction)
```

세 편의 선행 연구가 각자 제공하는 것:

| 선행 연구 | 가져오는 것 | 이 repo에서의 위치 |
|---|---|---|
| FreqSpec-Inpaint (WACV #415) | mask/boundary/frequency saliency prior, patch-wise 판단 | `token_selectors/{mask,boundary,frequency}.py` |
| One Verifier (task-agnostic) | Ω를 바꿔도 동일한 acceptance core → 여기서는 Ω = FLUX packed image-token grid | `token_selectors/combo.py`, rank-norm 결합 |
| DACE (AAAI) | depth-aligned anchor cache, fresh-cache exactness, anchor/delta selector, r=0 anchored reuse baseline | `models/flux_cache.py`, `models/flux_sparse_transformer.py`, `token_selectors/delta.py` |

DACE의 핵심 교훈을 그대로 계승한다:
1. **Frozen context는 실패한다** → easy 토큰은 절대 depth-m 상태로 얼리지 않고,
   같은 depth의 anchor 상태(depth-correct, time-stale)로 대체한다.
2. **Fresh cache exactness가 게이트다** → cache가 신선하면 sparse pass == dense pass
   (max|Δ|≈0, bf16 rounding 제외)가 나와야만 실험을 진행한다.
3. **r=0 anchored reuse가 가장 강한 draft-free baseline** → 모든 r>0 결과는
   r=0과 reduced-step dense 곡선 위에서 해석한다.
4. **DACE의 two-factor test**: temporal change가 (i) 크고 (ii) 공간적으로 집중되어야
   selective correction이 이긴다. Inpainting은 masked region이 빠르게 변하고
   context는 거의 안 변하므로 DACE가 예측한 "natural deployment"이며,
   이 가설 검증 자체가 논문의 1차 기여가 된다 (`eval/heterogeneity.py`).

---

## FLUX.1 Fill [dev] 구조 전제

- 12B rectified-flow transformer, `FluxFillPipeline` (diffusers)
- dual-stream 19 blocks (`transformer_blocks`) + single-stream 38 blocks
  (`single_transformer_blocks`), hidden 3072
- in_channels 384 = packed latent 64 + masked-image latent 64 + packed mask 256
- VAE 8× 압축 후 2×2 packing → 토큰 그리드 (H/16 × W/16)
- scheduler: FlowMatchEulerDiscrete (dynamic shift), 모델 출력은 velocity v
- clean estimate: x̂0 = z_t − σ_t · v  (DDPM 공식 사용 금지)

**첫 구현 원칙 (안전 우선)**
- dual-stream 19 blocks: dense 유지
- single-stream 38 blocks: text 토큰은 항상 fresh(=query), hard image 토큰만 fresh,
  easy image 토큰은 same-depth anchor cache가 K/V context로만 참여
- easy 토큰의 최종 prediction: anchor의 target prediction 재사용 (DACE r=0/부분 correction)

---

## Stage 계획 ↔ 코드 매핑

| Stage | 내용 | 실행 | Gate |
|---|---|---|---|
| 0 | 라이선스, 공식 baseline 재현, deterministic seed | `scripts/run_stage0_smoke.sh` → `samplers/dense_flux_fill.py --official` | 같은 seed 반복 시 출력 동일 |
| 1 | pipeline 분해, custom dense loop | `samplers/dense_flux_fill.py --mode gate_a` (official↔custom↔runner 3중 비교) | **Gate A**: latent/pixel max err ≈ bf16 rounding |
| 2 | block instrumentation + depth-aligned cache | `models/flux_cache.py`, `tests/test_cache_exactness.py` | **Gate B**: fresh-cache max\|Δv\| == 0 |
| 3 | token/mask mapping 검증 | `utils/token_mapping.py`, `tests/test_token_mapping.py`, overlay 저장 | roundtrip 일치 + 시각화 |
| 4 | mask-only selective refresh PoC | `samplers/cached_flux_fill.py --selector mask` | **Gate C/D**: mask 내부 Δ ≫ 외부, mask < random |
| 5 | FreqSpec selector 전체 ablation | `token_selectors/*`, `scripts/run_stage3_selectors.sh` | M+B+Δ vs M+B+F+Δ (frequency 순수 기여) |
| 6 | plug-in draft (selector → correction) | `models/drafts/*`, `training/` (차후), `samplers --draft` | **Gate G**: draft 포함 Pareto 개선 |
| 7 | structured sparsity (2×2, 4×4, mask window) | `token_selectors/combo.py --block-structure` | **Gate F**: wall-clock T_sparse < T_dense |
| 8 | 최종 5k/10k, 3 seeds, latency/VRAM | `eval/assemble.py`, `scripts/run_stage5_final.sh` | **Gate E**: 같은 품질에서 target compute↓ |

필수 baseline (모두 `samplers/`에서 config로 전환):
- dense 50-step / reduced-step {40,30,25,20,15}
- cache-only (r=0 anchored reuse, anchor period c∈{2,3,4,5})
- mask-only refresh / random same-budget / oracle
- previous-step output reuse (naive) vs depth-aligned cache (DACE 비교군)

---

## 연구 질문 → 판정 실험

| 질문 | 실험 | 판정 지표 |
|---|---|---|
| Q1. FLUX Fill에서도 변화가 mask/boundary에 집중되는가 | `eval/heterogeneity.py`: per-token δ_i(t), top-30% share, CV, in/out-mask ratio | in-mask share ≫ area share |
| Q2. FreqSpec prior가 random보다 hard token을 잘 찾는가 | selector sweep, oracle 대비 captured-change / quality | combo < random error, oracle gap |
| Q3. depth-aligned cache + selective refresh로 품질·latency 동시 개선 | Stage 7–8 Pareto | Gate E, F |
| Q4. draft가 FreqSpec+cache 대비 추가 정보를 주는가 | draft selector / temporal correction vs draft-free | Gate G |

---

## 디렉토리

```
flux_fill_sparse/
├── PLAN.md                      # 이 문서
├── configs/                     # baseline / cache / selector / draft yaml
├── data/                        # dataset.py masks.py prompt_cache.py
├── models/
│   ├── flux_fill_loader.py      # 로딩 + 구조 검증(assert) + 메모리 전략
│   ├── flux_cache.py            # FluxAnchorCache (depth-aligned)
│   ├── flux_sparse_transformer.py  # dense/sparse/anchor forward (핵심)
│   └── drafts/cnn_router.py     # plug-in draft (Stage 6, selector-only 먼저)
├── token_selectors/          # (주의: stdlib selectors와 충돌 방지 위해 개명)                   # mask boundary frequency delta draft combo
├── samplers/
│   ├── dense_flux_fill.py       # Stage 0–1: official ↔ custom equivalence
│   └── cached_flux_fill.py      # Stage 4–7: anchor + sparse refresh 루프
├── eval/                        # region_metrics latency heterogeneity assemble
├── tests/                       # Gate A/B + token mapping + scheduler
├── utils/                       # token_mapping seeds flow_math
└── scripts/                     # run_stage0..5
```

## 환경

```bash
conda create -n fluxspec python=3.11 -y && conda activate fluxspec
pip install -r requirements.txt
hf auth login        # FLUX.1-Fill-dev 라이선스 동의 후
```

`requirements.txt`는 diffusers 버전을 고정한다. `flux_fill_loader.py`가 로드 직후
블록 수(19/38)·in_channels(384)·single-block 속성(norm/proj_mlp/attn/proj_out)을
assert하고, 불일치 시 어떤 diffusers 버전을 쓰라는 메시지와 함께 즉시 실패한다.

---

## 리뷰 반영 (2026-07-08 수정 배치)

| # | 문제 | 수정 |
|---|------|------|
| 1 | `anchor_x0`가 z_t + stale v_a 혼합 estimate | `FluxAnchorCache.set_anchor_context()`가 anchor step에서 x̂₀ₐ = z_a − σ_a·v_a 를 정확히 저장; 기존 방식은 `cached_v_current_x0` ablation arm으로 분리 |
| 2 | 실제 FLUX block equivalence gate 부재 | Gate ladder B0(`tests/test_single_block_equivalence.py`) → B1(gate_a) → B2(cache exactness); loader가 `FluxAttnProcessor2_0`/QK RMSNorm/fused-QKV/added-KV를 hard-assert |
| 3 | block selection이 smoothing + token Top-K | `block_hard_easy_split`: block 평균 → **block 단위 Top-K** → 토큰 확장. k = kb·b² 보장, `r(actual)` 기록·보고 |
| 4 | two-factor 중 consequence 미측정 | hetero row에 E_rel = ‖Δv‖²/‖v‖² 추가; `eval.heterogeneity`가 dense sweep의 mask LPIPS 악화량(S_step)과 결합해 2×2 verdict 산출 |
| 5 | `LPIPS_t` 명칭 오류 | 전 metric `*_to_ref`로 개명 (최종 출력 divergence임을 명시) |
| 6 | known-region 평가 부족 | raw + pasted(M·x_model+(1−M)·x_input) 출력 분리 저장; `--manifest`로 원본 입력 대비 `known_psnr_to_input` 측정 |
| 7 | VRAM/latency 측정 오염 | sample별 sync+reset 후 측정 시작, 첫 sample warm-up 플래그(assemble에서 wall 제외), latency는 config별 peak 분리 |
| 8 | Stage 6 골격만 존재 | `dump_router_teacher`(trajectory dump, resumable/atomic) + `training/train_router`(EMA·rolling ckpt·resume·AUROC) + `RouterDraft` + `--draft-ckpt` + 실행 스크립트 |
| 9 | Stage 8 미구현 | `run_stage5_final.sh`: 3 seed offsets × 전체 suite → seed 평균±std 테이블 + Pareto CSV + latency |
| 10 | compute claim 과대 위험 | `estimate_transformer_macs`: dense dual 19 blocks + full-S K/V/norm 비용 포함한 전체 MAC ratio를 stats/latency/테이블에 분리 보고 |

### 3차 리뷰 반영 (2026-07-08)
- **Stable mask seed (필수)**: `data/masks.py`의 builtin `hash()` (프로세스별 salt로 비결정적) → `stable_seed()` (SHA-256 기반)으로 교체. `test_mask_determinism_across_processes`가 서로 다른 `PYTHONHASHSEED`의 두 subprocess에서 mask checksum 일치를 검증. 주의: seed 산출 방식이 바뀌었으므로 이 수정 이전에 생성된 mask/결과와는 호환되지 않음 (아직 본 실험 전이므로 영향 없음).
- **Brush mask 속도**: 전체 H×W distance test → 원 bounding-box stamping (O(r²)/point). 9개 (type×bucket) 생성 40s+ → 0.1s.

### Gate B2 진단 결과 (2026-07-09, 서버 실측)
- B0/B1/scheduler/Stage0 determinism: 전부 0 오차 통과.
- B2 v1(합성 randn 입력) 실패의 원인: **로직 버그 아님**. bisect probe로 특정 —
  P1 full-seq(Sq=S)는 0.0 exact (행 선택/rope/scatter 로직 정확), P1b subset에서
  입력이 bit-identical한 mlp가 256 발산 → 행 수(819 vs 1536)가 다르면 cuBLAS가
  다른 타일링을 선택해 bf16 축약 순서가 달라짐. 합성 랜덤 입력은 분포 밖이라
  dual stream을 지나며 활성값이 폭발(~1e9)했고 상대오차 ~2⁻⁸이 절대오차로 증폭.
- 조치: Gate B2를 실제 이미지/마스크/프롬프트 입력 + **상대오차 기준**
  (bf16 1e-2, fp32 1e-4)으로 재작성. run_stage2_cache.sh가 인자를 받도록 변경.
- 함의: 실제 sampling에서는 in-distribution 활성값이라 shape 의존 bf16 차이는
  상대 1e-3 수준의 무해한 노이즈 (bf16 고유 특성, 모든 kernel에 존재).

### Latency 실측 결론 (2026-07-09)
- Transformer-only (RTX Pro 6000): 512² dense 91.6ms, sparse r0.3 73.6ms (1.24×);
  1024² dense 303ms, sparse r0.15/0.3 = 192/210ms (**1.58×/1.45×**, 이론 MAC의 ~91% 실현).
- Sampler wall은 실측 step latency로 완전히 설명됨 (숨은 오버헤드 무시 가능).
  512²의 한계는 얕은 per-step 절감 + anchor 비중(c=3에서 wall의 ~40%) 구조.
- **결정: headline 해상도 1024² (FLUX native)로 이동** — Stage 3c. 512는 소해상도
  한계 분석으로 부록화. cache VRAM 1024²에서 0.92GB (예측치 일치).

### Stage 3c 결과와 구조 진단 (2026-07-10)
- 1024² matched-wall 승리: reuse_c2 > dense_s25 (0.0588 vs 0.0651), reuse_c3_t4 >
  dense_s20 (0.0761 vs 0.0873). Selector 서열/oracle-gap 유지 (random 0.0718 →
  mbd 0.0461, oracle 0.0406).
- 구조적 한계 확정: est MAC의 r→0 절편 ≈ 0.49 (dual dense ~33% + single K/V full-S)
  때문에 selective refresh는 matched-wall에서 oracle조차 dense 보간에 열세.
- Lever B (anchor K/V cache) 구현: sparse step에서 K/V를 (text+hard)행으로 축소,
  easy K/V는 anchor 동결 (anchor step exact — mock 0.0 검증, 이후 step은 temb
  staleness 근사). est MAC r0.15: 0.563→0.496. `--kv-cache` 플래그, Gate/latency/
  Stage 3d(run_stage3d_kvcache.sh) 완비.
- Lever A (dual-stream image-token sparsification) 계획: dual 33% 바닥 제거 시
  r0.15 총 MAC ~0.25 → mbd_c3_r015 예상 wall ~8.4s에서 dense 보간(~0.063) 대비
  품질 ~0.052로 frontier 첫 진입 전망. dual block manual 재구현 + Gate B0-dual
  필요 — 다음 구현 단계.

### Stage 3d 결과 + Lever A 구현 (2026-07-10)
- Lever B 실측 판정: temb-staleness 품질 비용 ≈ 0 (2/3 운영점에서 오히려 개선 —
  easy K/V가 easy 출력(anchor prediction)과 정합적이 되는 효과로 해석).
  latency 1.87×/1.65× (r0.15/0.3, est의 92~96% 실현). kv-cache는 무조건 켜는 arm.
- c2_r03_t4_kv (13.44s, 0.0269): dense 우세 metric 구간에서도 보간선 아래 —
  refresh arm의 첫 frontier 점.
- Lever A 구현 완료: `_dual_block_dense/_dual_block_sparse` (0.32.2 semantics),
  dual input/KV 기록, `--dual-sparse` 전 배선, Gate B0-dual, Gate B2 dual 모드,
  mock exactness (exact/kv 모두 0.0). est MAC r0.15: 0.563→**0.244** (dual+kv).
- Stage 3e: gate → latency(dual, dual+kv) → 품질 5 운영점 (c3/c2 + 깊은 c5 arm).

### Frontier 확정 (2026-07-11, 1024², N=100)
6.5~13.5s 전 예산대에서 dense_s30(9.78s) 한 점 제외 지배:
reuse_c3_t4(6.56, .0761) / reuse_c2(8.18, .0588) / c2_r015_dualkv(~11.2, .0369) /
c2_r03_dualkv(12.10, .0333) / c2_r05_dualkv(~13.0, .0296) / c2_r03_kv(13.44, .0269).
Latency 공식 수치: dual+kv r0.15 = 92.3ms/step (**3.35×**), r0.3 = 125ms (2.47×);
Gate 전체 fp32 exact (B0-dual 0.0, B2 dual+kv rel 1.2e-6). 관찰: single-kv
staleness ≈ 0(공짜), dual staleness는 유상(+3~24%, 얕은 c/r일수록) → kv-only와
dual+kv가 예산대를 분담; c5는 refresh로도 구제 불가(anchor 간격 하한 존재).
Stage 5 FINAL: 3 seeds × {dense 6단, reuse 3, refresh 5, headline ablation 4,
block 2} = 19 runs/seed, headline 운영점 = c2_r03_t4_dualkv.

### Stage 7 설계 (robustness & analysis, Stage 6 이후)
A) [무료] bucket/type 분해 + refresh-vs-reuse 실패 사례 정량화 (`eval/breakdown.py`) —
   Fig.4 row-2(multimodal completion으로 refresh < reuse인 샘플)의 체계화.
B) guidance {10,50} 민감도 @ headline (자체 ref 재생성으로 공정 비교).
C) 28-step schedule 전이 mini-frontier (FLUX 관례 step 수; tail 3으로 비례 축소).
D) FFHQ 도메인 전이 (`build_manifest_imagedir`, 고정 prompt, 100장 suite).
E) 해상도 latency ladder 768/1536 (절감-해상도 스케일링 주장 보강).
F) [옵션] KID @ headline (clean-fid; 소표본 unbiased — FID는 5k run으로 유예).
G) selection map 덤프(`--dump-selection`) + Fig.5 (선택의 시공간 이동 시각화).
예상 GPU 시간: B ~2.2h, C ~1.3h, D ~1.5h, E ~10분, G ~5분 → 총 ~5.5h.

### Stage 8 (N=500 sanity + KID) 설계
frontier 핵심 5 arm (dense 40/30, reuse_c2_t4, headline dualkv, kv-only)만
N=500 단일 seed로 재실행 — N=100 cherry-pick 공격 방어 + KID(N=500에서
unbiased) 확보. `run_stage8_n500.sh`, ~11h. 판정: N=500 mask-LPIPS가 N=100
3-seed mean의 ±1 std 안이면 표본 안정성 입증 문장 1개 + KID 표를 appendix에.

### Stage 9 (리뷰 대응) 설계
리뷰 P0 두 구멍을 메움:
1. Acceleration baseline (mask-blind, latency-matched): fora(격자 주기), blockcache(연속),
   random, mask-only, teacache(sigma-임계 조건부 dense) — 전부 headline과 같은 c/r/tail/wall.
   핵심 대비: "공간 선택 제거" 순수 대조군. `run_stage9_rebuttal.sh`.
2. 품질평가: CLIPScore(raw, open_clip ViT-B/32) + FID/KID. N=500는 out_n500 재사용,
   5k는 `run_stage9b_5k.sh` (별도 manifest, overnight×2).
Baseline은 sample_one에 method 3종(teacache/fora/blockcache) 추가 + _uniform_baseline_scores.
FORA/blockcache는 우리 anchor 프레임 위 adapted 버전(세부 논문 명시) — mask를 안 본다는 점만 다름.
