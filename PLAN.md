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

### Stage 9 수정 (리뷰 재지적 반영)
1. FID/KID reference 오류 수정: dense-50 대비(=trajectory fidelity)만이 아니라
   REAL COCO 대비(=생성 품질)를 추가. raw + composited + dense50 3refs 분리 필드.
   eval/kid.py 재작성. sanity: dense50 vs 자기자신 FID≈0 체크 포함.
2. baseline 정직한 명명: fora/blockcache/teacache -> uniform_grid/contiguous_block/
   temporal_thresh (control ablation + adapted, prior-work faithful 아님을 명시).
   두 그룹 분리: A) selector-control(동일 backend), B) adapted temporal(wall-matched sweep).
3. CLIPScore: batched + paired ΔCLIP vs dense-50 + bootstrap 95% CI (full-image는
   known region 지배로 둔감할 수 있음을 논문 명시).
4. 실행 순서: N=100 smoke -> N=500 (out_n500 재사용) -> 5k. Stage 9b는 sanity 후.
남은 P0: 실제 5k real-FID 실행. P1(2nd backbone)은 여전히 미착수 — scope 제한 옵션 유지.

### Stage 9 재수정 (fix.docx 9개 지적 반영)
1. _mask_blind_scores unknown method -> ValueError (silent misclassification 방지).
2. uniform_grid: stride 4->2->1 다중격자 순서 -> 어떤 ratio에서도 공간 균등
   (16/16 coarse cell 커버 검증). '% 7' periodic+random 방식 폐기.
3. contiguous_block: seed로 위치 이동하는 L1 다이아몬드 연속블록 (좌상단 편향 제거).
4. temporal_thresh: threshold-triggered dense를 새 anchor로 기록(방식 A) +
   last_anchor_sigma 추적 (periodic anchor 가정 버그 수정).
5. wallmatch.py: threshold sweep에서 목표 wall 자동 선택 + 중복 execution pattern
   제거. run.json에 thresh_dense/thresh_reuse 카운트 기록.
6. kid.py: 모든 집합을 raw stem 교집합으로 강제, n_raw==n_real==n_dense assert.
7. empty/mismatch 방어 (n_raw==0 -> RuntimeError 등) + eval_set_hash 기록.
8. tools/sanity_eval_sets.py: dense50 self-FID≈0 + stem 동일성 + composited 재구성.
9. tools/validate_run_compat.py: 재사용 run의 manifest/steps/limit/guidance/seed 검사.
실행 순서: validate -> (control/thresh 생성) -> wallmatch -> sanity -> FID/KID/CLIP.

### fix2 반영 (N=12 smoke 전 마지막 4개)
1. wallmatch._pattern: 전 non-warmup row 집계 + 이미지 간 동일성 assert
   (sigma-only 정책이므로 달라지면 정책 버그).
2. wallmatch None-safe: q is not None 비교 + "N/A" 표기 (0.0 유효 처리).
3. provenance: run.json에 manifest_sha256/resolution/versions 저장;
   validator가 sha 우선 비교, 구버전 run은 basename+WARN (out_n500 재사용 가능).
4. 전처리 단일화: data.dataset.load_image_rgb를 loader와 kid._stage_real이 공유
   — real-FID가 전처리 차이를 측정하는 것을 구조적으로 차단.
실행 사다리: N=12 smoke -> N=100 -> N=500 -> 5K. smoke 확인 항목: raw/pasted
개수=12, stem set 동일, self-FID≈0, threshold별 count 상이, uniform/contiguous
!= reuse, validation 통과.

### fix3 반영 + 추가 실험 우선순위
코드 (반영 완료):
1. get_model_provenance: model_id/revision/transformer·scheduler config sha/
   git commit·dirty -> run.json. Validator STRICT_PROV 상호비교 (구run WARN).
2. BOOTSTRAP_CORE=1: 새 OUT에서 core arm 자동 생성 -> smoke 한 명령.
3. Stage 9 말미 eval_set_hash 동일성 assert (전 arm).
4. assemble에 Evals(a/s/d) 컬럼 — anchor/sparse/thresh-dense 실제 계산 횟수.
정책: 최종 논문 표·5K는 새 provenance 형식으로 재생성 권장 (기존 N=500은
smoke/방향 확인용).

추가 실험 우선순위 (fix3 §4):
P0-1 faithful prior baseline 1~2개 (TeaCache/FORA 공식 코드 FLUX Fill 이식
     시도, 불가 시 구조적 이유 문서화) — 다음 구현 대상.
P0-2 human study (100~200 pairs, 3 raters, 질문 3분리, bootstrap CI;
     large box/polygon multimodal subset 별도 분석) — 도구 제작 예정.
P1-2 tail sweep 확장 K∈{0..8} + 조건별 collapse + adaptive tail 대조
     (fixed가 비슷하면 "robust simplification" 주장).
P1-4 budget calibration r∈{0.05..0.5} + mask-relative budget.
P1-3 memory 표 (peak alloc/reserved/cache/activation × res × ratio).
P2-1 router transfer matrix (r0.15/0.5, c3, FFHQ, 768/1536, g10/50 중 3~4개).
P2-2 kernel breakdown (proj/attn/FFN/gather-scatter/cache IO 프로파일).
P1-1 2nd backbone은 인페인팅 ckpt 가용성 확인 후 결정 (없으면 P0-1/P0-2 우선).

### Stage 10: faithful TeaCache (P0-1) — 구현 완료
공식 ali-vilab/TeaCache TeaCache4FLUX 소스 분석 후 runner에 teacache_forward
이식. 보존(verbatim): 첫 dual block norm1 modulated input 판정신호, FLUX poly1d
계수 [498.65, -283.78, 55.86, -3.82, 0.264], 누적 rel-L1 + thresh 판정+리셋,
cnt==0/last 강제 dense, hidden-space residual(x_embed+residual->final head),
prev_mod 매 step 갱신. Adaptation(논문 명시): Fill ckpt(384ch), guidance temb,
계수 transfer(repo 권장 관행). 표기: "TeaCache (adapted to FLUX Fill)".
mock 테스트 2종 통과. run_stage10_teacache.sh: 공식 운영점 {0.25,0.4,0.6,0.8}
sweep -> wallmatch. 주의: skip step 비용 = embed+norm1+final head (수 ms) —
speedup 상한은 dense step 수가 결정.

### fix4 반영 (smoke 실패 원인 = validator 설계 오류 2개)
1. steps를 GLOBAL 키에서 제거 — dense step-reduction baseline은 의도적으로
   다른 step 수 (dense_s30=30). --steps는 그룹 판별용으로만.
2. scheduler provenance 분리: scheduler_base_config_sha256 (runtime 필드
   timesteps/sigmas/num_inference_steps/_step_index/_begin_index 제거 후 hash,
   전 arm 동일해야) + timesteps_sha256/sigmas_sha256 (같은 step 그룹 내에서만
   비교; 30 vs 50 다른 건 정상).
3. bootstrap 조건: 디렉터리 존재 -> run.json 존재.
4. git_dirty WARN 문구에 "최종 표/5K는 clean commit에서 재생성" 명시.
검증: smoke 시나리오(RC 0, WARN만) + 그룹 내 schedule 오염(RC 1) 모두 재현.
기존 stage9_smoke12 core run은 유지·재사용 (구/신 provenance 혼재는 WARN 통과).

### fix5 반영 (경계 조건 3개)
1. validator 성공 메시지/docstring에서 steps 제거 (로그 혼동 방지).
2. run_core_if_missing: core 5 arm을 개별 확인·생성 — 부분 중단 smoke 복구
   가능 (metrics.json도 개별 확인).
3. eval-set hash 검사: fidkid.json 0개면 WARN+skip (FID 패키지 누락 시
   StopIteration 방지).

### Smoke 결과 (성공) + composited 오차 원인 규명
Smoke 전 항목 통과: validator OK, control 4종 (MBD .0349 vs random .0482,
same 27/23/0 counts), thresh sweep 정상 dedup (0.20/0.30->0.15와 동일 패턴),
self-FID 0.0000, fid_vs_dense50 2.7e-05, eval-set hash 일치.
composited recon 오차 16~31의 원인: sanity_eval_sets.py가 원본 재로드 시
default-filter resize(BICUBIC)를 사용 — 생성 파이프라인의 LANCZOS와 불일치.
합성 재현으로 증명 (구 방식 max 32, load_image_rgb 통일 후 0.0). 수정:
sanity도 load_image_rgb 공유 + mask resize NEAREST 명시 + 진단 강화
(mean/p99/>2 비율/경계 집중도, scipy fallback). 파이프라인·저장물은 무결.
N=100: THRESHES="0.03 0.025 0.02 0.015 0.01"로 wall-match 구간 촘촘히.

### N=100 validator 실패 원인 + 수정
scheduler_base_config_sha256을 run 종료 시(마지막 set_timesteps 후) 계산 —
set_timesteps가 config에 step-수 의존 runtime 항목을 남겨 5개 제외 필드로도
오염 잔존 (dense_s50 vs 나머지 불일치). 수정: (1) provenance를 파이프라인
로드 직후(어떤 set_timesteps보다 전) 캡처, schedule hash(timesteps/sigmas)만
종료 시점 값 병합; (2) scheduler_base_config dict 자체 저장 -> validator가
hash 불일치 시 키별 diff 출력. 기존 N=100 core run들은 재검증만 다시.

### Provenance canonicalization (최종 N=500/5K 전 필수)
_cfg_sha에 재귀 canonicalization 적용: dict 키 정렬, set/frozenset -> 정렬
리스트, Path/텐서 스칼라 정규화, separators 고정. 저장용
scheduler_base_config도 동일 canonical 소스 사용. PYTHONHASHSEED {0,1,42}
3-프로세스에서 동일 hash 검증 완료. N=100 patch는 core 5개로 명시 제한
(control/thresh run 덮어쓰기 방지).

### N=500 validator 실패 #2: _use_default_values (해결)
diffusers가 config에 넣는 '_' 프리픽스 내부 메타데이터(_use_default_values)가
set 유래 list라 프로세스마다 순서 상이 -> canonicalize가 list 순서를 보존해
hash 불일치. validator diff 출력이 정확히 범인을 지목 (diff 기능 검증됨).
수정: '_' 프리픽스 키를 base config에서 일괄 제외 (_step_index/_begin_index도
자동 커버). 재현 테스트: 순서 다른 두 config -> 동일 hash 확인.
N=500 core 5 arm 생성·metrics는 이미 완료 — provenance patch로 재사용.

### TeaCache N=50 검증 성공 + fix_t 반영
Sweep 통과: thresh 0.8/0.6/0.4/0.25 -> wall 3.98/4.96/6.26/8.55s, calc
12/15/19/26 단조, maskLPIPS 0.1331/0.1007/0.0709/0.0504. 관찰: TeaCache
최저 thresh(8.55s, 0.0504)가 같은 wall의 anchored reuse(8.9s, 0.0447)보다
열세 — faithful TeaCache도 저예산 tier에서 밀림. 12s(headline) 비교점은
N=500에서 THRESHES에 0.1/0.05 추가로 확보.
fix_t 반영: (1) assemble sig에 teacache_rel_l1 추가 — 4 thresh가 seed처럼
평균되던 버그 수정 (4행 분리 검증), Selector 칸에 rel-L1=x 표시.
(2) stage10에 gen_if_missing — 같은 OUT/N의 dense_s50+MBD 자동 생성
(다른 N 혼합 금지), assemble 경로 실패 해소. (3) wallmatch 헤더
full/thr-dense/reuse로 일반화.

### 5k 스크립트 보호장치 강화 (Stage 9 동급)
1. 이미지 단위 resume: sampler --skip-existing (raw+pasted 존재 시 skip,
   run.json에 resumed/n_skipped_existing 기록; wall 통계는 fresh 샘플만).
2. arm 단위 resume: run.json 존재 시 arm skip.
3. validator + sanity gate + eval-set hash assert 이식.
4. metrics/fidkid/clip도 존재 확인 후 skip (재실행 멱등).
권장 OUT: 새 디렉터리 stage9_5k_final. set -e 유지 — 실패 시 같은 명령
재실행이 안전한 지점부터 이어감.

### fix_5k 반영 (5K 시작 전 필수 2 + 권장 2)
1. sanity_eval_sets.py -> 진짜 gate: stem 불일치/self-FID>=1/recon>=3/검사파일
   0개 시 failed 누적 -> SystemExit(1). set -e가 이제 실제로 멈춤 (검증: 합성
   불일치에서 RC 1).
2. prompt cache: fallback 없음 확인 (load_cached는 miss 시 FileNotFoundError).
   5K 시작 전 manifest 생성 -> python -m data.prompt_cache로 확장 필수
   (기존 500개는 out.exists()로 skip, ~4500개만 추가 인코딩).
3. eval-set assert를 명시적 5 arm으로 — 누락 파일 시 즉시 실패, 이후 다른
   arm의 fidkid 추가에도 오염 안 됨.
4. warmup을 "n_fresh==0" 기준으로 (resume 시 첫 fresh 샘플이 warmup으로
   정확히 표시), 종료 메시지 target/fresh/skipped 분리.

### Resume 통계 명확화 (fix_5k 후속)
1. run.json: timing_scope("fresh_session_only"/"full_run"), n_timing_samples,
   n_output_samples 기록.
2. assemble: resumed run의 Wall에 '*' + 각주("마지막 세션 fresh 평균 —
   latency 주장에는 clean 단일 세션 사용"). 검증: resumed만 별표.
3. prompt_cache 로그: unique/newly cached/already existed 분리 출력.
정책: 5K 표의 wall은 FID 목적상 참고치 — 논문 latency 주장은 N=500 clean
run과 eval/latency.py 격리 측정만 사용.

### 5K 복구 마지막 edge case (해결)
이미지 5000장 저장 완료 후 run.json 직전 사망 -> --skip-existing 재실행 시
rows=[] (fresh=0). assemble 방어: wall=None(표 "-", resume이면 "-*"),
evals=("-","-","-"), CSV mean도 빈 리스트 가드. end-to-end 검증: 표/CSV
정상 출력, quality·Imgs는 metrics 기반이라 영향 없음.

### Human study 도구 (P0-2) 완성
tools/human_study/: prepare_pairs.py(층화: random N + large box/polygon
subset, pair별 좌우 무작위 고정, 라벨 숨김, input+mask 참조 오버레이 생성,
자체완결 study 폴더) + index.html(3질문 분리, 단축키, localStorage 진행저장,
JSON export) + aggregate.py(win/tie/loss + win-rate bootstrap 95% CI +
이미지당 다수결 + subset 분해 + 평가자 일치도). mock 3-rater E2E 검증.
운영: N=500 출력으로 3개 비교(ours vs dense30/dense40/reuse) x (100+50)쌍,
평가자 3인+. dense40이 없으면 stage9_n500에 dense_s40 1 arm 추가 생성 필요.

### Human study blinding 구조화 (운영 리뷰 반영)
라벨을 key.json으로 물리 분리 — 평가자 전달 파일(pairs.json/index.html/img)
어디에도 method 정보 없음 (검증: public label-free assert). key.json은
실험자 보관, aggregate가 study 폴더에서 읽음. RATER_README.txt + serve.sh
(정적 서버 옵션) 동봉. 운영 규칙: study 출력은 repo 밖
(/mnt/.../human_studies/) — 5K arm들의 git_dirty 오염 방지.

### Stage 11 (acceptance polish) 준비 완료
A) P1-4 budget sweep r{0.05,0.1,0.2,0.4,0.5} — sweet spot 폭 증명 (~4h)
B) P1-2 tail K{0..8}\{4} + adaptive-tail{0.02,0.05} 대조 (~6h)
   sampler --adaptive-tail: sparse step에서 anchor 대비 상대에너지 임계 초과
   시 잔여 전부 dense, adaptive_switch_step 기록.
C) P1-3 memory 표: tools/memory_table.py — res x lever별 peak alloc/reserved
   + cache 구성(states/single_kv/dual_kv) 분해, vram_bytes 총합 assert (~1h)
D) P2-1 router transfer 3점: r0.15(+동일 r mbd 짝) / g10 / FFHQ —
   DRAFT_CKPT/G10_REF/FFHQ_REF 환경변수, ckpt 없으면 자동 생략 (~3h)
실행: 5K 종료 후 OUT=stage9_n100 재사용 (ref 공유). run/met 멱등(존재 시 skip).

### Stage 12: 2nd-MMDiT cost-transfer (P1-1 축소판, 리뷰 설계 그대로)
Claim 한정: "cost-model and cost-removal transfer" (method/selector/frontier
전이는 주장하지 않음).
1. tools/mmdit_cost_model.py — 일반화 blockwise MAC (dual+single vs all-dual),
   프리셋 flux-fill/sd35-large/medium. 로컬 실행 결과가 리뷰 예측 재현:
   naive floor FLUX 0.39 vs SD3.5-L 0.67 (all-dual의 K/V 재계산 부담),
   +KV로 양쪽 <0.1, dualkv r0.15 ~0.12. floor의 원인 = block composition.
   주의: 논문 §6.3의 0.49는 dual-dense 보수 변형 — 각주로 정의 명시 예정.
2. models/sd3_sparse_runner.py — SD3 joint block 수동 실행 + naive/kv/dualkv.
3. tools/sd3_exactness.py — official vs manual (bf16 전체 + fp32 블록별),
   fresh-cache kv/dualkv hard-row == dense. in-distribution latent 사용.
4. tools/sd3_latency.py — 512/1024 x lever x r{.15,.3}: median latency,
   speedup, analytic MAC, realization, VRAM.
5. scripts/run_stage12_sd3.sh — analytic -> exactness(FAIL시 중단) -> latency.
요구: SD3.5 HF gated 동의 + hf auth login. 총 ~1-2h (다운로드 별도).

### Stage 12 v2 (P0 6개 전부 반영)
1. cost model 재정의 -> 논문 lever 1:1: naive(보수 정책: dual dense + single
   sparse) FLUX r->0 = 0.486 ~= 논문 0.49 재현, SD3 = 1.000 (all-dual은
   sparse-eligible 블록 없음 -> "dual sparsification이 필수"). dual(Lever A:
   text fully fresh + img KV 2N 재계산) FLUX 0.230/SD3 0.193. dualkv(A+B:
   img KV 캐시 + hard 2k) floor = text 비용 (FLUX 0.111/SD3 0.075, 0 아님);
   FLUX dualkv@r.15 = 0.244 ~= 논문 0.24.
2. runner v2: text 스트림 모든 변형에서 fully fresh (Q/K/V/O/FF), 캐시는
   이미지 K/V만 + hard 행 fresh scatter, easy 행은 depth별 anchor 상태
   (img_states 길이 n+1, final 출력 포함). naive는 runner에서 제외(=dense).
   CPU mock fresh-cache 검증: dual/dualkv 오차 0.
3. exactness v2: official forward로 순차 진행하며 target 블록에서 같은
   in-distribution 입력 비교 (경로 왜곡 제거), transformer.float()를 embed
   전에, rel+max_abs 동시 판정 (fp32 rel<1e-5, bf16 rel<3e-2, fresh-cache
   rel<1e-3), 실패 시 SystemExit(1) 직접.
4. latency: lever = dense/dual/dualkv + naive 행은 "= dense (paper policy)"
   명시. 5. shell: set -euo pipefail + 전 변수 quote + grep 게이트 제거.

### Stage 12: SD3.5 dual_attention_layers (attn2) 지원
runner: _norm1 분기(7-tuple), dense에 attn2(image-only self-attn, FFN 전) +
self_img_kv 기록; sparse에 attn2 hard-Q + (dual: 2N 재계산 / dualkv: 캐시
+hard scatter). exactness helper 동일 반영. cost model: attn2_dense =
n_attn2(4N D²+2N²D), sparse = (2k+kv2)D²+2kND; naive는 attn2 dense 포함.
latency가 config.dual_attention_layers에서 n_attn2 자동 로드 (Large=0,
Medium=13 — 어느 쪽이 와도 안전). mock(attn2 블록 포함) fresh-cache
dual/dualkv 오차 0 재검증.

### Stage 12 최종 보강
1. exactness [2]: full-output rel + hard rel 동시 검사 (fresh cache면 easy
   행도 dense와 동일해야 — 최종 scatter/easy 경로 오류 검출). mock 검증:
   dual/dualkv 모두 full rel 0.
2. latency: dense를 anchor cache 생성 **전에** 측정 (empty_cache +
   reset_peak) -> dense peak VRAM이 순수값. sparse 행은 cache 상주 상태
   (실사용 조건). 표에 각주로 명시.
3. realization = measured speedup x analytic MAC ratio (= measured/predicted),
   기존 `(1/pred) and (sp/(1/pred))` 정리.

### Stage 11 첫 실행 실패 원인 (수정)
--adaptive-tail argparse 정의가 로컬 파일에서부터 누락 — Stage 11 작업 시
치환 anchor 불일치로 silent no-op (컴파일 검사는 통과해 미검출). 교훈:
argparse 추가는 --help 렌더링 assert로 검증해야 함 (이번에 적용, 5개 플래그
전부 확인). 괄호 균형 기반 삽입으로 멀티라인 문장 파손도 방지.

### Stage 11 결과: A+B 완주, C 버그 수정
A(r-sweep 5) + B(tail 7 + adaptive 2) 전 arm fresh=100 완주. C(memory_table)
버그 2개 수정: (1) state.sigmas 없음 -> pipe.scheduler.sigmas[0] 사용,
(2) base(레버 없음) 조합이 sparse_forward의 cache assert에 걸림 -> base는
순수 dense-only 측정(cache=None)으로 재정의, grid는 state.grid(실제
TokenGrid) 사용. API 전수 대조 완료 (dense_forward 반환/finish_anchor 내장/
selector 시그니처). 서버에서 같은 명령 재실행 시 A/B skip -> assemble ->
C -> D 이어짐.

### memory_table 3차 수정 (동적 속성)
FluxAnchorCache의 dual_block_inputs/single_block_kv/dual_block_kv는 dataclass
필드가 아니라 begin_anchor()에서 동적 생성 — base(cache=None) 조합에서 빈
객체 분해 시 AttributeError. getattr(cache, name, []) 안전 접근 + vram_bytes
교차 assert는 kv/dual 셀에서만. CPU 검증: 빈 cache 분해 0, populated 시
vram_bytes 합계 일치.

### Stage 11 표 오염 발견·수정 (adaptive_tail sig)
table_polish.md에서 t0 행이 Seeds=3 — adaptive_tail이 assemble sig에 없어
pol_tail0 + pol_adapt002/005가 seed처럼 병합 (wall 13.34±2.50 비정상 분산이
증거; TeaCache 때와 동일 유형). 수정: sig에 ad_tail 추가, Tail 컬럼에
"ad0.02" 표기 (tail 튜플 인덱싱 파손 우회 — 표기부에서 분기). 3행 분리 검증.
r-sweep/t1–t8은 오염 없음(각 단일 run) — 즉시 논문 반영 가능; t0/adaptive는
서버 assemble 재실행 후 확정.

### Stage 12 OOM 수정 (진단 문서 반영)
증상: fp32 depth 루프에서 94.9GB 소진 (bf16 full [1]과 block0 [1b]는 통과).
원인 3중: (a) fp32 루프의 official block 호출이 no_grad 밖 -> 38블록
autograd 그래프 축적(주범), (b) 전체 transformer.float() = 8B x2 VRAM,
(c) 텍스트 인코더 3개(T5-XXL 포함)+VAE 상주.
수정: exactness main에 @torch.inference_mode(), encode 후 인코더/VAE
CPU offload + empty_cache, fp32는 대상 블록만 승격-검사-즉시복원
(depth progression은 bf16 official). latency에도 동일 가드.
mock 검증: 승격/복원 사이클 dtype 정상.

### Stage 12 gate 임계 수정 — 이번엔 실제 반영 (2차)
1차 시도가 old-string 불일치로 silent no-op (변수명 hard_rel/full_rel 버전이
실제 파일; grep 검증이 head의 exit 0에 가려짐). 교훈 재확인: 치환 후
내용 assert 필수 (적용됨: 신규 임계 존재 + 구 gate 부재 + [2b] 존재 assert).
반영: FULL 3e-3 / HARD 6e-3 (bf16 커널 배치 차이; official-vs-manual 4.6e-3
동규모), threshold 로그 표기, [2b] cache-consistency (dual vs dualkv
rel<1e-6 — fresh cache에서 재계산==캐시의 직접 증거).

### fig2 좌측 라벨 잘림 수정
cm-mathtext/serif에서 tight bbox가 ylabel 폭 과소평가 + pad 0.02 과소.
수정: pad 0.06 시작 + 저장 후 200dpi 렌더의 가장자리 3px 잉크 자동검사,
검출 시 pad +0.05 재저장(최대 4회) — fig1 bbox 자동검사와 동일 철학.
mock 렌더: pad=0.06에서 edges clean.

### 총평(종합 리뷰) 반영 + Stage 13/14 구성
논문 수정 10건 완료: std/variance 표현(3곳+fig1), bit-consistent 정밀화,
staleness hypothesize, 통계 단위(3x100-image seed-offset runs), dense 보간
격하(actual dense-40 전면), 기여 4->3 압축(router optional 격하), abstract
~18% 압축(training-free vs optional router 분리), suppl H FID 프로토콜
상세(clean-fid/raw·composite 정의/CI 근거), baseline applicability 표,
Q 순서 roadmap 문장, "pure pure" 오타. 제목 변경은 사용자 결정 대기.
실험 3종 구성:
- Stage 13-A (scripts/run_stage13_gpu2.sh): 추가 GPU latency — env.json 기록,
  FLUX 4-lever x r{.15,.3} x 해상도, iters 100 (p10 추가), summary 표.
- Stage 13-B (run_stage13_dense_curve.sh): dense {20,25,27,34,37,41} seed0
  탐색 + {25,27,37,41} 3-seed (seed별 dense_s50 ref 필수 — SEED_REF_BASE),
  nearest-dense ΔQ/Δt 직접 비교 표.
- Stage 14 (run_stage14_sd3_quality.sh + tools/sd3_quality[_eval].py):
  SD3.5-L t2i 4-arm (dense_ref 28 / dense_matched 자동 step / reuse c2t4 /
  dualkv c2 r0.3 delta). 계약 검증: timestep 1000-scale 그대로, unpatchify
  exactness [1] 공식 재사용, anchor 상태는 sampler 로컬(v_anchor/prev) —
  SD3AnchorCache에 final_prediction 없음. CPU mock 3-mode 완주 검증.

### sd3_quality 파일명 정규화 (사용자 수정 반영 + 생성측 일관화)
사용자 수정: eval의 items 키를 Path(sample_id).stem으로 정규화 (manifest
sample_id에 확장자 포함 케이스). 동일 가정이 생성측(sd3_quality.py)에도
있어 함께 수정: 저장 파일명 stem + seed는 숫자만 추출(int(sid)의
ValueError 방지). 4개 케이스(순수/확장자/경로/정수) 파일명·seed 동일성
검증 — 확장자 유무와 무관하게 같은 파일명·같은 seed.

### sd3_quality 최종본 = 사용자 서버 버전 채택
splitext(basename()) 정규화 + int(stem) seed — 서버 실행본과 로컬/zip 통일.
(내 버전과 기능 동등; 유일한 차이는 stem에 비숫자 포함 시 int() 실패인데
COCO sample_id는 순수 숫자라 무관.)

### Stage 13-B 실패 원인·수정 (KeyError mask_lpips_to_ref)
사용자가 OUT을 새 디렉터리(stage13_dense_curve)로 잡음 -> $OUT/dense_s50
부재 -> seed0 met이 n=0 metrics 저장(dc_s34 Imgs=0이 증거) -> ΔQ/Δt 집계
KeyError. 생성 14 run은 전부 성공 — 이미지 재생성 불필요.
수정 3중: (1) DREF 부재 시 stage9_n100/dense_s50 fallback + 조기 FATAL,
(2) met()이 기존 n=0 metrics를 감지해 자동 삭제·재계산,
(3) 집계가 빈 aggregate run을 경고 후 스킵 + sparse/dense_s{30,40,50}는
SPARSE_BASE(stage9_n100)에서 로드. 같은 명령 재실행이면 metrics만 재계산.

### Stage 14 1차 실행: dualkv 붕괴 원인·수정
증상: dualkv만 노이즈 (LPIPS .587, CLIP 26.3); 같은 cache의 reuse는 정상
(CLIP 35.9 최고) -> cache 아닌 dualkv 경로 문제. 원인: sparse_forward는
FULL [B,N,out]을 반환하는데 (exactness [2] full-output 검사가 보증) hard
전용 값으로 오인해 scatter_ — PyTorch scatter는 index 크기만큼 src
앞부분(토큰 0..k-1)을 쓰므로 잘못된 값이 hard 위치에 삽입돼 붕괴.
수정: scatter 제거, v = sparse_forward(...) 직접 사용. mock: fresh-cache
dualkv full rel 0.00e+00 (dense와 동일) 재검증.
재실행: dualkv 디렉터리 삭제 후 dualkv -> dense_matched (wall 재매칭) 순.
관찰 메모: 1차에서 reuse가 CLIP 최고(35.87) — SD3 t2i에서 anchored reuse가
매우 강함 (16/12 evals, wall 4.34 vs dense 7.62).

### Stage 14 2차: run.json 덮어쓰기 버그 (empty median)
dualkv 재생성은 성공 (n=100, wall 5.75s). 하지만 resume 시 이미 완료된
arm(dense_ref/reuse)의 run.json을 rows=[]로 덮어써 wall 기록 소실 ->
MATCH 산출 empty median. 수정: resume 시 기존 rows 로드·보존(done_ids),
신규 생성만 append; MATCH med()는 rows 빈 arm에 FATAL+복구 안내 출력.
서버 복구: dense_ref/reuse는 rows가 이미 소실됐으므로 두 arm 삭제 후
재생성 (같은 seed — 이미지 bit-동일 재현, ~55분).

### Stage 15 구성 (신규 우선순위: GPU 추가 없이 cost floor 실측 증거)
A) tools/floor_curve.py + figures/make_fig_floor_curve.py: r {0(=k1 r→0),
   .01,.025,.05,.1,.15,.3,.5,1.0} x {naive,kv,dual,dualkv} transformer
   latency; r=0은 sparse path 전부 실행(k=1) — reuse 우회 금지 요건 충족;
   그림은 analytic 곡선 오버레이 + floor 주석 + edge-ink 자동검사 (mock
   렌더 검증 완료).
B) tools/e2e_variance.py: 기존 run.json wall 분포 재집계 (median/mean±std/
   p10-p90/CV/VRAM) — GPU 0분.
C) models/flux_sparse_transformer.py에 FLUX_PROFILE=1 조건부 record_function
   태깅 5구역 (dual_stream/single_kv_cached/single_kv_recompute/
   sparse_overhead/cache_record/final_head; 평소 nullcontext zero-cost,
   양 모드 회귀 통과) + tools/latency_breakdown.py: 4그룹 CUDA-time 분해
   (dense/naive/kv/dual/dualkv).
D) tools/selector_overhead.py: mbd scoring(delta+rank결합)/rank+topk/
   router(실제 ckpt·시그니처)/gather prep 마이크로벤치 — "unmeasurable"
   대신 ms 수치.
공용화: eval/latency.py에서 load_transformer_only() 추출.
실행: scripts/run_stage15_cost_evidence.sh (총 ~1.5h; B는 즉시).

### Stage 15 리뷰 반영 (필수 3 + 권장 6)
필수 1: e2e_variance -> e2e_runtime_distribution 개명 — "per-sample runtime
  distribution across evaluation samples"로 해석 제한 (반복성 주장 금지 명시).
필수 2: single 블록 내부 세분 태깅 — 3개 helper(_single_block_dense/
  _single_block_sparse/_single_block_sparse_kv)에 single_q_mlp /
  single_kv_projection / single_attention / single_kv_scatter 삽입.
  breakdown GROUPS를 K/V 중심 재편 (dual_blocks / single_kv[projection+
  scatter만] / single_other / overhead / head) — "full-S K/V가 floor"를
  profiler로 직접 증명 가능. 양 모드 회귀 통과.
필수 3: other 제거 — profiled-tag 합 기준 share + clean wall-clock total
  별도 열 (측정 도메인 혼합 금지).
권장: floor_curve에 anchor_record[lever] 실측 + manual_seed(0) + env 메타
  (gpu/torch/cuda/git); "r→0(k=1)" 표기 (출력/그림 축라벨/docstring 경고);
  selector에 실제 _gather_tokens/_scatter_tokens 벤치(+scatter clone 상한
  명시), index_preparation 개명, --num-sparse-steps 필수 인자(헤드라인 23);
  스크립트 RUNS 배열 + 필수 run 사전검증 exit 1.

### Stage 15 2차 리뷰 반영 (3건)
1) scatter 벤치 이중 clone 제거 — _scatter_tokens 내부 clone 1회만
   (runner와 동일 비용), caption도 사실로 갱신.
2) full-sampling 분모 정확화 — --dense-step-ms/--num-dense-steps 추가,
   full = dense_ms*27 + sparse_ms*23 (headline 실측 구성); 스크립트가
   floor_curve JSON에서 dense_ms까지 읽어 전달 (read-split 스모크 검증).
3) latency_breakdown docstring의 옛 태그명(single_kv_cached/recompute) ->
   세분 태그명으로 갱신.

### Stage 15 실행: A/B/C 성공, D scatter off-by-one 수정
성공 결과 (핵심): r→0(k=1) 실측 floor — naive 0.573x, kv 0.476x,
dual 0.372x, dualkv 0.197x (analytic 0.49/0.24와 방향·크기 일치, dualkv는
analytic보다도 낮음); anchor record 1.01-1.08x dense; breakdown에서
naive single_kv 61ms ≈ dense 69ms(잔존) vs kv 39ms(= (1229+512)/4608 예측
그대로), dual_blocks는 dual에서 142ms로 감소 — "K/V가 floor" profiler 증명
완성. breakdown의 profiled 합 > clean total (profiler 계측 오버헤드) —
share 방식 보고가 옳았음을 실증.
D 오류: select_hard_tokens가 block 정렬로 k=1229 반환하는데 fresh를
사전계산 k=1228로 생성 -> scatter index>src. 수정: k_actual=hard.shape[1]
기준으로 fresh 생성 (스모크 통과). selector_overhead만 재실행하면 됨.

### 종합평가2 대응: 3 실험 패키지 구현 (Stage 16/17/18)
약점 3.1/3.2/3.3에 1:1 대응. 우선순위: 18(C) -> 16(B) -> 17(A).
- Stage 18 (패키지 C, GPU 0): tools/mask_local_quality.py — 기존 5k 이미지
  재사용; mask tight-bbox crop FID/KID (margin 32, min 128, size-bucket
  s/m/l), masked-DINO distance (DINOv2 feature pooling, open_clip
  fallback), mask-LPIPS→real (spatial LPIPS + mask pooling). mask는
  manifest spec에서 make_mask로 런타임 재생성 (생성시와 동일 규약).
  bbox/bucket/edge-clamp 스모크 검증.
- Stage 16 (패키지 B): mechanism-matched baselines — (1) delta_only
  selector preset (generic dynamic pruning, mask 미사용), (2) runner에
  blockcache_forward (per-block temporal reuse; delta_threshold=block-
  caching 계열, fixed_period=FORA 계열, mask_weight로 mask-aware variant;
  첫/끝 스텝 강제 계산; mock 3종 통과: exact reuse/period 패턴/mask-aware),
  (3) sampler method "blockcache" + CLI 4인자 (cfg=vars(a)로 run.json 자동
  기록). 스크립트: N=50 sweep(threshold 5점+MA, period 3점, delta_only
  2점) -> FINAL_ARMS 지정 시 N=300 wall-matched 확정.
- Stage 17 (패키지 A): tools/sd3_masked_transfer.py — SD3.5-Large
  controlled masked-generation ("native inpainting"이라 부르지 않음);
  매 스텝 known reinjection z_t=(1-σ)z0+σε (스케줄러 sigmas 사용, FLUX
  공식 복사 금지 요건 충족); mask+boundary+delta selector를 FLUX와 동일
  가중치로 재튜닝 없이; arms 6종 (dense_ref/matched/reuse/refresh
  r015/r03/kv_r03; SD3는 all-joint라 명칭 joint-token refresh). CFG 배치
  간 동일 hard 선택(배치 평균 score). CPU mock 4 config 완주. eval:
  mask-LPIPS→dense_ref + known PSNR + wall (스크립트 내장).

### edit_16 리뷰 반영 (필수 5 + 권장 3)
1) Stage 18/17 LPIPS spatial=True + 1x1 map 거부 (핵심 — 이전 코드는
   full-image LPIPS를 mask-local로 오표기할 뻔함; 수정 전 결과 사용 금지).
2) Stage 18 bucket = manifest bucket 그대로 (자체 재분류 삭제 — 기존 논문
   mask-condition breakdown과 정합).
3) Stage 18 crop side를 int(min(max(side,MIN),H,W))로 clamp + 좌표 assert
   (전체-이미지 mask 케이스 검증).
4) Stage 18 DINO fallback 삭제 — masked pooling은 spatial token 필수,
   timm 실패 시 명확히 죽음. real feature는 sid당 1회 캐시 (5 arm 중복
   forward 제거: 50k -> 30k). 스크립트에 LIMIT 노출 (smoke: LIMIT=20).
5) Stage 17 dense_matched: 프로세스 분리로 walls가 비는 문제 — 기존
   run.json에서 load_wall로 복원, 없으면 FATAL(+--matched-steps 안내);
   실행 후 실측 wall이 target 대비 10%+ 어긋나면 dense-curve 재매칭 경고.
6) Stage 16 DREF 기본 안내 n500 + 사전검증: png 개수>=N2, manifest 앞
   N2개 stem 커버리지 assert.
7) Stage 16 명칭 정직화: "BlockCache-style/FORA-style mechanism-matched
   adaptation" — 공식 재현 주장 금지 주석.
서버 확인 요청: pytest 종료 여부 (tests는 plain python 실행도 지원).
