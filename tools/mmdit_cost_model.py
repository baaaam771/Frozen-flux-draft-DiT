"""tools.mmdit_cost_model — blockwise MAC 모델의 아키텍처 간 전이 (P1-1 축소판).

같은 sparse-policy 정의(anchor states + hard-token 갱신)를 두 종류의 MMDiT에
적용했을 때의 analytic cost를 계산한다:

  * FLUX 계열: N_dual dual-stream + N_single single-stream (기존 모델과 동일 항)
  * SD3 계열: 전부 joint(dual)-stream 블록

레버 정의 (양쪽 공통 의미):
  dense      : 전 토큰 전 블록
  naive      : hard 이미지 토큰만 Q/attn-출력을 갱신하되, K/V를 매 블록
               재계산 -> K/V의 소스가 되는 모든 토큰의 hidden(FF 포함)을
               갱신해야 함. 절약은 Q/O/attn뿐.
  +KV        : anchor K/V 캐시 재사용 -> K/V 재계산 제거. 텍스트 스트림은
               여전히 fresh(Q/O/FF 계산).
  +dual+KV   : text/context는 fully fresh 유지; easy-이미지 K/V는 anchor
               캐시에서 오고 hard-이미지 K/V만 refresh -> r->0 floor는
               텍스트 스트림 비용 (0이 아님).

출력: r -> MAC ratio 표 + r->0 floor. 프리셋 flux-fill / sd35-large /
sd35-medium (config 하드코딩; --model-id 주면 HF config에서 자동 로드).

    python -m tools.mmdit_cost_model --arch flux-fill sd35-large \
        --resolution 1024 --ratios 0.0 0.15 0.3 --out cost_transfer.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

PRESETS = {
    # n_attn2 = dual_attention_layers 수 (SD3.5의 image-only self-attn 추가
    # 블록; config의 dual_attention_layers에서 옴. Large=0, Medium=13)
    "flux-fill":   dict(n_dual=19, n_single=38, D=3072, m=4, T=512, out=64,
                        n_attn2=0),
    "sd35-large":  dict(n_dual=38, n_single=0,  D=2432, m=4, T=333, out=64,
                        n_attn2=0),
    "sd35-medium": dict(n_dual=24, n_single=0,  D=1536, m=4, T=333, out=64,
                        n_attn2=13),
}


def mac(arch: dict, N: int, r: float, lever: str) -> float:
    """이미지 토큰 N, hard 비율 r에서 한 step의 transformer MAC (상대 단위).

    레버 정의 (논문과 1:1; 텍스트 스트림은 모든 sparse 변형에서 fully fresh):
      dense  : 전 토큰 전 블록.
      naive  : 논문의 보수적 naive 정책 — dual/joint 블록은 dense 유지,
               single-stream 블록만 sparse(K/V는 매 step 전체 재계산).
               all-dual 모델에는 sparse-eligible 블록이 없어 floor = 1.0.
      dual   : Lever A — dual 블록에서 hard 이미지 행만 Q/O/FF, 이미지 K/V는
               depth별 anchor 상태에서 전체 재계산(2N), 텍스트 fully fresh.
      dualkv : Lever A+B — 이미지 K/V를 anchor 캐시에서 재사용하고 hard 행만
               fresh(2k), 텍스트 fully fresh. r->0 floor = 텍스트 비용
               (0이 아님).
    """
    nd, ns, D, m, T = (arch["n_dual"], arch["n_single"], arch["D"],
                       arch["m"], arch["T"])
    n_attn2 = arch.get("n_attn2", 0)
    S = N + T
    k = max(int(round(r * N)), 0)
    Sq = k + T                     # sparse 시 attention 쿼리 행 (text fresh)
    D2 = D * D

    dual_dense = nd * ((4 + 2 * m) * S * D2 + 2 * S * S * D)
    single_dense = ns * ((4 + 2 * m) * S * D2 + 2 * S * S * D)
    # SD3.5 dual-attention 블록: image-only self-attn 추가 (QKVO 4N + attn 2N²)
    attn2_dense = n_attn2 * (4 * N * D2 + 2 * N * N * D)
    head_dense = N * D * arch["out"]
    dense_total = dual_dense + single_dense + attn2_dense + head_dense
    if lever == "dense":
        return 1.0

    head = k * D * arch["out"]

    if lever == "naive":
        # dual 블록 dense(+attn2 dense) + single 블록 sparse(K/V 전체 재계산)
        single = ns * (((2 + 2 * m) * Sq + 2 * S) * D2 + 2 * Sq * S * D)
        return (dual_dense + attn2_dense + single + head) / dense_total

    # Lever A / A+B: 텍스트는 Q/K/V/O/FF 전부 fresh = (4+2m)T
    kv_img = 2 * N if lever == "dual" else 2 * k
    dual = nd * (((2 + 2 * m) * k + kv_img + (4 + 2 * m) * T) * D2
                 + 2 * Sq * S * D)
    if ns:
        kv_single = 2 * S if lever == "dual" else 2 * Sq
        single = ns * (((2 + 2 * m) * Sq + kv_single) * D2 + 2 * Sq * S * D)
    else:
        single = 0.0
    # attn2 sparse: hard Q/O(2k) + K/V(dual: 2N 재계산 / dualkv: 2k fresh)
    #               + hard-query x full-image attention (2kND)
    kv2 = 2 * N if lever == "dual" else 2 * k
    attn2 = n_attn2 * ((2 * k + kv2) * D2 + 2 * k * N * D)
    return (dual + single + attn2 + head) / dense_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["flux-fill", "sd35-large"])
    ap.add_argument("--model-id", default="",
                    help="주어지면 HF config에서 depth/D를 로드해 프리셋 검증")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=[0.0, 0.15, 0.3])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.model_id:
        from diffusers import SD3Transformer2DModel
        cfg = SD3Transformer2DModel.load_config(a.model_id,
                                                subfolder="transformer")
        print(f"[config check] {a.model_id}: layers={cfg['num_layers']}, "
              f"D={cfg['num_attention_heads'] * cfg['attention_head_dim']}")

    N = (a.resolution // 16) ** 2      # patch 2 on 8x latent
    lines = [f"# Analytic MAC transfer (resolution {a.resolution}², "
             f"N={N} image tokens)\n",
             "| arch | lever | " + " | ".join(f"r={r}" for r in a.ratios)
             + " |",
             "|---|---|" + "---|" * len(a.ratios)]
    for name in a.arch:
        arch = PRESETS[name]
        for lever in ("dense", "naive", "dual", "dualkv"):
            vals = [mac(arch, N, r, lever) for r in a.ratios]
            lines.append(f"| {name} ({arch['n_dual']}d+{arch['n_single']}s, "
                         f"D={arch['D']}) | {lever} | "
                         + " | ".join(f"{v:.3f}" for v in vals) + " |")
    md = "\n".join(lines) + "\n\n" + (
        "해석: naive(논문의 보수 정책)의 floor는 dense로 남는 dual 블록에서 "
        "발생 — FLUX 0.49는 19 dual + full-seq K/V, all-dual SD3에서는 "
        "sparse-eligible 블록이 없어 1.0으로 상승 (dual sparsification이 "
        "선택이 아니라 필수). Lever A(dual)는 이미지 K/V 전체 재계산 비용을, "
        "Lever A+B(dualkv)는 텍스트-fresh 비용만 floor로 남긴다(r→0에서도 "
        "0이 아님). 블록 구성이 floor를 결정한다.\n")
    print(md)
    if a.out:
        Path(a.out).write_text(md)


if __name__ == "__main__":
    main()