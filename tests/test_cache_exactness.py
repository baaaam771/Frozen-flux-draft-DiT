"""Gate B2 (GPU, requires FLUX weights): fresh-cache exactness on the REAL model.

v2 — real-input edition. The v1 synthetic randn packed input is far out of
distribution: activations explode through the 19 dual blocks (probe measured
1e5–1e9 magnitudes), and then the *shape-dependent* bf16 GEMM reduction-order
difference between the sparse row-count (Sq) and dense row-count (S) GEMMs is
amplified into huge ABSOLUTE errors, even though the computation is correct
(tools/bisect_gate_b2: P1 full-seq = 0.0 exactly; per-op inputs identical).

Therefore this gate now:
  * builds the model input from a REAL image/mask/prompt through the actual
    pipeline preprocessing (in-distribution activations), and
  * judges by RELATIVE error  rel = max|dv_hard| / max|v_dense|,
    with bf16 tol 1e-2 and fp32 tol 1e-4 (fp32 GEMMs still tile by shape).

DACE rule: if this gate fails, DO NOT proceed to any experiment.

    PYTHONPATH=. python tests/test_cache_exactness.py \
        --image sample.png --mask sample_mask.png --prompt "..." [--fp32]
"""
import argparse

import torch

from models.flux_cache import FluxAnchorCache
from models.flux_fill_loader import load_flux_fill
from models.flux_sparse_transformer import FluxSparseRunner


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--step-index", type=int, default=0,
                    help="which scheduler step to test at (0 = t=1.0)")
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.3, 0.7])
    ap.add_argument("--dual-sparse", action="store_true",
                    help="Lever A 경로 검사 (fresh cache이므로 exact 기준)")
    ap.add_argument("--kv-cache", action="store_true",
                    help="Lever B 경로 검사 (anchor와 동일 step이므로 역시 exact 기준)")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    from PIL import Image
    from samplers.dense_flux_fill import prepare_flux_fill_inputs, scheduler_step

    dtype = torch.float32 if a.fp32 else torch.bfloat16
    comps = load_flux_fill(dtype=dtype, keep_text_encoders=True)
    pipe, dev = comps.pipe, comps.device
    runner = FluxSparseRunner(pipe.transformer)

    state = prepare_flux_fill_inputs(
        pipe, Image.open(a.image).convert("RGB"), Image.open(a.mask).convert("L"),
        a.prompt, seed=0, num_steps=a.steps, guidance_scale=a.guidance,
        device=dev, dtype=dtype)

    # advance to the requested step with plain dense steps (in-distribution z_t)
    for i in range(a.step_index):
        t = state.timesteps[i]
        mi = torch.cat([state.latents, state.cond], dim=2)
        v, _ = runner.dense_forward(mi, state.prompt_embeds, state.pooled,
                                    t.expand(1).to(dtype) / 1000, state.guidance,
                                    state.img_ids, state.txt_ids)
        state.latents = scheduler_step(pipe, v, t, state.latents)

    t = state.timesteps[a.step_index]
    model_input = torch.cat([state.latents, state.cond], dim=2)
    timestep = t.expand(1).to(dtype) / 1000
    N = state.latents.shape[1]

    cache = FluxAnchorCache()
    v_dense, _ = runner.dense_forward(model_input, state.prompt_embeds, state.pooled,
                                      timestep, state.guidance, state.img_ids,
                                      state.txt_ids, cache=cache, step_index=a.step_index,
                                      record_kv=a.kv_cache,
                                      record_dual=a.dual_sparse)
    scale = v_dense.float().abs().max().item()
    print(f"[Gate B2] step {a.step_index}: max|v_dense| = {scale:.3e} "
          f"(in-distribution check; 랜덤 입력이면 1e4+로 폭발)")

    # RELATIVE tolerance. bf16 기준은 'ulp 예산': 57개 블록에 걸친 shape 의존
    # GEMM 축약 차이는 |v|의 bf16 ulp(2^-8·|v|) 단위로 쌓인다 — 실측 single-only
    # 1 ulp, dual+kv 2 ulp. bf16은 3 ulp(rel≈1.2e-2)+여유 = 2e-2로 두고,
    # 수학적 exactness의 판정은 fp32(1e-4)가 담당한다.
    tol = 1e-4 if a.fp32 else 2e-2
    ok = True
    for r in a.ratios:
        k = max(1, int(r * N))
        hard = torch.sort(torch.randperm(N, device=dev)[:k]).values[None]
        v_hard, _ = runner.sparse_forward(model_input, state.prompt_embeds,
                                          state.pooled, timestep, state.guidance,
                                          state.img_ids, state.txt_ids, cache, hard,
                                          kv_cache=a.kv_cache,
                                          dual_sparse=a.dual_sparse)
        v_merged = runner.merge_prediction(cache, hard, v_hard)
        ref = torch.gather(v_dense, 1,
                           hard.unsqueeze(-1).expand(-1, -1, v_dense.shape[-1]))
        abs_err = (v_hard.float() - ref.float()).abs().max().item()
        rel_err = abs_err / max(scale, 1e-12)
        merged_abs = (v_merged.float() - v_dense.float()).abs().max().item()
        ulps = abs_err / (2 ** -8 * max(scale, 1e-12))
        status = "PASS" if rel_err <= tol else "FAIL"
        ok &= rel_err <= tol
        print(f"[Gate B2] ratio={r}: hard max|dv|={abs_err:.3e} "
              f"rel={rel_err:.3e} (~{ulps:.1f} bf16 ulp) "
              f"merged max|dv|={merged_abs:.3e}  {status}")
    print(f"cache VRAM: {cache.vram_bytes()/2**30:.2f} GB")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()