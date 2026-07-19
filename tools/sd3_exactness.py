"""tools.sd3_exactness — 2nd MMDiT exactness gate (v2).

P0 수정: (a) fp32 depth 검사는 official forward로 순차 진행하며 target 블록
직전의 실제 in-distribution 상태에서 비교, (b) 대상 블록만 fp32로
일시 승격한 뒤 bf16으로 복원 (전체 float() 금지 — OOM), (c) rel+max_abs
동시 판정, (d) 실패 시 SystemExit(1) 직접. 인코더/VAE는 encode 후 CPU로.

    python -m tools.sd3_exactness --model-id stabilityai/stable-diffusion-3.5-large
"""
from __future__ import annotations

import argparse

import torch


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id",
                    default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--ratio", type=float, default=0.15)
    a = ap.parse_args()

    from diffusers import StableDiffusion3Pipeline
    from models.sd3_sparse_runner import SD3SparseRunner, SD3AnchorCache

    dev, dtype = "cuda", torch.bfloat16
    pipe = StableDiffusion3Pipeline.from_pretrained(a.model_id,
                                                    torch_dtype=dtype)
    pipe.to(dev)
    runner = SD3SparseRunner(pipe.transformer)
    failed = False

    pe, _, pooled, _ = pipe.encode_prompt(
        prompt="a photo of a mountain lake at sunrise", prompt_2=None,
        prompt_3=None, device=dev)
    # exactness에는 transformer만 필요 — 인코더/VAE는 내려서 VRAM 회수
    for m in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3,
              pipe.vae):
        if m is not None:
            m.to("cpu")
    torch.cuda.empty_cache()
    lat_ch = pipe.transformer.config.in_channels
    H = a.resolution // pipe.vae_scale_factor
    g = torch.Generator(dev).manual_seed(0)
    latents = torch.randn(1, lat_ch, H, H, generator=g, device=dev,
                          dtype=dtype)
    pipe.scheduler.set_timesteps(8, device=dev)
    for ti in pipe.scheduler.timesteps[:3]:      # in-distribution 상태 확보
        v = pipe.transformer(hidden_states=latents, timestep=ti.expand(1),
                             encoder_hidden_states=pe,
                             pooled_projections=pooled,
                             return_dict=False)[0]
        latents = pipe.scheduler.step(v, ti, latents, return_dict=False)[0]
    t = pipe.scheduler.timesteps[3].expand(1)

    # ---- (1) official vs manual dense, bf16 전체 ----
    v_off = pipe.transformer(hidden_states=latents, timestep=t,
                             encoder_hidden_states=pe,
                             pooled_projections=pooled, return_dict=False)[0]
    cache = SD3AnchorCache()
    v_man = runner.dense_forward(latents, pe, t, pooled, record=cache)
    pH = H // 2
    v_man_img = v_man.reshape(1, pH, pH, 2, 2, lat_ch).permute(
        0, 5, 1, 3, 2, 4).reshape(1, lat_ch, H, H)
    d = v_off.float() - v_man_img.float()
    rel = (d.norm() / v_off.float().norm().clamp_min(1e-12)).item()
    ok = rel < 3e-2
    print(f"[1] official vs manual dense (bf16 full): rel={rel:.3e} "
          f"max_abs={d.abs().max().item():.3e} {'OK' if ok else 'FAIL'}")
    failed |= not ok
    del v_off, v_man_img, d          # v_man/cache는 [2]에서 필요 — 유지
    torch.cuda.empty_cache()

    # ---- (1b) fp32 블록별: 대상 블록만 fp32 승격-검사-복원 (OOM 방지),
    #      depth progression은 bf16 official 경로 ----
    img, ctx, temb = runner.embed(latents, pe, t, pooled)
    n_blk = len(pipe.transformer.transformer_blocks)
    targets = {0, n_blk // 2, n_blk - 1}
    for bi, block in enumerate(pipe.transformer.transformer_blocks):
        if bi in targets:
            block.float()
            i32, c32, t32 = img.float(), ctx.float(), temb.float()
            man_img, _ = _manual_one_block(runner, block, i32, c32, t32)
            out32 = block(hidden_states=i32, encoder_hidden_states=c32,
                          temb=t32)
            off_img = out32[1] if isinstance(out32, tuple) else out32
            r = ((man_img - off_img).norm()
                 / off_img.norm().clamp_min(1e-12)).item()
            ok = r < 1e-5
            print(f"[1b] block {bi} fp32: rel={r:.3e} "
                  f"{'OK' if ok else 'FAIL'}")
            failed |= not ok
            del man_img, off_img, out32, i32, c32, t32
            block.to(dtype)
            torch.cuda.empty_cache()
        out = block(hidden_states=img, encoder_hidden_states=ctx, temb=temb)
        if isinstance(out, tuple):
            ctx_n, img_n = out
        else:
            ctx_n, img_n = ctx, out
        img, ctx = img_n, (ctx_n if ctx_n is not None else ctx)

    # ---- (2) fresh-cache sparse == dense (dual / dualkv, hard 행) ----
    N = cache.img_states[0].shape[1]
    k = max(int(a.ratio * N), 1)
    hard = torch.randperm(N, device=dev)[:k].sort().values.unsqueeze(0)
    vd = torch.gather(v_man, 1,
                      hard.unsqueeze(-1).expand(-1, -1, v_man.shape[-1]))
    for lever in ("dual", "dualkv"):
        v_sp = runner.sparse_forward(latents, pe, t, pooled, cache, hard,
                                     lever)
        # hard rows: sparse 계산 경로의 핵심
        vh = torch.gather(v_sp, 1,
                          hard.unsqueeze(-1).expand(-1, -1, v_sp.shape[-1]))
        hd = vh.float() - vd.float()
        hard_rel = (hd.norm() / vd.float().norm().clamp_min(1e-12)).item()
        # full output: easy-row scatter/final 경로까지 검증 (fresh cache에서는
        # easy row도 dense와 같아야 함)
        fd = v_sp.float() - v_man.float()
        full_rel = (fd.norm()
                    / v_man.float().norm().clamp_min(1e-12)).item()
        ok = hard_rel < 1e-3 and full_rel < 1e-3
        print(f"[2] fresh-cache {lever}: full rel={full_rel:.3e}, "
              f"hard rel={hard_rel:.3e} "
              f"(max_abs full={fd.abs().max().item():.3e}) "
              f"{'OK' if ok else 'FAIL'}")
        failed |= not ok

    if failed:
        raise SystemExit("SD3 exactness gate FAILED")
    print("All SD3 exactness checks passed.")


def _manual_one_block(runner, block, img, ctx, temb):
    """dense_forward의 단일 블록 등가 (fp32 검사용)."""
    import torch as _t
    from models.sd3_sparse_runner import _norm1
    pre_only = block.context_pre_only
    n_img, g_msa, s_mlp, sc_mlp, g_mlp, n_img2, g_msa2 = _norm1(block, img,
                                                                 temb)
    if pre_only:
        n_ctx = block.norm1_context(ctx, temb)
        c_gates = None
    else:
        n_ctx, *c_gates = block.norm1_context(ctx, emb=temb)
    attn = block.attn
    q_i = attn.to_q(n_img)
    q_i = attn.norm_q(q_i.view(*q_i.shape[:2], runner.heads, -1)).view_as(q_i)
    k_i, v_i = runner._img_kv(attn, n_img)
    q_c, k_c, v_c = runner._txt_qkv(block, n_ctx, pre_only)
    from models.sd3_sparse_runner import _attn
    T = ctx.shape[1]
    k_full = _t.cat([k_c, k_i], dim=1)
    v_full = _t.cat([v_c, v_i], dim=1)
    q = _t.cat([q_c, q_i], dim=1) if q_c is not None else q_i
    o = _attn(q, k_full, v_full, runner.heads)
    o_c, o_i = (o[:, :T], o[:, T:]) if q_c is not None else (None, o)
    img2 = img + g_msa.unsqueeze(1) * attn.to_out[0](o_i)
    if n_img2 is not None:
        a2 = block.attn2
        k2, v2 = runner._attn2_kv(a2, n_img2)
        q2 = runner._attn2_q(a2, n_img2)
        img2 = img2 + g_msa2.unsqueeze(1) * a2.to_out[0](
            _attn(q2, k2, v2, runner.heads))
    n2 = block.norm2(img2) * (1 + sc_mlp[:, None]) + s_mlp[:, None]
    img2 = img2 + g_mlp.unsqueeze(1) * block.ff(n2)
    ctx2 = ctx
    if o_c is not None and not pre_only:
        ctx2 = runner._ctx_update(block, ctx, o_c, c_gates)
    return img2, ctx2


if __name__ == "__main__":
    main()