"""tools/sd3_quality.py — Stage 14: 두 번째 backbone(SD3.5-Large)의 실제
품질-속도 4-arm 실험 (총평 1순위).

목적 (총평 문구 그대로): "cost-floor 예측 → naive refresh 실패(all-dual에서
naive는 dense와 동일) → 구조에 맞는 lever 적용 → 실제 frontier 개선"의
인과관계를 재현한다. selector transfer가 아니라 cost-removal transfer 검증.

주의: SD3.5-Large는 inpainting checkpoint가 아니므로 text-to-image로
실험한다 (mask 없음 → mask/boundary prior 없이 delta selector만 사용;
이는 논문 주장 범위와 일치 — "cost-removal principle이 아키텍처가 달라도
작동하는가").

Arms:
  dense_ref      : 공식 28-step dense (reference; LPIPS의 기준)
  dense_matched  : dualkv wall에 맞춘 감축 step dense (가장 강한 baseline)
  reuse_c2_t4    : anchored reuse (sparse step은 v_anchor 재사용)
  dualkv_c2_r03_t4 : dual+KV selective refresh, delta selector r=0.3

동일 프롬프트(COCO 캡션)·latent seed·guidance를 모든 arm이 공유.
Metric: LPIPS(전체 이미지) -> 각자 자기 backbone의 dense_ref, CLIPScore,
end-to-end wall (동기화 측정).

  python -m tools.sd3_quality --manifest data/coco_manifest_1024.json \
      --out $OUT --limit 100 --arms dense_ref dualkv reuse dense_matched
"""
import argparse
import json
import os
import time

import torch

from models.sd3_sparse_runner import SD3SparseRunner, SD3AnchorCache


def _load(model_id, dev, dtype):
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id,
                                                    torch_dtype=dtype)
    pipe.to(dev)
    return pipe


@torch.inference_mode()
def _encode(pipe, prompt, dev, guidance):
    pe, npe, po, npo = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None, device=dev,
        do_classifier_free_guidance=guidance > 1.0)
    if guidance > 1.0:
        pe = torch.cat([npe, pe])          # [2, T, D] (uncond, cond)
        po = torch.cat([npo, po])
    return pe, po


def _delta_topk(v, prev, r):
    """anchor 간 velocity 변화 크기 top-r (delta selector — mask 없는 t2i에서
    사용 가능한 유일한 학습-불필요 신호). 첫 anchor 직후(prev 없음)는 현재
    v 크기 기준."""
    score = ((v - prev) if prev is not None else v).abs().mean(-1)  # [B, N]
    k = max(1, int(r * score.shape[1]))
    idx = score.topk(k, dim=1).indices
    return torch.sort(idx, dim=1).values


def _unpatchify(v, latH, latW, ch):
    """[B, N, 4*ch] 패치 시퀀스 -> [B, ch, latH, latW] (exactness [1]과 동일)."""
    B = v.shape[0]
    pH, pW = latH // 2, latW // 2
    return v.reshape(B, pH, pW, 2, 2, ch).permute(0, 5, 1, 3, 2, 4).reshape(
        B, ch, latH, latW)


@torch.inference_mode()
def sample(pipe, runner, prompt, seed, steps, dev, dtype, mode,
           anchor_c=2, ratio=0.3, tail=4, guidance=4.5, height=1024,
           width=1024):
    """mode: dense | reuse | dualkv. 반환 (PIL image, wall_s, evals dict)."""
    sched = pipe.scheduler
    sched.set_timesteps(steps, device=dev)
    timesteps = sched.timesteps

    pe, po = _encode(pipe, prompt, dev, guidance)
    B = pe.shape[0]                                # 2 with CFG

    g = torch.Generator(dev).manual_seed(seed)
    ch = pipe.transformer.config.in_channels
    lat = torch.randn(1, ch, height // 8, width // 8, generator=g,
                      device=dev, dtype=dtype)
    lat = lat * sched.init_noise_sigma if hasattr(sched, "init_noise_sigma") \
        else lat

    cache = SD3AnchorCache()
    v_anchor = prev_v_anchor = None    # 패치-시퀀스 공간의 anchor velocity
    n_anchor = n_sparse = 0
    latH, latW = height // 8, width // 8
    ch = pipe.transformer.config.out_channels
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, t in enumerate(timesteps):
        lm = torch.cat([lat] * B) if B > 1 else lat
        tt = t.expand(lm.shape[0])         # SD3: 1000-scale timestep 그대로
        is_anchor = (mode == "dense" or i % anchor_c == 0
                     or i >= steps - tail)
        if is_anchor:
            v = runner.dense_forward(lm, pe, tt, po, record=cache)
            prev_v_anchor, v_anchor = v_anchor, v
            n_anchor += 1
        else:
            if mode == "reuse":
                v = v_anchor
            else:                                   # dualkv
                hard = _delta_topk(v_anchor, prev_v_anchor, ratio)
                # sparse_forward는 FULL [B, N, out]을 반환한다 (easy 행은
                # anchor depth-state 기반으로 이미 채워짐 — exactness [2]의
                # full-output 검사가 보증). scatter 금지: PyTorch scatter는
                # index 크기만큼 src 앞부분을 쓰므로 토큰 0..k-1 값이 hard
                # 위치에 흩뿌려져 출력이 붕괴한다 (Stage 14 1차 실행의 원인).
                v = runner.sparse_forward(lm, pe, tt, po, cache, hard,
                                          "dualkv")
            n_sparse += 1
        v_img = _unpatchify(v, latH, latW, ch)
        if B > 1:
            vu, vc = v_img.chunk(2)
            v_img = vu + guidance * (vc - vu)
        lat = sched.step(v_img, t, lat).prev_sample
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    dec = (lat / pipe.vae.config.scaling_factor) \
        + getattr(pipe.vae.config, "shift_factor", 0.0)
    img = pipe.vae.decode(dec).sample
    img = pipe.image_processor.postprocess(img, output_type="pil")[0]
    return img, wall, dict(anchor=n_anchor, sparse=n_sparse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--model-id",
                    default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--matched-steps", type=int, default=0,
                    help="dense_matched의 step 수 (0이면 dualkv wall 실측으로"
                         " 자동 산출)")
    ap.add_argument("--arms", nargs="+",
                    default=["dense_ref", "dualkv", "reuse", "dense_matched"])
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    pipe = _load(a.model_id, dev, dtype)
    runner = SD3SparseRunner(pipe.transformer)

    items = json.load(open(a.manifest))["items"][: a.limit]
    os.makedirs(a.out, exist_ok=True)

    arm_cfg = {
        "dense_ref": dict(mode="dense", steps=a.steps),
        "dense_matched": dict(mode="dense", steps=None),   # 뒤에서 결정
        "reuse": dict(mode="reuse", steps=a.steps),
        "dualkv": dict(mode="dualkv", steps=a.steps),
    }

    walls = {}
    for arm in a.arms:
        cfg = arm_cfg[arm]
        steps = cfg["steps"]
        if arm == "dense_matched":
            steps = a.matched_steps or max(
                4, round(a.steps * walls.get("dualkv", 0.5)
                         / max(walls.get("dense_ref", 1.0), 1e-6)))
            print(f"[dense_matched] steps={steps} "
                  f"(dualkv wall {walls.get('dualkv', 0):.2f}s / "
                  f"dense {walls.get('dense_ref', 0):.2f}s)")
        d = os.path.join(a.out, arm)
        os.makedirs(d, exist_ok=True)
        # resume: 기존 run.json의 rows를 보존 (skip된 이미지의 wall 기록 유지
        # — 이걸 빈 리스트로 덮어쓰면 매칭/집계가 죽는다: Stage 14 2차의 원인)
        rows = []
        rj = os.path.join(d, "run.json")
        if os.path.exists(rj):
            try:
                rows = json.load(open(rj)).get("rows", [])
            except Exception:
                rows = []
        done_ids = {r["sample_id"] for r in rows}
        for it in items:
            sid = str(it["sample_id"])
            sid_stem = os.path.splitext(os.path.basename(sid))[0]
            fp = os.path.join(d, f"{sid_stem}.png")
            if os.path.exists(fp) and sid_stem in done_ids:
                continue
            if os.path.exists(fp):
                continue          # 이미지만 있고 기록 없음 — wall 재측정 불가
            img, wall, ev = sample(pipe, runner, it["prompt"],
                                   10000 + int(sid_stem) % 10 ** 6
                                   + a.seed_offset,
                                   steps, dev, dtype, cfg["mode"],
                                   guidance=a.guidance,
                                   height=a.resolution, width=a.resolution)
            img.save(fp)
            rows.append(dict(sample_id=sid_stem, wall_s=wall, **ev))
        json.dump(dict(config=dict(arm=arm, steps=steps, mode=cfg["mode"],
                                   guidance=a.guidance, model=a.model_id,
                                   resolution=a.resolution,
                                   anchor_c=2, ratio=0.3, tail=4),
                       rows=rows),
                  open(os.path.join(d, "run.json"), "w"), indent=1)
        if rows:
            import statistics
            walls[arm] = statistics.median(r["wall_s"] for r in rows)
            print(f"[{arm}] n={len(rows)} median wall={walls[arm]:.2f}s "
                  f"steps={steps}")


if __name__ == "__main__":
    main()