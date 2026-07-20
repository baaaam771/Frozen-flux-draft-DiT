"""tools/sd3_masked_transfer.py — 패키지 A (3순위): SD3.5-Large에서
controlled masked-generation quality-latency transfer (종합평가2 최종
추천 1안).

native inpainting checkpoint가 아니므로 매 스텝 known-region reinjection
으로 masked generation을 구성한다 (논문 표기: "controlled
masked-generation transfer" — "native inpainting"이라 부르지 않는다):

  z_t = M ⊙ z_t^{gen} + (1-M) ⊙ z_t^{known},
  z_t^{known} = (1-σ_t)·z_0 + σ_t·ε_fixed        (SD3 rectified flow;
  σ_t는 스케줄러의 sigmas에서 읽음 — FLUX 공식을 복사하지 않는다)

mask가 존재하므로 FLUX와 동일한 mask+boundary+delta selector를 재튜닝
없이 사용한다 (c=2, r∈{0.15, 0.3}, tail은 스텝 수 비례).

Arms (--arms): dense_ref / dense_matched / reuse / refresh_r015 /
refresh_r03 / refresh_kv_r03
SD3.5는 전 블록이 joint(dual)라 FLUX의 K/V-only tier는 존재하지 않음 —
명칭은 joint-token refresh (+K/V cache).

  python -m tools.sd3_masked_transfer --manifest data/coco_manifest_1024.json \
      --out $OUT --limit 100 --arms dense_ref refresh_r03 ...
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from data.dataset import load_image_rgb
from data.masks import MaskSpec, make_mask
from models.sd3_sparse_runner import SD3SparseRunner, SD3AnchorCache
from token_selectors.combo import ComboWeights, combo_score
from tools.sd3_quality import _encode, _load, _unpatchify


def _mask_tokens(mask_px, latH):
    """[H, W] 픽셀 mask -> patch-token [1, N] (N=(latH/2)²) + boundary."""
    m = torch.from_numpy(mask_px)[None, None]
    m_tok = torch.nn.functional.interpolate(m, (latH // 2, latH // 2),
                                            mode="area").reshape(1, -1)
    k = torch.ones(1, 1, 3, 3)
    md = (torch.nn.functional.conv2d(m, k, padding=1) > 0).float()
    me = (torch.nn.functional.conv2d(m, k, padding=1) >= 9).float()
    b_tok = torch.nn.functional.interpolate(md - me, (latH // 2, latH // 2),
                                            mode="area").reshape(1, -1)
    return m_tok, b_tok


def _topk_sorted(score, r):
    k = max(1, int(r * score.shape[1]))
    return torch.sort(score.topk(k, dim=1).indices, dim=1).values


@torch.inference_mode()
def sample(pipe, runner, prompt, z0, mask_px, seed, steps, dev, dtype,
           mode, ratio=0.3, kv=True, anchor_c=2, tail=None, guidance=4.5,
           height=1024):
    latH = height // 8
    tail = tail if tail is not None else max(2, round(steps * 4 / 50))
    sched = pipe.scheduler
    sched.set_timesteps(steps, device=dev)
    timesteps = sched.timesteps
    sigmas = sched.sigmas                       # len == steps+1

    pe, po = _encode(pipe, prompt, dev, guidance)
    B = pe.shape[0]
    g = torch.Generator(dev).manual_seed(seed)
    eps = torch.randn(z0.shape, generator=g, device=dev, dtype=torch.float32)
    m_lat = torch.nn.functional.interpolate(
        torch.from_numpy(mask_px)[None, None].to(dev), (latH, latH),
        mode="area")                             # [1,1,latH,latH] soft
    m_tok, b_tok = _mask_tokens(mask_px, latH)
    m_tok, b_tok = m_tok.to(dev), b_tok.to(dev)
    w = ComboWeights(1.0, 0.5, 0.0, 1.0, 0.0)   # mbd — FLUX와 동일, 재튜닝 X

    def known(sig):
        return ((1.0 - sig) * z0.float() + sig * eps).to(dtype)

    lat = known(float(sigmas[0]))                # 시작: fully-noised known
    cache = SD3AnchorCache()
    v_anchor = prev_v = None
    n_anchor = n_sparse = 0
    ch = pipe.transformer.config.out_channels
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, t in enumerate(timesteps):
        lm = torch.cat([lat] * B) if B > 1 else lat
        tt = t.expand(lm.shape[0])
        is_anchor = (mode == "dense" or i % anchor_c == 0
                     or i >= steps - tail)
        if is_anchor:
            v = runner.dense_forward(lm, pe, tt, po, record=cache)
            prev_v, v_anchor = v_anchor, v
            n_anchor += 1
        elif mode == "reuse":
            v = v_anchor
            n_sparse += 1
        else:                                    # refresh (joint-token)
            delta = ((v_anchor - prev_v) if prev_v is not None
                     else v_anchor).abs().mean(-1).float()
            # CFG 배치 간 동일 선택 (uncond/cond 일관성): 배치 평균 score
            score = combo_score(w, mask=m_tok, boundary=b_tok,
                                frequency=None,
                                delta=delta.mean(0, keepdim=True))
            hard = _topk_sorted(score, ratio).expand(v_anchor.shape[0], -1)
            v = runner.sparse_forward(lm, pe, tt, po, cache, hard,
                                      "dualkv" if kv else "dual")
            n_sparse += 1
        v_img = _unpatchify(v, latH, latH, ch)
        if B > 1:
            vu, vc = v_img.chunk(2)
            v_img = vu + guidance * (vc - vu)
        lat = sched.step(v_img, t, lat).prev_sample
        # known-region reinjection at σ_{t+1} (rectified-flow forward)
        sig_next = float(sigmas[i + 1])
        lat = (m_lat * lat.float()
               + (1 - m_lat) * known(sig_next).float()).to(dtype)
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
    ap.add_argument("--matched-steps", type=int, default=0)
    ap.add_argument("--arms", nargs="+",
                    default=["dense_ref", "refresh_r03", "refresh_kv_r03",
                             "reuse", "dense_matched"])
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    pipe = _load(a.model_id, dev, dtype)
    runner = SD3SparseRunner(pipe.transformer)
    items = json.load(open(a.manifest))["items"][: a.limit]
    R = a.resolution
    os.makedirs(a.out, exist_ok=True)

    arm_cfg = {
        "dense_ref": dict(mode="dense"),
        "dense_matched": dict(mode="dense"),
        "reuse": dict(mode="reuse"),
        "refresh_r015": dict(mode="refresh", ratio=0.15, kv=False),
        "refresh_r03": dict(mode="refresh", ratio=0.3, kv=False),
        "refresh_kv_r03": dict(mode="refresh", ratio=0.3, kv=True),
    }
    def load_wall(arm):
        p = os.path.join(a.out, arm, "run.json")
        if not os.path.exists(p):
            return None
        r = json.load(open(p)).get("rows", [])
        if not r:
            return None
        import statistics
        return statistics.median(x["wall_s"] for x in r)

    # 별도 프로세스로 arm을 나눠 실행해도 wall-matching이 유지되도록
    # 기존 run.json에서 measured wall을 복원 (edit_16 문제 5)
    walls = {}
    for existing in arm_cfg:
        wl = load_wall(existing)
        if wl is not None:
            walls[existing] = wl
    for arm in a.arms:
        cfg = dict(arm_cfg[arm])
        steps = a.steps
        if arm == "dense_matched":
            ref_arm = next((x for x in ("refresh_kv_r03", "refresh_r03")
                            if x in walls), None)
            if not a.matched_steps and (ref_arm is None
                                        or "dense_ref" not in walls):
                raise SystemExit(
                    "FATAL: dense_matched는 dense_ref와 refresh arm의 "
                    "run.json wall이 필요합니다 — 해당 arm을 먼저 "
                    "실행하거나 --matched-steps를 지정하세요")
            steps = a.matched_steps or max(4, round(
                a.steps * walls[ref_arm] / walls["dense_ref"]))
            print(f"[dense_matched] steps={steps} "
                  f"({ref_arm} {walls[ref_arm]:.2f}s / "
                  f"dense_ref {walls['dense_ref']:.2f}s)")
        d = os.path.join(a.out, arm)
        os.makedirs(d, exist_ok=True)
        rows = []
        rj = os.path.join(d, "run.json")
        if os.path.exists(rj):
            try:
                rows = json.load(open(rj)).get("rows", [])
            except Exception:
                rows = []
        done = {r["sample_id"] for r in rows}
        for it in items:
            sid = os.path.splitext(os.path.basename(str(it["sample_id"])))[0]
            fp = os.path.join(d, f"{sid}.png")
            if os.path.exists(fp) and sid in done:
                continue
            if os.path.exists(fp):
                continue
            img_pil = load_image_rgb(it["image"], R)
            x = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)[None] \
                .float().to(dev) / 127.5 - 1.0
            z0 = (pipe.vae.encode(x.to(dtype)).latent_dist.mode()
                  - getattr(pipe.vae.config, "shift_factor", 0.0)) \
                * pipe.vae.config.scaling_factor
            spec = MaskSpec(it["sample_id"], it["mask_type"], it["bucket"],
                            it["mask_seed"])
            mask_px = make_mask(R, R, spec)[0].numpy().astype(np.float32)
            img, wall, ev = sample(
                pipe, runner, it["prompt"], z0.float(), mask_px,
                20000 + int(re.sub(r"\D", "", sid) or 0) % 10 ** 6
                + a.seed_offset, steps, dev, dtype, cfg["mode"],
                ratio=cfg.get("ratio", 0.3), kv=cfg.get("kv", True),
                guidance=a.guidance, height=R)
            img.save(fp)
            rows.append(dict(sample_id=sid, wall_s=wall, **ev))
        json.dump(dict(config=dict(arm=arm, steps=steps, **{
            k: v for k, v in cfg.items()}, guidance=a.guidance,
            model=a.model_id, resolution=R, protocol="masked_reinjection"),
            rows=rows), open(rj, "w"), indent=1)
        if rows:
            import statistics
            walls[arm] = statistics.median(r["wall_s"] for r in rows)
            print(f"[{arm}] n={len(rows)} wall={walls[arm]:.2f}s steps={steps}")


if __name__ == "__main__":
    main()