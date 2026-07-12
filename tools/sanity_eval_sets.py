"""tools.sanity_eval_sets — Stage 9 시작 전 필수 sanity (#8).
1) dense50 vs dense50 FID≈0  2) stem set 동일성  3) composited 재구성 일치.

    python -m tools.sanity_eval_sets --runs dense_s50 mbd_... reuse_... \
        --manifest MAN --resolution 1024
"""
import argparse, json, tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _raw_stems(run: Path):
    return {p.stem for p in run.glob("*.png") if not p.stem.endswith("_pasted")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--dense-ref", required=True)
    a = ap.parse_args()

    # (2) stem set 동일성
    sets = {Path(r).name: _raw_stems(Path(r)) for r in a.runs}
    ref = next(iter(sets.values()))
    all_same = all(v == ref for v in sets.values())
    print(f"[2] stem-set identity: {'OK' if all_same else 'FAIL'} "
          f"({len(ref)} images)")
    if not all_same:
        for n, v in sets.items():
            print(f"    {n}: {len(v)} (diff {len(v ^ ref)})")

    # (1) dense50 vs 자기자신 FID≈0
    from cleanfid import fid
    dref = Path(a.dense_ref)
    with tempfile.TemporaryDirectory() as td:
        import shutil
        d1, d2 = Path(td) / "a", Path(td) / "b"
        d1.mkdir(); d2.mkdir()
        for p in dref.glob("*.png"):
            if p.stem.endswith("_pasted"):
                continue
            shutil.copy(p, d1 / p.name); shutil.copy(p, d2 / p.name)
        f = fid.compute_fid(str(d1), str(d2))
        print(f"[1] dense50 self-FID = {f:.4f} {'OK' if f < 1.0 else 'SUSPICIOUS'}")

    # (3) composited 재구성 일치 (첫 3장)
    items = {Path(it["sample_id"]).stem: it for it in
             json.load(open(a.manifest))["items"]}
    R = a.resolution
    checked = 0
    for stem in list(ref)[:3]:
        run = Path(a.runs[0])
        raw_p = run / f"{stem}.png"
        pasted_p = run / f"{stem}_pasted.png"
        mask_p = run / f"{stem}_mask.pt"
        if not (raw_p.exists() and pasted_p.exists() and mask_p.exists()):
            continue
        it = items.get(stem)
        if it is None:
            continue
        # 반드시 생성 파이프라인과 동일 전처리(load_image_rgb=LANCZOS) 사용 —
        # default-filter resize(BICUBIC)를 쓰면 known region 전체에 10~30 레벨의
        # 가짜 오차가 생긴다 (smoke에서 실측된 16~31 오차의 원인).
        from data.dataset import load_image_rgb
        raw = np.asarray(Image.open(raw_p).convert("RGB")).astype(np.float32)
        inp = np.asarray(load_image_rgb(it["image"], R)).astype(np.float32)
        m = torch.load(mask_p).squeeze().numpy()
        if m.shape != (R, R):
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                           .resize((R, R), Image.NEAREST)) / 255.0
        m = m[..., None]
        recon = m * raw + (1 - m) * inp
        pasted = np.asarray(Image.open(pasted_p).convert("RGB").resize((R, R))).astype(np.float32)
        diff = np.abs(recon - pasted)
        # 진단 분포 + 경계 집중도 (경계 집중=soft mask/resize, 전역=전처리 불일치)
        try:
            from scipy import ndimage
            hard = m[..., 0] > 0.5
            edge = ndimage.binary_dilation(hard, iterations=3) ^ \
                   ndimage.binary_erosion(hard, iterations=3)
        except ImportError:
            edge = np.zeros(m.shape[:2], dtype=bool)
        frac_gt2 = float((diff > 2).mean())
        on_edge = float((diff[edge] > 2).mean()) if edge.any() else 0.0
        print(f"[3] {stem}: max={diff.max():.0f} mean={diff.mean():.2f} "
              f"p99={np.percentile(diff, 99):.0f} >2={frac_gt2:.4f} "
              f"(경계부 >2 비율={on_edge:.3f}) "
              f"{'OK' if diff.max() < 3 else 'CHECK'}")
        checked += 1
    if not checked:
        print("[3] composited 재구성: 파일 없음(skip)")


if __name__ == "__main__":
    main()
