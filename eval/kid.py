"""eval.kid — distributional metrics with strict paired-set enforcement (#6/#7/#8).

References:
  - REAL COCO   : 절대 생성 품질 (FID/KID vs 원본)
  - dense-50    : teacher-trajectory fidelity (다른 의미)
raw + composited 분리. 모든 집합을 raw stem 교집합으로 강제하고 크기 동일성 assert.

    python -m eval.kid --run .../mbd --manifest .../coco_manifest_1024.json \
        --resolution 1024 --dense-ref .../dense_s50 --out fidkid.json
"""
import argparse, hashlib, json, shutil, tempfile
from pathlib import Path

from PIL import Image


def _raw_stems(run: Path) -> set:
    return {p.stem for p in run.glob("*.png") if not p.stem.endswith("_pasted")}


def _stage(src: Path, tmp: Path, pasted: bool, stems: set) -> int:
    tmp.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.glob("*.png"):
        is_p = p.stem.endswith("_pasted")
        if pasted != is_p:
            continue
        stem = p.stem.replace("_pasted", "")
        if stem not in stems:                      # #6: 교집합 강제
            continue
        shutil.copy(p, tmp / f"{stem}.png")
        n += 1
    return n


def _stage_real(manifest, resolution, tmp: Path, stems: set) -> int:
    """#4: 생성 파이프라인과 동일한 전처리(load_image_rgb 공유)를 강제 —
    real reference가 전처리 차이를 측정하지 않도록."""
    from data.dataset import load_image_rgb
    tmp.mkdir(parents=True, exist_ok=True)
    items = json.load(open(manifest))["items"]
    n = 0
    for it in items:
        stem = Path(it["sample_id"]).stem
        if stem not in stems:
            continue
        load_image_rgb(it["image"], resolution).save(tmp / f"{stem}.png")
        n += 1
    return n


def _set_hash(stems) -> str:
    return hashlib.sha256("\n".join(sorted(stems)).encode()).hexdigest()[:16]


def _kidfid(a_dir, b_dir):
    from cleanfid import fid
    return fid.compute_fid(str(a_dir), str(b_dir)), fid.compute_kid(str(a_dir), str(b_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--dense-ref", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run = Path(a.run)
    stems = _raw_stems(run)
    if not stems:                                  # #7: empty 방어
        raise RuntimeError(f"No raw PNG files in {run}")
    # dense-ref도 같은 stem을 모두 가져야 paired benchmark 성립
    if a.dense_ref:
        dstems = _raw_stems(Path(a.dense_ref))
        missing = stems - dstems
        if missing:
            raise RuntimeError(
                f"dense-ref missing {len(missing)} stems (e.g. {sorted(missing)[:3]})")

    res = {"eval_set_size": len(stems), "eval_set_hash": _set_hash(stems)}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw, comp, real = td / "raw", td / "comp", td / "real"
        n_raw = _stage(run, raw, False, stems)
        n_comp = _stage(run, comp, True, stems)
        n_real = _stage_real(a.manifest, a.resolution, real, stems)

        if n_real != n_raw:                        # #7
            raise RuntimeError(f"real/gen mismatch: gen={n_raw} real={n_real} "
                               "(manifest에 없는 stem이 run에 있음)")
        if n_comp not in (0, n_raw):
            raise RuntimeError(f"composited 불완전: raw={n_raw} comp={n_comp}")

        f, k = _kidfid(raw, real)
        res.update(fid_vs_real_raw=f, kid_vs_real_raw=k)
        if n_comp == n_raw:
            f, k = _kidfid(comp, real)
            res.update(fid_vs_real_composited=f, kid_vs_real_composited=k)
        if a.dense_ref:
            dref = td / "dref"
            n_d = _stage(Path(a.dense_ref), dref, False, stems)
            assert n_d == n_raw, f"dense-ref staged {n_d} != {n_raw}"
            f, k = _kidfid(raw, dref)
            res.update(fid_vs_dense50=f, kid_vs_dense50=k)

        res.update(n_raw=n_raw, n_comp=n_comp, n_real=n_real)

    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
