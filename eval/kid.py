"""eval.kid — 분포 metric (KID, 소표본에서 unbiased; FID는 5k+ 필요해 유예).

run 디렉터리의 raw 출력 vs (a) dense-50 reference 출력, (b) 원본 입력 분포.
clean-fid의 folder-to-folder KID 사용. pasted 파일은 자동 제외.

    python -m eval.kid --run .../mbd_c2_r03_t4_dualkv --ref .../dense_s50 \
        --out kid.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def _stage_raw(src: Path, tmp: Path) -> int:
    tmp.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.glob("*.png"):
        if p.stem.endswith("_pasted"):
            continue
        shutil.copy(p, tmp / p.name)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from cleanfid import fid
    with tempfile.TemporaryDirectory() as td:
        ra, rb = Path(td) / "run", Path(td) / "ref"
        na, nb = _stage_raw(Path(a.run), ra), _stage_raw(Path(a.ref), rb)
        kid = fid.compute_kid(str(ra), str(rb))
    res = {"kid_run_vs_ref": kid, "n_run": na, "n_ref": nb}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
