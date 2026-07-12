"""tools.validate_run_compat — 재사용 전 config 호환성 검사 (#9).
비교할 run들의 run.json config가 manifest/steps/limit/resolution/guidance/
seed_offset에서 일치하는지 확인. 하나라도 다르면 비교 무효이므로 에러.

    python -m tools.validate_run_compat --manifest MAN --limit 500 --guidance 30 \
        run_a run_b run_c
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()

    import hashlib
    def _sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    keys = ["steps", "limit", "guidance", "seed_offset"]
    want = {"steps": a.steps, "limit": a.limit,
            "guidance": a.guidance, "seed_offset": a.seed_offset}
    man_sha = _sha(a.manifest) if a.manifest else None
    want_res = (json.load(open(a.manifest)).get("resolution")
                if a.manifest else None)
    bad = False
    for r in a.runs:
        cfg = json.load(open(Path(r) / "run.json")).get("config", {})
        for k in keys:
            if want[k] is None:
                continue
            got = cfg.get(k)
            if got != want[k]:
                print(f"MISMATCH {Path(r).name}: {k}={got!r} != {want[k]!r}")
                bad = True
        # manifest: sha 우선(강한 검사), 구버전 run은 basename fallback + WARN
        if man_sha:
            got_sha = cfg.get("manifest_sha256")
            if got_sha:
                if got_sha != man_sha:
                    print(f"MISMATCH {Path(r).name}: manifest_sha256 다름")
                    bad = True
            else:
                got_name = Path(cfg.get("manifest", "")).name
                if got_name != Path(a.manifest).name:
                    print(f"MISMATCH {Path(r).name}: manifest={got_name!r}")
                    bad = True
                else:
                    print(f"WARN {Path(r).name}: manifest sha 미기록(구버전 run) — "
                          "basename만 일치 확인")
        # resolution: 기록된 run만 비교, 없으면 WARN
        if want_res is not None:
            got_res = cfg.get("resolution")
            if got_res is None:
                print(f"WARN {Path(r).name}: resolution 미기록(구버전 run)")
            elif got_res != want_res:
                print(f"MISMATCH {Path(r).name}: resolution={got_res} != {want_res}")
                bad = True

    # provenance STRICT 비교: 기록된 run들 사이에서 상호 일치해야 (fix3 #1).
    # 구버전 run(미기록)은 WARN — 단 "최종 논문 표/5K는 새 provenance로 재생성" 권장.
    STRICT_PROV = ["model_revision", "transformer_config_sha256",
                   "scheduler_class", "scheduler_config_sha256", "code_commit"]
    provs = {}
    for r in a.runs:
        cfg = json.load(open(Path(r) / "run.json")).get("config", {})
        p = cfg.get("provenance")
        if p is None:
            print(f"WARN {Path(r).name}: provenance 미기록(구버전 run)")
        else:
            provs[Path(r).name] = p
            if p.get("git_dirty"):
                print(f"WARN {Path(r).name}: git_dirty=True (커밋 안 된 코드로 생성)")
    if len(provs) >= 2:
        names = list(provs)
        ref_n, ref_p = names[0], provs[names[0]]
        for n in names[1:]:
            for k in STRICT_PROV:
                if provs[n].get(k) != ref_p.get(k):
                    print(f"MISMATCH {n}: provenance.{k} != {ref_n}의 값")
                    bad = True
    if bad:
        print("\n호환성 검사 실패 — 이 run들을 함께 비교하면 안 됩니다.")
        sys.exit(1)
    print(f"OK: {len(a.runs)} runs share manifest/steps/limit/guidance/seed.")


if __name__ == "__main__":
    main()
