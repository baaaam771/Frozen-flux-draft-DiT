"""tools.validate_run_compat — 재사용 전 config 호환성 검사 (#9).
비교할 run들의 run.json config가 manifest/limit/resolution/guidance/seed_offset
에서 일치하는지 확인 (steps는 arm마다 의도적으로 다를 수 있어 전역 비교 제외;
runtime schedule은 같은 step 그룹 내에서만 비교). 불일치 시 비교 무효 -> 에러.

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

    # steps는 GLOBAL 키가 아님: dense step-reduction baseline은 의도적으로 다른
    # step 수를 사용 (dense_s30=30). --steps는 이제 "50-step 그룹" 판별에만 사용.
    keys = ["limit", "guidance", "seed_offset"]
    want = {"limit": a.limit, "guidance": a.guidance, "seed_offset": a.seed_offset}
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
    # 전역: 모델·코드·scheduler BASE config는 전 arm 동일해야.
    # runtime schedule(timesteps/sigmas)은 같은 step 수 그룹에서만 동일해야.
    GLOBAL_PROV = ["pipeline_name_or_path", "transformer_name_or_path",
                   "model_revision", "transformer_config_sha256",
                   "scheduler_class", "scheduler_base_config_sha256",
                   "code_commit"]
    SCHEDULE_KEYS = ["timesteps_sha256", "sigmas_sha256"]
    provs, steps_of = {}, {}
    for r in a.runs:
        cfg = json.load(open(Path(r) / "run.json")).get("config", {})
        p = cfg.get("provenance")
        steps_of[Path(r).name] = cfg.get("steps")
        if p is None:
            print(f"WARN {Path(r).name}: provenance 미기록(구버전 run)")
        else:
            provs[Path(r).name] = p
            if p.get("git_dirty"):
                print(f"WARN {Path(r).name}: git_dirty=True "
                      "(smoke는 허용; 최종 표/5K는 clean commit에서 재생성)")
    if len(provs) >= 2:
        names = list(provs)
        ref_n, ref_p = names[0], provs[names[0]]
        for n in names[1:]:
            for k in GLOBAL_PROV:
                gv, rv = provs[n].get(k), ref_p.get(k)
                if gv is None or rv is None:
                    if gv != rv:
                        print(f"WARN {n}: provenance.{k} 한쪽 미기록(구/신 run 혼재)")
                    continue
                if gv != rv:
                    print(f"MISMATCH {n}: provenance.{k} != {ref_n}의 값")
                    if k == "scheduler_base_config_sha256":
                        da = provs[n].get("scheduler_base_config") or {}
                        db = ref_p.get("scheduler_base_config") or {}
                        for kk in sorted(set(da) | set(db)):
                            if da.get(kk) != db.get(kk):
                                print(f"    diff {kk}: {da.get(kk)!r} vs {db.get(kk)!r}")
                    bad = True
        # schedule: 같은 steps 그룹 내부에서만 비교
        from collections import defaultdict
        groups = defaultdict(list)
        for n in names:
            groups[steps_of.get(n)].append(n)
        for st, members in groups.items():
            if len(members) < 2:
                continue
            base = provs[members[0]]
            for n in members[1:]:
                for k in SCHEDULE_KEYS:
                    gv, rv = provs[n].get(k), base.get(k)
                    if gv is None or rv is None:
                        continue
                    if gv != rv:
                        print(f"MISMATCH {n}: {k} != {members[0]} "
                              f"(같은 {st}-step 그룹인데 schedule 다름)")
                        bad = True
    if bad:
        print("\n호환성 검사 실패 — 이 run들을 함께 비교하면 안 됩니다.")
        sys.exit(1)
    print(f"OK: {len(a.runs)} runs share "
          "manifest/limit/resolution/guidance/seed and compatible provenance.")


if __name__ == "__main__":
    main()
