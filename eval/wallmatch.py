"""eval.wallmatch — threshold sweep에서 목표 wall에 가장 가까운 점 선택 +
중복 execution pattern 제거 (#5).

    python -m eval.wallmatch --runs adapt_thresh_* --target-wall 12.12 --out sel.md
"""
import argparse, json
from pathlib import Path


def _wall(run: Path):
    rows = [r for r in json.load(open(run / "run.json"))["rows"] if not r.get("warmup")]
    return sum(r["wall_s"] for r in rows) / max(len(rows), 1)


def _pattern(run: Path):
    """#1: 전 non-warmup row에서 패턴 집계. threshold 정책은 sigma schedule만
    보므로 모든 이미지에서 동일해야 함 — 다르면 정책 버그이므로 assert."""
    rows = [r for r in json.load(open(run / "run.json"))["rows"]
            if not r.get("warmup")]
    pats = {(r.get("anchor_evals", 0), r.get("thresh_dense", 0),
             r.get("thresh_reuse", 0)) for r in rows}
    assert len(pats) == 1, (
        f"{run.name}: execution pattern이 이미지마다 다름 {pats} — "
        "sigma-only threshold 정책과 모순 (정책 버그 의심)")
    return next(iter(pats))


def _quality(run: Path):
    m = run / "metrics.json"
    return json.load(open(m))["aggregate"].get("mask_lpips_to_ref") if m.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--target-wall", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seen, uniq = {}, []
    for r in a.runs:
        rp = Path(r)
        pat = _pattern(rp)
        if pat in seen:                            # 중복 execution pattern 제거
            continue
        seen[pat] = r
        uniq.append((rp, _wall(rp), _quality(rp), pat))

    lines = [f"# temporal_thresh sweep (target wall {a.target_wall}s)\n",
             "| run | wall(s) | maskLPIPS | anchor/dense/reuse |",
             "|---|---|---|---|"]
    for rp, w, q, pat in sorted(uniq, key=lambda x: x[1]):
        q_txt = f"{q:.4f}" if q is not None else "-"          # #2: 0.0도 유효
        lines.append(f"| {rp.name} | {w:.2f} | {q_txt} | "
                     f"{pat[0]}/{pat[1]}/{pat[2]} |")
    best = min(uniq, key=lambda x: abs(x[1] - a.target_wall))
    best_q = f"{best[2]:.4f}" if best[2] is not None else "N/A"
    lines.append(f"\n**Wall-matched pick** ({a.target_wall}s): "
                 f"`{best[0].name}` at {best[1]:.2f}s, maskLPIPS {best_q}")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
