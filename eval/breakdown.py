"""eval.breakdown — Stage 7-A (무료 분석): 이미 저장된 metrics/run에서
(1) mask bucket × type별 성능 분해, (2) refresh가 reuse보다 나쁜 실패 샘플
목록화 (Fig.4 row-2 유형의 정량화).

    python -m eval.breakdown --runs seed0/mbd_c2_r03_t4_dualkv seed0/reuse_c2_t4 ... \
        --manifest data/coco_manifest_1024.json --out breakdown.md \
        [--failures seed0/mbd_c2_r03_t4_dualkv:seed0/reuse_c2_t4]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _per_sample(run_dir: Path) -> dict:
    met = json.load(open(run_dir / "metrics.json"))
    return {r["sample_id"]: r["mask_lpips_to_ref"] for r in met["rows"]
            if "mask_lpips_to_ref" in r}


def _sample_meta(manifest: str) -> dict:
    items = json.load(open(manifest))["items"]
    return {Path(it["sample_id"]).stem: (it["bucket"], it["mask_type"])
            for it in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--failures", default="",
                    help="'refresh_dir:reuse_dir' — refresh가 뒤지는 샘플 목록")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    meta = _sample_meta(a.manifest)
    lines = ["# Breakdown by mask bucket / type\n"]
    buckets = ["small", "medium", "large"]
    types = ["box", "brush", "polygon"]

    for run in a.runs:
        rd = Path(run)
        per = _per_sample(rd)
        lines.append(f"\n## {rd.name}  (n={len(per)})\n")
        lines.append("| | " + " | ".join(types) + " | all |")
        lines.append("|---|" + "---|" * (len(types) + 1))
        for b in buckets:
            row = [b]
            for t in types:
                v = [x for s, x in per.items() if meta.get(s) == (b, t)]
                row.append(f"{statistics.mean(v):.4f} (n={len(v)})" if v else "-")
            vb = [x for s, x in per.items() if meta.get(s, (None,))[0] == b]
            row.append(f"**{statistics.mean(vb):.4f}**" if vb else "-")
            lines.append("| " + " | ".join(row) + " |")
        va = list(per.values())
        lines.append(f"| all | | | | **{statistics.mean(va):.4f}** |")

    if a.failures:
        ref_dir, reuse_dir = a.failures.split(":")
        pr, pu = _per_sample(Path(ref_dir)), _per_sample(Path(reuse_dir))
        common = sorted(set(pr) & set(pu),
                        key=lambda s: pr[s] - pu[s], reverse=True)
        n_worse = sum(pr[s] > pu[s] for s in common)
        lines.append(f"\n# Refresh-vs-reuse failures "
                     f"({Path(ref_dir).name} > {Path(reuse_dir).name}: "
                     f"{n_worse}/{len(common)} samples)\n")
        lines.append("| sample | bucket/type | refresh | reuse | Δ |")
        lines.append("|---|---|---|---|---|")
        for s in common[:a.top]:
            b, t = meta.get(s, ("?", "?"))
            lines.append(f"| {s} | {b}/{t} | {pr[s]:.4f} | {pu[s]:.4f} "
                         f"| +{pr[s]-pu[s]:.4f} |")
        # 실패 샘플의 bucket 분포 (row-2 유형이 특정 조건에 몰리는지)
        worse = [s for s in common if pr[s] > pu[s]]
        lines.append("\n실패 샘플 분포: " + ", ".join(
            f"{b}/{t}: {sum(meta.get(s) == (b, t) for s in worse)}"
            for b in buckets for t in types
            if sum(meta.get(s) == (b, t) for s in worse)))

    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
