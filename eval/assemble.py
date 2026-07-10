"""eval.assemble — builds the paper's core table + Pareto CSV (plan Sec. 22).

Fix 5: consumes the renamed *_to_ref metrics.
Fix 7: warm-up rows (first sample per run) are excluded from wall-clock means.
Fix 9: runs sharing an identical config signature (method/steps/c/r/selector/
       block) — e.g. the same setting at 3 seed offsets — are grouped and
       reported as mean ± std with n_runs.

    python -m eval.assemble --runs out_s0/mbfd_c3_r03 out_s1/mbfd_c3_r03 ... \
        --out table_main.md --csv pareto.csv
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

COLS = ["Method", "Steps", "Anchor c", "Ratio r(req)", "r(actual)", "Block",
        "Selector", "MaskLPIPS→ref", "BndLPIPS→ref", "LPIPS→ref",
        "KnownPSNR→input", "MACratio(est)", "Wall(s)", "VRAM(GB)",
        "Imgs", "Seeds"]

MET_KEYS = {"MaskLPIPS→ref": "mask_lpips_to_ref",
            "BndLPIPS→ref": "boundary_lpips_to_ref",
            "LPIPS→ref": "lpips_to_ref",
            "KnownPSNR→input": "known_psnr_to_input"}


def _load(run_dir: Path):
    run = json.load(open(run_dir / "run.json"))
    met_p = run_dir / "metrics.json"
    met_n = 0
    met = {}
    if met_p.exists():
        mj = json.load(open(met_p))
        met = mj["aggregate"]
        met_n = mj.get("n", 0)          # 평가된 이미지 수 (metrics.json 기준)
    cfg = run["config"]
    rows = [r for r in run["rows"] if not r.get("warmup")]      # Fix 7
    wall = sum(r["wall_s"] for r in rows) / max(len(rows), 1)
    vram = max((r.get("peak_vram_gb", 0) for r in run["rows"]), default=0)
    ratios = [r["mean_actual_ratio"] for r in run["rows"]
              if r.get("mean_actual_ratio") is not None]
    macs = [r["mean_est_transformer_mac_ratio"] for r in run["rows"]
            if r.get("mean_est_transformer_mac_ratio") is not None]
    sig = (cfg["method"], cfg["steps"],
           cfg.get("cache_period") if cfg["method"] != "dense" else None,
           cfg.get("ratio") if cfg["method"] == "cache_sparse" else None,
           cfg.get("block", 1) if cfg["method"] == "cache_sparse" else None,
           cfg.get("selector") if cfg["method"] == "cache_sparse" else None)
    return {"sig": sig, "wall": wall, "vram": vram, "met_n": met_n,
            "r_actual": statistics.mean(ratios) if ratios else None,
            "mac": statistics.mean(macs) if macs else None,
            "met": {k: met.get(v) for k, v in MET_KEYS.items()}}


def _fmt(vals, prec=4):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "-"
    m = statistics.mean(vals)
    if len(vals) > 1:
        return f"{m:.{prec}f}±{statistics.stdev(vals):.{prec}f}"
    return f"{m:.{prec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="", help="optional Pareto CSV (wall vs quality)")
    a = ap.parse_args()

    loaded = [_load(Path(r)) for r in a.runs]
    groups: dict[tuple, list] = {}
    for L in loaded:
        groups.setdefault(L["sig"], []).append(L)

    table, csv_rows = [], []
    for sig, Ls in groups.items():
        method, steps, c, r, block, selector = sig
        row = {"Method": method, "Steps": steps,
               "Anchor c": c if c is not None else "-",
               "Ratio r(req)": r if r is not None else "-",
               "r(actual)": _fmt([L["r_actual"] for L in Ls], 3),
               "Block": block if block is not None else "-",
               "Selector": selector if selector is not None else "-",
               "MACratio(est)": _fmt([L["mac"] for L in Ls], 3),
               "Wall(s)": _fmt([L["wall"] for L in Ls], 2),
               "VRAM(GB)": _fmt([L["vram"] for L in Ls], 1),
               # Imgs: metrics.json이 평가한 이미지 수 (seed별 동일해야 정상)
               "Imgs": "/".join(str(L["met_n"]) for L in Ls)
                       if len({L["met_n"] for L in Ls}) > 1 else Ls[0]["met_n"],
               "Seeds": len(Ls)}
        for col in MET_KEYS:
            row[col] = _fmt([L["met"][col] for L in Ls])
        table.append(row)
        walls = [L["wall"] for L in Ls]
        mlp = [L["met"]["MaskLPIPS→ref"] for L in Ls
               if L["met"]["MaskLPIPS→ref"] is not None]
        csv_rows.append([method, steps, c, r, block, selector,
                         statistics.mean(walls),
                         statistics.mean(mlp) if mlp else ""])

    lines = ["| " + " | ".join(COLS) + " |",
             "|" + "|".join("---" for _ in COLS) + "|"]
    for row in table:
        lines.append("| " + " | ".join(str(row[c]) for c in COLS) + " |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if a.csv:                                       # Pareto: wall vs mask LPIPS
        hdr = "method,steps,cache_period,ratio,block,selector,mean_wall_s,mean_mask_lpips_to_ref"
        body = "\n".join(",".join("" if v is None else str(v) for v in r)
                         for r in csv_rows)
        Path(a.csv).write_text(hdr + "\n" + body + "\n")
        print(f"pareto csv -> {a.csv}")


if __name__ == "__main__":
    main()
