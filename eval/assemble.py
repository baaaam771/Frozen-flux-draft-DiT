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
        "Selector", "Tail", "KV", "Dual", "MaskLPIPS→ref", "BndLPIPS→ref", "LPIPS→ref",
        "KnownPSNR→input", "MACratio(est)", "Evals(a/s/d)", "Wall(s)",
        "VRAM(GB)", "Imgs", "Seeds"]

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
    # edge case: 이미지 저장 완료 직후 run.json 직전에 죽은 arm을 --skip-existing
    # 으로 재실행하면 rows가 빈 리스트가 됨 (fresh=0). wall은 None으로.
    wall = (sum(r["wall_s"] for r in rows) / len(rows)) if rows else None
    # resume된 run의 wall은 마지막 세션 fresh subset만 대표 — 표에 별표 표기.
    # (최종 latency 주장에는 clean 단일 세션 run 사용)
    wall_partial = bool(cfg.get("resumed"))
    vram = max((r.get("peak_vram_gb", 0) for r in run["rows"]), default=0)
    ratios = [r["mean_actual_ratio"] for r in run["rows"]
              if r.get("mean_actual_ratio") is not None]
    macs = [r["mean_est_transformer_mac_ratio"] for r in run["rows"]
            if r.get("mean_est_transformer_mac_ratio") is not None]
    tail = (cfg.get("dense_head", 0), cfg.get("dense_tail", 0)) \
        if cfg["method"] in ("reuse", "cache_sparse") else None
    # fix_t #1: teacache는 rel-L1 threshold가 운영점을 정의 — sig에 없으면
    # sweep의 4개 threshold가 같은 설정의 seed들처럼 평균돼 표가 오염됨.
    tc_thresh = (cfg.get("teacache_rel_l1")
                 if cfg["method"] == "teacache" else None)
    sig = (cfg["method"], cfg["steps"],
           cfg.get("cache_period") if cfg["method"] != "dense" else None,
           cfg.get("ratio") if cfg["method"] == "cache_sparse" else None,
           cfg.get("block", 1) if cfg["method"] == "cache_sparse" else None,
           cfg.get("selector") if cfg["method"] == "cache_sparse" else None,
           tail, bool(cfg.get("kv_cache", False)),
           bool(cfg.get("dual_sparse", False)), tc_thresh)
    if run["rows"]:
        r0 = next((r for r in run["rows"] if not r.get("warmup")),
                  run["rows"][0])
        evals = (r0.get("anchor_evals", 0), r0.get("sparse_steps", 0),
                 r0.get("thresh_dense", 0))
    else:
        evals = ("-", "-", "-")
    return {"sig": sig, "wall": wall, "wall_partial": wall_partial,
            "vram": vram, "met_n": met_n,
            "evals": evals,
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
        method, steps, c, r, block, selector, tail, kv, dual, tc_thresh = sig
        if method == "teacache":
            selector = f"rel-L1={tc_thresh}"
        row = {"Method": method, "Steps": steps,
               "Anchor c": c if c is not None else "-",
               "Ratio r(req)": r if r is not None else "-",
               "r(actual)": _fmt([L["r_actual"] for L in Ls], 3),
               "Block": block if block is not None else "-",
               "Selector": selector if selector is not None else "-",
               "KV": ("Y" if kv else "-"),
               "Dual": ("Y" if dual else "-"),
               "Tail": ("-" if tail is None else
                        (f"h{tail[0]}+t{tail[1]}" if tail[0] else f"t{tail[1]}")
                        if tail and (tail[0] or tail[1]) else
                        ("0" if tail is not None else "-")),
               "MACratio(est)": _fmt([L["mac"] for L in Ls], 3),
               "Wall(s)": _fmt([L["wall"] for L in Ls], 2)
                          + ("*" if any(L.get("wall_partial") for L in Ls)
                             else ""),
               # 실제 계산 횟수: anchor/sparse/threshold-dense (I/O 아닌 진짜 연산량 증거)
               "Evals(a/s/d)": "/".join(str(x) for x in Ls[0]["evals"]),
               # (아래 Wall은 partial 여부 별표 처리 위해 후처리)
               "VRAM(GB)": _fmt([L["vram"] for L in Ls], 1),
               # Imgs: metrics.json이 평가한 이미지 수 (seed별 동일해야 정상)
               "Imgs": "/".join(str(L["met_n"]) for L in Ls)
                       if len({L["met_n"] for L in Ls}) > 1 else Ls[0]["met_n"],
               "Seeds": len(Ls)}
        for col in MET_KEYS:
            row[col] = _fmt([L["met"][col] for L in Ls])
        table.append(row)
        walls = [L["wall"] for L in Ls if L["wall"] is not None]
        mlp = [L["met"]["MaskLPIPS→ref"] for L in Ls
               if L["met"]["MaskLPIPS→ref"] is not None]
        csv_rows.append([method, steps, c, r, block, selector,
                         statistics.mean(walls) if walls else "",
                         statistics.mean(mlp) if mlp else ""])

    lines = ["| " + " | ".join(COLS) + " |",
             "|" + "|".join("---" for _ in COLS) + "|"]
    for row in table:
        lines.append("| " + " | ".join(str(row[c]) for c in COLS) + " |")
    if any(L.get("wall_partial") for L in loaded):
        lines.append("")
        lines.append("\\* Wall: resume된 run — 마지막 세션 fresh 샘플만의 "
                     "평균이며 전체 run을 대표하지 않음 (latency 주장에는 "
                     "clean 단일 세션 결과 사용).")
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
