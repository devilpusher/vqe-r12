#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7j: Scan residual-weighted dual-space ECG-NO R12 stability.

This is a focused follow-up to Step7i.  It reuses existing Step7c r12-only
files and compares the two non-reference residual tapers that looked plausible:

* inverse_missing_trace
* inverse_excluded_occ_sum

The scan intentionally keeps full-active CABS-only and raw fixed-core rows as
baselines.  Weighted rows remain research diagnostics, not production SF-[2]R12
formulas.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from step7i_residual_weighted_dual_space import (
    DEFAULT_REFERENCE_ECG14,
    case_record,
    clipped01,
    parse_core_tokens,
    safe_ratio,
)


FOCUS_MODES = [
    "full_active_cabs_only",
    "dual_core_raw",
    "inverse_missing_trace",
    "inverse_excluded_occ_sum",
    "inverse_trace_occ_average",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=None)
    p.add_argument("--glob", default="step7c_*fitN*_r12only_step4b_like.npz")
    p.add_argument("--core", default="s0,s1,s2,p0")
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--out-json", default="step7j_residual_weight_scan.json")
    p.add_argument("--out-csv", default="step7j_residual_weight_scan.csv")
    p.add_argument("--stats-csv", default="step7j_residual_weight_scan_stats.csv")
    p.add_argument("--summary", default="step7j_residual_weight_scan_summary.txt")
    return p.parse_args()


def parse_fitn(path: str) -> int:
    m = re.search(r"fitN(\d+)", Path(path).name)
    return int(m.group(1)) if m else -1


def case_family(label: str) -> str:
    return re.sub(r"_fitN\d+$", "", label)


def sorted_inputs(args) -> List[str]:
    paths = args.inputs if args.inputs else [str(p) for p in Path(".").glob(args.glob)]
    return sorted(paths, key=lambda p: (case_family(Path(p).name), parse_fitn(p), Path(p).name))


def weights_for_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    max_missing_trace = max(r["core_missing_trace_dm1"] for r in records)
    max_excluded_occ = max(r["excluded_occ_sum"] for r in records)
    out: Dict[str, Dict[str, float]] = {}
    for r in records:
        w_trace = clipped01(1.0 - safe_ratio(r["core_missing_trace_dm1"], max_missing_trace))
        w_occ = clipped01(1.0 - safe_ratio(r["excluded_occ_sum"], max_excluded_occ))
        out[r["path"]] = {
            "full_active_cabs_only": 1.0,
            "dual_core_raw": 1.0,
            "inverse_missing_trace": w_trace,
            "inverse_excluded_occ_sum": w_occ,
            "inverse_trace_occ_average": 0.5 * (w_trace + w_occ),
        }
    return out


def build_rows(records: List[Dict[str, Any]], reference_energy: float) -> List[Dict[str, Any]]:
    weights = weights_for_records(records)
    rows: List[Dict[str, Any]] = []
    for r in records:
        fitn = parse_fitn(r["path"])
        family = case_family(r["case"])
        gap = max(0.0, r["E_obs"] - reference_energy)
        for mode in FOCUS_MODES:
            if mode == "full_active_cabs_only":
                delta = r["full_delta"]
                raw_source = "full_active"
            elif mode == "dual_core_raw":
                delta = r["core_delta"]
                raw_source = "fixed_core"
            else:
                delta = weights[r["path"]][mode] * r["core_delta"]
                raw_source = "weighted_fixed_core"
            E_total = r["E_obs"] + delta
            resid = E_total - reference_energy
            rows.append(
                {
                    "family": family,
                    "case": r["case"],
                    "fitN": fitn,
                    "nqubits": r["nqubits"],
                    "mode": mode,
                    "raw_source": raw_source,
                    "weight": weights[r["path"]][mode],
                    "E_obs": r["E_obs"],
                    "reference_energy": reference_energy,
                    "gap_to_reference_mEh": 1000.0 * gap,
                    "full_active_delta_mEh": 1000.0 * r["full_delta"],
                    "core_raw_delta_mEh": 1000.0 * r["core_delta"],
                    "weighted_delta_mEh": 1000.0 * delta,
                    "E_total": E_total,
                    "residual_to_reference_mEh": 1000.0 * resid,
                    "abs_residual_to_reference_mEh": 1000.0 * abs(resid),
                    "core_missing_trace_dm1": r["core_missing_trace_dm1"],
                    "excluded_occ_sum": r["excluded_occ_sum"],
                    "excluded_occ_max": r["excluded_occ_max"],
                    "cabs_coupling_norm_full_active": r["cabs_coupling_norm_full_active"],
                    "core_labels": ";".join(r["core_labels"]),
                    "excluded_labels": ";".join(r["excluded_labels"]),
                    "path": r["path"],
                }
            )
    return rows


def build_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["mode"])].append(row)
    stats = []
    for (family, mode), group in sorted(grouped.items()):
        deltas = np.array([r["weighted_delta_mEh"] for r in group], dtype=float)
        residuals = np.array([r["abs_residual_to_reference_mEh"] for r in group], dtype=float)
        weights = np.array([r["weight"] for r in group], dtype=float)
        stats.append(
            {
                "family": family,
                "mode": mode,
                "npoints": len(group),
                "fitNs": ",".join(str(r["fitN"]) for r in sorted(group, key=lambda x: x["fitN"])),
                "delta_mean_mEh": float(np.mean(deltas)),
                "delta_std_mEh": float(np.std(deltas)),
                "delta_min_mEh": float(np.min(deltas)),
                "delta_max_mEh": float(np.max(deltas)),
                "delta_span_mEh": float(np.max(deltas) - np.min(deltas)),
                "abs_residual_mean_mEh": float(np.mean(residuals)),
                "abs_residual_max_mEh": float(np.max(residuals)),
                "weight_mean": float(np.mean(weights)),
                "weight_std": float(np.std(weights)),
            }
        )
    return stats


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args, inputs: List[str], records: List[Dict[str, Any]], rows: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> None:
    payload = {
        "reference_energy": args.reference_energy,
        "core": args.core,
        "inputs": inputs,
        "records": records,
        "rows": rows,
        "stats": stats,
        "notes": [
            "inverse_missing_trace and inverse_excluded_occ_sum do not use the reference energy to set their weights.",
            "The reference energy is used only to report residual diagnostics.",
            "Weighted rows are research candidates; full_active_cabs_only remains the conservative production candidate.",
        ],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(args.out_csv, rows)
    write_csv(args.stats_csv, stats)

    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    lines = []
    lines.append("=" * 132)
    lines.append("Step 7j | Residual-weighted ECG-NO R12 stability scan")
    lines.append("=" * 132)
    lines.append(f"reference = {args.reference_energy:.14f} Eh")
    lines.append(f"fixed core = {args.core}")
    lines.append("")
    for family in sorted(by_family):
        lines.append(f"[{family}]")
        lines.append("fitN mode                         weight    dE(mEh)      E_total             resid_ref(mEh)")
        for row in sorted(by_family[family], key=lambda r: (r["fitN"], FOCUS_MODES.index(r["mode"]))):
            lines.append(
                f"{row['fitN']:>4d} {row['mode']:<28s} {row['weight']:>7.4f} "
                f"{row['weighted_delta_mEh']:>10.5f} {row['E_total']: .14f} "
                f"{row['residual_to_reference_mEh']:>13.6f}"
            )
        lines.append("")

    lines.append("[FitN Stability: delta span in mEh]")
    for st in stats:
        if st["mode"] in ("inverse_missing_trace", "inverse_excluded_occ_sum", "inverse_trace_occ_average"):
            lines.append(
                f"{st['family']:<24s} {st['mode']:<28s} "
                f"span={st['delta_span_mEh']:.6f} std={st['delta_std_mEh']:.6f} "
                f"abs_resid_max={st['abs_residual_max_mEh']:.6f}"
            )
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out_json}")
    lines.append(f"  {args.out_csv}")
    lines.append(f"  {args.stats_csv}")
    lines.append(f"  {args.summary}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    inputs = sorted_inputs(args)
    if not inputs:
        raise SystemExit(f"No inputs matched {args.glob!r}")
    core_tokens = parse_core_tokens(args.core)
    records = [case_record(path, core_tokens) for path in inputs]
    rows = build_rows(records, args.reference_energy)
    stats = build_stats(rows)
    write_outputs(args, inputs, records, rows, stats)


if __name__ == "__main__":
    main()
