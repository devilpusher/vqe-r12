#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7k: Reference-scale residual-weighted ECG-NO dual-core R12 row.

Step7j normalized residual weights by the largest residual in the scanned set.
That is useful for diagnosis but too implicit for a reproducible candidate row.

This script fixes the scale explicitly from a named reference ECG-NO space,
defaulting to spd_s012_p01_d0.  The candidate of interest is

    experimental_refscale_inverse_excluded_occ_sum

where

    weight = clip(1 - excluded_occ_sum / excluded_occ_sum_ref, 0, 1)
    deltaE = weight * deltaE_dual_core_raw

The weighted rows are still experimental.  full_active_cabs_only remains the
conservative production candidate.
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
from step7j_scan_residual_weights import build_stats, parse_fitn


MODES = [
    "full_active_cabs_only",
    "dual_core_raw",
    "experimental_refscale_inverse_excluded_occ_sum",
    "experimental_refscale_inverse_missing_trace",
    "experimental_refscale_inverse_average",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=None)
    p.add_argument("--glob", default="step7c_*fitN*_r12only_step4b_like.npz")
    p.add_argument("--core", default="s0,s1,s2,p0")
    p.add_argument("--scale-family", default="spd_s012_p01_d0")
    p.add_argument("--scale-fitNs", default="5,7,9", help="FitN values averaged to define the reference scale.")
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--out-json", default="step7k_refscale_weighted_dual_core.json")
    p.add_argument("--out-csv", default="step7k_refscale_weighted_dual_core.csv")
    p.add_argument("--stats-csv", default="step7k_refscale_weighted_dual_core_stats.csv")
    p.add_argument("--summary", default="step7k_refscale_weighted_dual_core_summary.txt")
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def case_family(label: str) -> str:
    return re.sub(r"_fitN\d+$", "", label)


def sorted_inputs(args) -> List[str]:
    paths = args.inputs if args.inputs else [str(p) for p in Path(".").glob(args.glob)]
    return sorted(paths, key=lambda p: (Path(p).name, parse_fitn(p)))


def reference_scales(records: List[Dict[str, Any]], scale_family: str, scale_fitns: List[int]) -> Dict[str, float]:
    selected = [
        r
        for r in records
        if case_family(r["case"]) == scale_family and parse_fitn(r["path"]) in scale_fitns
    ]
    if not selected:
        raise SystemExit(f"No records matched scale family {scale_family!r} and fitNs {scale_fitns}")
    excluded = np.array([r["excluded_occ_sum"] for r in selected], dtype=float)
    missing = np.array([r["core_missing_trace_dm1"] for r in selected], dtype=float)
    return {
        "excluded_occ_sum_ref": float(np.mean(excluded)),
        "excluded_occ_sum_ref_std": float(np.std(excluded)),
        "missing_trace_ref": float(np.mean(missing)),
        "missing_trace_ref_std": float(np.std(missing)),
        "nscale_points": len(selected),
    }


def build_rows(records: List[Dict[str, Any]], scales: Dict[str, float], reference_energy: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in records:
        fitn = parse_fitn(r["path"])
        family = case_family(r["case"])
        gap = max(0.0, r["E_obs"] - reference_energy)
        w_occ = clipped01(1.0 - safe_ratio(r["excluded_occ_sum"], scales["excluded_occ_sum_ref"]))
        w_trace = clipped01(1.0 - safe_ratio(r["core_missing_trace_dm1"], scales["missing_trace_ref"]))
        weights = {
            "full_active_cabs_only": 1.0,
            "dual_core_raw": 1.0,
            "experimental_refscale_inverse_excluded_occ_sum": w_occ,
            "experimental_refscale_inverse_missing_trace": w_trace,
            "experimental_refscale_inverse_average": 0.5 * (w_occ + w_trace),
        }
        for mode in MODES:
            if mode == "full_active_cabs_only":
                delta = r["full_delta"]
                raw_source = "full_active"
            elif mode == "dual_core_raw":
                delta = r["core_delta"]
                raw_source = "fixed_core"
            else:
                delta = weights[mode] * r["core_delta"]
                raw_source = "refscale_weighted_fixed_core"
            E_total = r["E_obs"] + delta
            residual = E_total - reference_energy
            rows.append(
                {
                    "family": family,
                    "case": r["case"],
                    "fitN": fitn,
                    "nqubits": r["nqubits"],
                    "mode": mode,
                    "raw_source": raw_source,
                    "weight": weights[mode],
                    "E_obs": r["E_obs"],
                    "reference_energy": reference_energy,
                    "gap_to_reference_mEh": 1000.0 * gap,
                    "full_active_delta_mEh": 1000.0 * r["full_delta"],
                    "core_raw_delta_mEh": 1000.0 * r["core_delta"],
                    "weighted_delta_mEh": 1000.0 * delta,
                    "E_total": E_total,
                    "residual_to_reference_mEh": 1000.0 * residual,
                    "abs_residual_to_reference_mEh": 1000.0 * abs(residual),
                    "excluded_occ_sum": r["excluded_occ_sum"],
                    "excluded_occ_sum_ref": scales["excluded_occ_sum_ref"],
                    "core_missing_trace_dm1": r["core_missing_trace_dm1"],
                    "missing_trace_ref": scales["missing_trace_ref"],
                    "core_labels": ";".join(r["core_labels"]),
                    "excluded_labels": ";".join(r["excluded_labels"]),
                    "path": r["path"],
                }
            )
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args, inputs: List[str], records: List[Dict[str, Any]], scales: Dict[str, float], rows: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> None:
    payload = {
        "reference_energy": args.reference_energy,
        "core": args.core,
        "scale_family": args.scale_family,
        "scale_fitNs": parse_int_list(args.scale_fitNs),
        "scales": scales,
        "inputs": inputs,
        "records": records,
        "rows": rows,
        "stats": stats,
        "notes": [
            "The experimental refscale rows do not use the reference energy to set their weights.",
            "The reference energy is used only to report residual diagnostics.",
            "full_active_cabs_only remains the conservative production candidate.",
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
    lines.append("=" * 138)
    lines.append("Step 7k | Refscale residual-weighted ECG-NO dual-core R12 candidate")
    lines.append("=" * 138)
    lines.append(f"reference energy = {args.reference_energy:.14f} Eh")
    lines.append(f"fixed core       = {args.core}")
    lines.append(f"scale family     = {args.scale_family}")
    lines.append(f"scale fitNs      = {args.scaleFitNs if hasattr(args, 'scaleFitNs') else args.scale_fitNs}")
    lines.append(f"excluded_occ_ref = {scales['excluded_occ_sum_ref']:.12e} +/- {scales['excluded_occ_sum_ref_std']:.3e}")
    lines.append(f"missing_trace_ref= {scales['missing_trace_ref']:.12e} +/- {scales['missing_trace_ref_std']:.3e}")
    lines.append("")
    for family in sorted(by_family):
        lines.append(f"[{family}]")
        lines.append("fitN mode                                             weight     dE(mEh)      E_total             resid_ref(mEh)")
        for row in sorted(by_family[family], key=lambda r: (r["fitN"], MODES.index(r["mode"]))):
            lines.append(
                f"{row['fitN']:>4d} {row['mode']:<48s} {row['weight']:>8.5f} "
                f"{row['weighted_delta_mEh']:>11.6f} {row['E_total']: .14f} "
                f"{row['residual_to_reference_mEh']:>13.6f}"
            )
        lines.append("")
    lines.append("[FitN Stability: delta span in mEh]")
    for st in stats:
        if st["mode"].startswith("experimental_refscale"):
            lines.append(
                f"{st['family']:<24s} {st['mode']:<48s} "
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
    records = [case_record(path, parse_core_tokens(args.core)) for path in inputs]
    scales = reference_scales(records, args.scale_family, parse_int_list(args.scale_fitNs))
    rows = build_rows(records, scales, args.reference_energy)
    stats = build_stats(rows)
    write_outputs(args, inputs, records, scales, rows, stats)


if __name__ == "__main__":
    main()
