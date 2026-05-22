#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7l: Refscale sensitivity check for ECG-NO dual-core R12.

This is a lightweight follow-up to Step7k.  It does not build new tensors.
By default it reuses the compact records saved in
step7k_refscale_weighted_dual_core.json, and only falls back to reading Step7c
npz files if that JSON is missing.

The goal is to test whether the experimental
refscale_inverse_excluded_occ_sum row is sensitive to the exact choice of the
large ECG-NO reference scale.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from step7i_residual_weighted_dual_space import DEFAULT_REFERENCE_ECG14, case_record, clipped01, parse_core_tokens, safe_ratio
from step7j_scan_residual_weights import parse_fitn
from step7k_refscale_weighted_dual_core import case_family, sorted_inputs


DEFAULT_SCALE_SPECS = [
    ("spd_fitN3", "spd_s012_p01_d0", [3]),
    ("spd_fitN5", "spd_s012_p01_d0", [5]),
    ("spd_fitN7", "spd_s012_p01_d0", [7]),
    ("spd_fitN9", "spd_s012_p01_d0", [9]),
    ("spd_fitN579_avg", "spd_s012_p01_d0", [5, 7, 9]),
    ("spd_fitN3579_avg", "spd_s012_p01_d0", [3, 5, 7, 9]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records-json", default="step7k_refscale_weighted_dual_core.json")
    p.add_argument("--inputs", nargs="*", default=None)
    p.add_argument("--glob", default="step7c_*fitN*_r12only_step4b_like.npz")
    p.add_argument("--core", default="s0,s1,s2,p0")
    p.add_argument("--target-families", default="sp_s012_p0,sp_s012_p01,spd_s012_p01_d0")
    p.add_argument("--scale-multipliers", default="0.8,0.9,1.0,1.1,1.2")
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--out-json", default="step7l_refscale_sensitivity.json")
    p.add_argument("--out-csv", default="step7l_refscale_sensitivity.csv")
    p.add_argument("--summary", default="step7l_refscale_sensitivity_summary.txt")
    return p.parse_args()


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_records(args) -> tuple[List[Dict[str, Any]], str]:
    path = Path(args.records_json)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload["records"], str(path)

    inputs = sorted_inputs(args)
    records = [case_record(path, parse_core_tokens(args.core)) for path in inputs]
    return records, "recomputed_from_npz"


def scale_from_records(records: List[Dict[str, Any]], family: str, fitns: List[int]) -> Dict[str, float]:
    selected = [r for r in records if case_family(r["case"]) == family and parse_fitn(r["path"]) in fitns]
    if not selected:
        raise SystemExit(f"No scale records for family={family!r}, fitNs={fitns}")
    occ = np.array([r["excluded_occ_sum"] for r in selected], dtype=float)
    trace = np.array([r["core_missing_trace_dm1"] for r in selected], dtype=float)
    return {
        "excluded_occ_sum_ref": float(np.mean(occ)),
        "excluded_occ_sum_ref_std": float(np.std(occ)),
        "missing_trace_ref": float(np.mean(trace)),
        "missing_trace_ref_std": float(np.std(trace)),
        "nscale": len(selected),
    }


def build_rows(records: List[Dict[str, Any]], target_families: List[str], reference_energy: float, multipliers: List[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    target_records = [r for r in records if case_family(r["case"]) in target_families]
    for scale_name, scale_family, scale_fitns in DEFAULT_SCALE_SPECS:
        scale = scale_from_records(records, scale_family, scale_fitns)
        for mult in multipliers:
            scaled_ref = mult * scale["excluded_occ_sum_ref"]
            for r in target_records:
                weight = clipped01(1.0 - safe_ratio(r["excluded_occ_sum"], scaled_ref))
                delta = weight * r["core_delta"]
                E_total = r["E_obs"] + delta
                residual = E_total - reference_energy
                rows.append(
                    {
                        "scale_name": scale_name,
                        "scale_family": scale_family,
                        "scale_fitNs": ",".join(str(x) for x in scale_fitns),
                        "scale_multiplier": mult,
                        "excluded_occ_sum_ref": scale["excluded_occ_sum_ref"],
                        "scaled_excluded_occ_sum_ref": scaled_ref,
                        "excluded_occ_sum_ref_std": scale["excluded_occ_sum_ref_std"],
                        "target_family": case_family(r["case"]),
                        "case": r["case"],
                        "fitN": parse_fitn(r["path"]),
                        "nqubits": r["nqubits"],
                        "E_obs": r["E_obs"],
                        "weight": weight,
                        "core_raw_delta_mEh": 1000.0 * r["core_delta"],
                        "weighted_delta_mEh": 1000.0 * delta,
                        "E_total": E_total,
                        "residual_to_reference_mEh": 1000.0 * residual,
                        "abs_residual_to_reference_mEh": 1000.0 * abs(residual),
                        "excluded_occ_sum": r["excluded_occ_sum"],
                        "core_missing_trace_dm1": r["core_missing_trace_dm1"],
                    }
                )
    return rows


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    keys = sorted({(r["target_family"], r["fitN"]) for r in rows})
    for family, fitn in keys:
        group = [r for r in rows if r["target_family"] == family and r["fitN"] == fitn]
        deltas = np.array([r["weighted_delta_mEh"] for r in group], dtype=float)
        weights = np.array([r["weight"] for r in group], dtype=float)
        residuals = np.array([r["abs_residual_to_reference_mEh"] for r in group], dtype=float)
        out.append(
            {
                "target_family": family,
                "fitN": fitn,
                "nscale_choices": len(group),
                "delta_mean_mEh": float(np.mean(deltas)),
                "delta_std_mEh": float(np.std(deltas)),
                "delta_min_mEh": float(np.min(deltas)),
                "delta_max_mEh": float(np.max(deltas)),
                "delta_span_mEh": float(np.max(deltas) - np.min(deltas)),
                "weight_mean": float(np.mean(weights)),
                "weight_span": float(np.max(weights) - np.min(weights)),
                "abs_residual_max_mEh": float(np.max(residuals)),
            }
        )
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args, records_source: str, rows: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> None:
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "records_source": records_source,
                "reference_energy": args.reference_energy,
                "scale_specs": [
                    {"name": name, "family": family, "fitNs": fitns}
                    for name, family, fitns in DEFAULT_SCALE_SPECS
                ],
                "scale_multipliers": parse_float_list(args.scale_multipliers),
                "rows": rows,
                "stats": stats,
                "notes": [
                    "This check reuses existing data and does not build new tensors.",
                    "Small delta spans across scale choices indicate weak refscale sensitivity.",
                ],
            },
            f,
            indent=2,
        )
    write_csv(args.out_csv, rows)

    lines = []
    lines.append("=" * 122)
    lines.append("Step 7l | Refscale sensitivity of experimental ECG-NO dual-core R12")
    lines.append("=" * 122)
    lines.append(f"records source = {records_source}")
    lines.append(f"reference      = {args.reference_energy:.14f} Eh")
    lines.append("")
    lines.append("[Sensitivity across scale choices]")
    lines.append("target_family             fitN  delta_mean  delta_span  weight_span  abs_resid_max  (all mEh except weight)")
    for st in stats:
        lines.append(
            f"{st['target_family']:<24s} {st['fitN']:>4d} "
            f"{st['delta_mean_mEh']:>11.6f} {st['delta_span_mEh']:>11.6f} "
            f"{st['weight_span']:>11.6f} {st['abs_residual_max_mEh']:>14.6f}"
        )
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out_json}")
    lines.append(f"  {args.out_csv}")
    lines.append(f"  {args.summary}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    records, source = load_records(args)
    rows = build_rows(records, parse_csv_list(args.target_families), args.reference_energy, parse_float_list(args.scale_multipliers))
    stats = summarize(rows)
    write_outputs(args, source, rows, stats)


if __name__ == "__main__":
    main()
