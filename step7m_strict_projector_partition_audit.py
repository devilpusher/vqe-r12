#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7m: Strict, unweighted ECG-NO R12 projector partition audit.

This audit compares projector partitions that contain no empirical refscale or
occupation weight:

* full_active_cabs_only
* strict_fixed_core_lowNO_plus_CABS
* strict_occ_ge_T_lowNO_plus_CABS

The goal is to identify whether an unweighted active/passive partition can be a
strict ECG-NO+[2]R12 candidate, or whether the refscale-weighted row must remain
an empirical/extrapolative diagnostic.
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

from step7g_audit_ecg_no_r12_subterms import load_case, signed_subterms
from step7h_dual_space_projector_prototype import fixed_core_indices, load_labels, parse_core_tokens
from step7i_residual_weighted_dual_space import DEFAULT_REFERENCE_ECG14
from step7j_scan_residual_weights import parse_fitn


DEFAULT_HE_EXACT_NONREL = -2.9037243770341195


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=None)
    p.add_argument("--glob", default="step7c_*fitN*_r12only_step4b_like.npz")
    p.add_argument("--core", default="s0,s1,s2,p0")
    p.add_argument("--occ-thresholds", default="1e-5,5e-5,1e-4,2e-4,5e-4,1e-3")
    p.add_argument("--nocc", type=int, default=1, help="Always keep the first nocc spatial NOs active.")
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--exact-energy", type=float, default=DEFAULT_HE_EXACT_NONREL)
    p.add_argument("--out-json", default="step7m_strict_projector_partition_audit.json")
    p.add_argument("--out-csv", default="step7m_strict_projector_partition_audit.csv")
    p.add_argument("--stats-csv", default="step7m_strict_projector_partition_audit_stats.csv")
    p.add_argument("--summary", default="step7m_strict_projector_partition_audit_summary.txt")
    return p.parse_args()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def case_family(label: str) -> str:
    return re.sub(r"_fitN\d+$", "", label)


def sorted_inputs(args) -> List[str]:
    paths = args.inputs if args.inputs else [str(p) for p in Path(".").glob(args.glob)]
    return sorted(paths, key=lambda p: (case_family(Path(p).name), parse_fitn(p), Path(p).name))


def policies_for_case(case: Dict[str, Any], labels: List[str], args) -> List[Dict[str, Any]]:
    nobs = case["nobs"]
    nri = case["nri"]
    obs = list(range(nobs))
    cabs = list(range(nobs, nri))
    occ = np.array(case["natural_occupations"], dtype=float)
    core = fixed_core_indices(labels, parse_core_tokens(args.core))
    if not core:
        core = obs[: min(6, nobs)]

    policies = [
        {
            "policy": "full_active_cabs_only",
            "partition_type": "baseline",
            "threshold": None,
            "active": obs,
            "passive": cabs,
            "description": "All selected ECG-NO OBS orbitals are active; passive is CABS only.",
        },
        {
            "policy": "strict_fixed_core_lowNO_plus_CABS",
            "partition_type": "fixed_core",
            "threshold": None,
            "active": core,
            "passive": sorted([i for i in obs if i not in core] + cabs),
            "description": "Fixed high-occupation/core projector; all remaining ECG-NO OBS orbitals are passive with CABS.",
        },
    ]
    for th in parse_float_list(args.occ_thresholds):
        active = [i for i in obs if i < args.nocc or occ[i] >= th]
        passive = sorted([i for i in obs if i not in active] + cabs)
        policies.append(
            {
                "policy": f"strict_occ_ge_{th:g}_lowNO_plus_CABS",
                "partition_type": "occupation_threshold",
                "threshold": th,
                "active": active,
                "passive": passive,
                "description": f"NOs with occupation >= {th:g} remain active; lower-occupation NOs are passive with CABS.",
            }
        )
    return policies


def audit_case(path: str, args) -> List[Dict[str, Any]]:
    case = load_case(path)
    labels = load_labels(path, case["nobs"])
    rows = []
    for pol in policies_for_case(case, labels, args):
        sub = signed_subterms(
            case["g_phys"],
            case["r_phys"],
            case["fock"],
            case["dm1_obs"],
            case["dm2_obs"],
            pol["active"],
            pol["passive"],
            case["nri"],
        )
        delta = sub["correction_total"]
        E_total = case["E_obs"] + delta
        resid_ref = E_total - args.reference_energy
        above_exact = E_total - args.exact_energy
        missing_trace = 2.0 - sub["active_trace_dm1"]
        rows.append(
            {
                "family": case_family(case["label"]),
                "case": case["label"],
                "path": path,
                "fitN": parse_fitn(path),
                "nobs": case["nobs"],
                "nqubits": 2 * case["nobs"],
                "nri": case["nri"],
                "policy": pol["policy"],
                "partition_type": pol["partition_type"],
                "threshold": pol["threshold"],
                "description": pol["description"],
                "active_size": len(pol["active"]),
                "passive_size": len(pol["passive"]),
                "active_indices": pol["active"],
                "active_labels": [labels[i] for i in pol["active"]],
                "passive_obs_count": len([i for i in pol["passive"] if i < case["nobs"]]),
                "active_trace_dm1": sub["active_trace_dm1"],
                "active_trace_dm2": sub["active_trace_dm2"],
                "missing_trace_dm1": missing_trace,
                "E_obs": case["E_obs"],
                "delta_E_r12": delta,
                "delta_E_r12_mEh": 1000.0 * delta,
                "E_total": E_total,
                "reference_energy": args.reference_energy,
                "residual_to_reference_mEh": 1000.0 * resid_ref,
                "abs_residual_to_reference_mEh": 1000.0 * abs(resid_ref),
                "above_exact_mEh": 1000.0 * above_exact,
                "below_ionization_threshold_Eh": -2.0 - E_total,
                "overcorrects_reference": E_total < args.reference_energy - 1.0e-10,
                "below_exact_nonrel": E_total < args.exact_energy,
                "V_mEh": 1000.0 * sub["V_total"],
                "B_mEh": 1000.0 * sub["B_total"],
                "X_mEh": 1000.0 * sub["X_total"],
                "Delta_mEh": 1000.0 * sub["Delta_total"],
            }
        )
    return rows


def build_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["policy"])].append(row)
    stats = []
    for (family, policy), group in sorted(grouped.items()):
        deltas = np.array([r["delta_E_r12_mEh"] for r in group], dtype=float)
        abs_resid = np.array([r["abs_residual_to_reference_mEh"] for r in group], dtype=float)
        missing = np.array([r["missing_trace_dm1"] for r in group], dtype=float)
        stats.append(
            {
                "family": family,
                "policy": policy,
                "npoints": len(group),
                "fitNs": ",".join(str(r["fitN"]) for r in sorted(group, key=lambda x: x["fitN"])),
                "active_size": group[0]["active_size"],
                "passive_obs_count": group[0]["passive_obs_count"],
                "delta_mean_mEh": float(np.mean(deltas)),
                "delta_std_mEh": float(np.std(deltas)),
                "delta_min_mEh": float(np.min(deltas)),
                "delta_max_mEh": float(np.max(deltas)),
                "delta_span_mEh": float(np.max(deltas) - np.min(deltas)),
                "abs_residual_mean_mEh": float(np.mean(abs_resid)),
                "abs_residual_max_mEh": float(np.max(abs_resid)),
                "missing_trace_mean": float(np.mean(missing)),
                "missing_trace_max": float(np.max(missing)),
                "any_overcorrects_reference": any(r["overcorrects_reference"] for r in group),
                "any_below_exact_nonrel": any(r["below_exact_nonrel"] for r in group),
            }
        )
    return stats


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [k for k in rows[0].keys() if k not in ("active_indices", "active_labels")]
    fields += ["active_labels"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            if "active_labels" in row:
                out["active_labels"] = ";".join(row["active_labels"])
            writer.writerow(out)


def write_outputs(args, inputs: List[str], rows: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> None:
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": inputs,
                "reference_energy": args.reference_energy,
                "exact_energy": args.exact_energy,
                "core": args.core,
                "occ_thresholds": parse_float_list(args.occ_thresholds),
                "rows": rows,
                "stats": stats,
                "notes": [
                    "No empirical refscale or occupation weights are used in any strict policy.",
                    "Rows with active_trace_dm1 < 2 are projector partitions, not standalone variational energies.",
                    "A strict candidate should avoid large overcorrection while remaining stable across fitN.",
                ],
            },
            f,
            indent=2,
        )
    write_csv(args.out_csv, rows)
    write_csv(args.stats_csv, stats)

    lines = []
    lines.append("=" * 132)
    lines.append("Step 7m | Strict unweighted ECG-NO R12 projector partition audit")
    lines.append("=" * 132)
    lines.append(f"reference ECG14 = {args.reference_energy:.14f} Eh")
    lines.append(f"He exact nonrel  = {args.exact_energy:.14f} Eh")
    lines.append(f"core             = {args.core}")
    lines.append("")
    lines.append("[FitN stability by family/policy]")
    lines.append("family                   policy                                      act passOBS  dE_mean  dE_span  resid_max overRef belowExact")
    for st in stats:
        if st["policy"] in (
            "full_active_cabs_only",
            "strict_fixed_core_lowNO_plus_CABS",
            "strict_occ_ge_0.0001_lowNO_plus_CABS",
            "strict_occ_ge_0.0005_lowNO_plus_CABS",
            "strict_occ_ge_0.001_lowNO_plus_CABS",
        ):
            lines.append(
                f"{st['family']:<24s} {st['policy']:<43s} "
                f"{st['active_size']:>3d} {st['passive_obs_count']:>7d} "
                f"{st['delta_mean_mEh']:>8.3f} {st['delta_span_mEh']:>8.3f} "
                f"{st['abs_residual_max_mEh']:>9.3f} "
                f"{str(st['any_overcorrects_reference']):>7s} {str(st['any_below_exact_nonrel']):>10s}"
            )
    lines.append("")
    lines.append("[Guardrail]")
    lines.append("A strict unweighted partition that overcorrects the ECG14 reference is a diagnostic, not yet a final formula.")
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
    rows = []
    for path in inputs:
        rows.extend(audit_case(path, args))
    stats = build_stats(rows)
    write_outputs(args, inputs, rows, stats)


if __name__ == "__main__":
    main()
