#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7i: Residual-weighted dual-space ECG-NO R12 prototypes.

This script starts from the Step7h Scheme-A split:

    OBS_energy  : full ECG-NO variational/RDM space
    OBS_R12proj : compact fixed core used in the R12 projector
    passive_R12 : remaining ECG-NO OBS orbitals plus CABS

and tests several scalar weights for damping the fixed-core R12 correction as
the variational ECG-NO space approaches the reference.  These are prototypes,
not production formulas.  The table intentionally keeps the full-active
CABS-only correction beside the weighted variants so overcorrection is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from typing import Any, Dict, List

import numpy as np

from r12_correction import block4
from step7g_audit_ecg_no_r12_subterms import DEFAULT_INPUTS, load_case, signed_subterms
from step7h_dual_space_projector_prototype import load_labels, parse_core_tokens


DEFAULT_REFERENCE_ECG14 = -2.9017962843565535


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--core", default="s0,s1,s2,p0", help="Fixed R12 projector core labels.")
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--out-json", default="step7i_residual_weighted_dual_space.json")
    p.add_argument("--out-csv", default="step7i_residual_weighted_dual_space.csv")
    p.add_argument("--summary", default="step7i_residual_weighted_dual_space_summary.txt")
    return p.parse_args()


def label_matches_core(label: str, tokens: List[str]) -> bool:
    m = re.search(r"ecg-([spdf])(\d+)_m\d+", label.lower())
    if not m:
        return False
    return f"{m.group(1)}{m.group(2)}" in tokens


def fixed_core_indices(labels: List[str], tokens: List[str], nobs: int) -> List[int]:
    core = [i for i, lab in enumerate(labels) if label_matches_core(lab, tokens)]
    return core if core else list(range(min(6, nobs)))


def clipped01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def safe_ratio(num: float, den: float) -> float:
    return 0.0 if abs(den) < 1.0e-15 else float(num / den)


def coupling_norm(r_phys: np.ndarray, active: List[int], passive: List[int]) -> float:
    """Frobenius norm of the explicit passive-active F12 coupling block.

    This is only a scalar diagnostic for tapering; it is not a paper formula.
    """
    if not active or not passive:
        return 0.0
    blk = block4(r_phys, passive, active, active, active)
    return float(np.linalg.norm(blk.ravel()))


def case_record(path: str, core_tokens: List[str]) -> Dict[str, Any]:
    case = load_case(path)
    nobs = case["nobs"]
    nri = case["nri"]
    obs = list(range(nobs))
    cabs = list(range(nobs, nri))
    labels = load_labels(path, nobs)
    core = fixed_core_indices(labels, core_tokens, nobs)
    core_passive = sorted([i for i in obs if i not in core] + cabs)

    full_sub = signed_subterms(
        case["g_phys"],
        case["r_phys"],
        case["fock"],
        case["dm1_obs"],
        case["dm2_obs"],
        obs,
        cabs,
        nri,
    )
    core_sub = signed_subterms(
        case["g_phys"],
        case["r_phys"],
        case["fock"],
        case["dm1_obs"],
        case["dm2_obs"],
        core,
        core_passive,
        nri,
    )

    occ = np.array(case["natural_occupations"], dtype=float)
    excluded = [i for i in obs if i not in core]
    cabs_norm_full = coupling_norm(case["r_phys"], obs, cabs)
    cabs_norm_core = coupling_norm(case["r_phys"], core, core_passive)

    return {
        "case": case["label"],
        "path": path,
        "nobs": nobs,
        "nri": nri,
        "nqubits": 2 * nobs,
        "E_obs": case["E_obs"],
        "core_indices": core,
        "core_labels": [labels[i] for i in core],
        "excluded_indices": excluded,
        "excluded_labels": [labels[i] for i in excluded],
        "excluded_occ_sum": float(np.sum(occ[excluded])) if excluded else 0.0,
        "excluded_occ_max": float(np.max(occ[excluded])) if excluded else 0.0,
        "full_delta": full_sub["correction_total"],
        "core_delta": core_sub["correction_total"],
        "core_active_trace_dm1": core_sub["active_trace_dm1"],
        "core_missing_trace_dm1": 2.0 - core_sub["active_trace_dm1"],
        "cabs_coupling_norm_full_active": cabs_norm_full,
        "cabs_coupling_norm_core_projector": cabs_norm_core,
        "V_full": full_sub["V_total"],
        "B_full": full_sub["B_total"],
        "X_full": full_sub["X_total"],
        "Delta_full": full_sub["Delta_total"],
        "V_core": core_sub["V_total"],
        "B_core": core_sub["B_total"],
        "X_core": core_sub["X_total"],
        "Delta_core": core_sub["Delta_total"],
    }


def make_weight_rows(records: List[Dict[str, Any]], reference_energy: float) -> List[Dict[str, Any]]:
    max_gap = max(max(0.0, r["E_obs"] - reference_energy) for r in records)
    max_missing_trace = max(r["core_missing_trace_dm1"] for r in records)
    max_excluded_occ = max(r["excluded_occ_sum"] for r in records)
    max_full_cabs_norm = max(r["cabs_coupling_norm_full_active"] for r in records)

    rows: List[Dict[str, Any]] = []
    for r in records:
        gap = max(0.0, r["E_obs"] - reference_energy)
        full_delta = r["full_delta"]
        core_delta = r["core_delta"]
        weights = {
            "full_active_cabs_only": 1.0,
            "dual_core_raw": 1.0,
            "gap_fraction_of_first_case": clipped01(safe_ratio(gap, max_gap)),
            "gap_cap_to_reference": clipped01(safe_ratio(gap, abs(core_delta))),
            "full_active_over_core_magnitude": clipped01(safe_ratio(abs(full_delta), abs(core_delta))),
            "inverse_missing_trace": clipped01(1.0 - safe_ratio(r["core_missing_trace_dm1"], max_missing_trace)),
            "inverse_excluded_occ_sum": clipped01(1.0 - safe_ratio(r["excluded_occ_sum"], max_excluded_occ)),
            "full_cabs_coupling_norm_fraction": clipped01(safe_ratio(r["cabs_coupling_norm_full_active"], max_full_cabs_norm)),
        }

        for mode, weight in weights.items():
            if mode == "full_active_cabs_only":
                weighted_delta = full_delta
            elif mode == "dual_core_raw":
                weighted_delta = core_delta
            else:
                weighted_delta = weight * core_delta
            E_total = r["E_obs"] + weighted_delta
            resid = E_total - reference_energy
            rows.append(
                {
                    "case": r["case"],
                    "nqubits": r["nqubits"],
                    "mode": mode,
                    "weight": weight,
                    "E_obs": r["E_obs"],
                    "reference_energy": reference_energy,
                    "gap_to_reference_mEh": 1000.0 * gap,
                    "full_active_delta_mEh": 1000.0 * full_delta,
                    "core_raw_delta_mEh": 1000.0 * core_delta,
                    "weighted_delta_mEh": 1000.0 * weighted_delta,
                    "E_total": E_total,
                    "residual_to_reference_mEh": 1000.0 * resid,
                    "abs_residual_to_reference_mEh": 1000.0 * abs(resid),
                    "core_missing_trace_dm1": r["core_missing_trace_dm1"],
                    "excluded_occ_sum": r["excluded_occ_sum"],
                    "excluded_occ_max": r["excluded_occ_max"],
                    "cabs_coupling_norm_full_active": r["cabs_coupling_norm_full_active"],
                    "cabs_coupling_norm_core_projector": r["cabs_coupling_norm_core_projector"],
                    "core_labels": ";".join(r["core_labels"]),
                    "excluded_labels": ";".join(r["excluded_labels"]),
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


def write_outputs(args, records: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    payload = {
        "reference_energy": args.reference_energy,
        "core": args.core,
        "records": records,
        "rows": rows,
        "notes": [
            "Weighted dual-space rows are prototypes, not final SF-[2]R12 formulas.",
            "gap_cap_to_reference uses the supplied reference energy and is therefore a calibration diagnostic.",
            "Trace/occupation weights are included to show whether simple projector-residual scalars have the right trend.",
        ],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(args.out_csv, rows)

    modes = [
        "full_active_cabs_only",
        "dual_core_raw",
        "gap_fraction_of_first_case",
        "gap_cap_to_reference",
        "full_active_over_core_magnitude",
        "inverse_missing_trace",
        "inverse_excluded_occ_sum",
        "full_cabs_coupling_norm_fraction",
    ]
    by_key = {(r["case"], r["mode"]): r for r in rows}
    lines = []
    lines.append("=" * 128)
    lines.append("Step 7i | Residual-weighted dual-space ECG-NO R12 prototypes")
    lines.append("=" * 128)
    lines.append(f"reference = {args.reference_energy:.14f} Eh")
    lines.append(f"fixed core = {args.core}")
    lines.append("")
    lines.append("case                         qubits mode                              weight     dE(mEh)       E_total             resid_ref(mEh)")
    for rec in records:
        for mode in modes:
            row = by_key[(rec["case"], mode)]
            lines.append(
                f"{row['case']:<28s} {row['nqubits']:>6d} {mode:<33s} "
                f"{row['weight']:>8.5f} {row['weighted_delta_mEh']:>11.6f} "
                f"{row['E_total']: .14f} {row['residual_to_reference_mEh']:>13.6f}"
            )
        lines.append("")
    lines.append("[Interpretation Guardrail]")
    lines.append("Only full_active_cabs_only is the current conservative CABS-only SF-[2]R12 candidate.")
    lines.append("The weighted dual-core modes are calibration/audit rows for choosing a future ECG-NO active-projector policy.")
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
    core_tokens = parse_core_tokens(args.core)
    records = [case_record(path, core_tokens) for path in args.inputs]
    rows = make_weight_rows(records, args.reference_energy)
    write_outputs(args, records, rows)


if __name__ == "__main__":
    main()
