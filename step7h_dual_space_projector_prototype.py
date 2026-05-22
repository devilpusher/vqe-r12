#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7h: Prototype dual-space ECG-NO R12 projector policies.

Scheme A separates the variational/energy space from the R12 projector space:

    OBS_energy  : full ECG-NO space used for FCI/VQE and the reference energy
    OBS_R12proj : compact core/high-occupation subset used in the R12 projector
    passive_R12 : the remaining ECG-NO OBS orbitals plus CABS

This is an audit/prototype, not the final production correction.  The active
RDM traces are reported for every policy so any loss from truncating the
projector space is explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from step7g_audit_ecg_no_r12_subterms import DEFAULT_INPUTS, load_case, signed_subterms


DEFAULT_REFERENCE_ECG14 = -2.9017962843565535


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--core", default="s0,s1,s2,p0", help="Fixed core projector labels, e.g. s0,s1,s2,p0.")
    p.add_argument("--occ-thresholds", default="1e-4,5e-4,1e-3")
    p.add_argument("--nocc", type=int, default=1)
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--out-json", default="step7h_dual_space_projector_prototype.json")
    p.add_argument("--out-csv", default="step7h_dual_space_projector_prototype.csv")
    p.add_argument("--summary", default="step7h_dual_space_projector_prototype_summary.txt")
    return p.parse_args()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_core_tokens(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def load_labels(path: str, nobs: int) -> List[str]:
    data = np.load(path, allow_pickle=True)
    if "labels" in data.files:
        return [str(x) for x in data["labels"]]
    return [f"orb{i}" for i in range(nobs)]


def label_matches_core(label: str, tokens: List[str]) -> bool:
    label_l = label.lower()
    m = re.search(r"ecg-([spdf])(\d+)_m\d+", label_l)
    if not m:
        return False
    tag = f"{m.group(1)}{m.group(2)}"
    return tag in tokens


def fixed_core_indices(labels: List[str], tokens: List[str]) -> List[int]:
    return [i for i, lab in enumerate(labels) if label_matches_core(lab, tokens)]


def policy_rows_for_case(case: Dict[str, Any], labels: List[str], args) -> List[Dict[str, Any]]:
    nobs = case["nobs"]
    nri = case["nri"]
    obs = list(range(nobs))
    cabs = list(range(nobs, nri))
    occ = np.array(case["natural_occupations"], dtype=float)
    core_tokens = parse_core_tokens(args.core)
    fixed_core = fixed_core_indices(labels, core_tokens)
    if not fixed_core:
        fixed_core = obs[: min(6, nobs)]

    policies = [
        {
            "policy": "full_active_cabs_only",
            "active": obs,
            "passive": cabs,
            "description": "Step7d baseline: all ECG-NO OBS orbitals participate in the R12 projector.",
        },
        {
            "policy": f"dual_fixed_core_{'_'.join(core_tokens)}",
            "active": fixed_core,
            "passive": sorted([i for i in obs if i not in fixed_core] + cabs),
            "description": "Scheme A fixed projector core: full energy OBS, compact R12 projector core.",
        },
    ]
    for th in parse_float_list(args.occ_thresholds):
        active = [i for i in obs if i < args.nocc or occ[i] >= th]
        passive = sorted([i for i in obs if i not in active] + cabs)
        policies.append(
            {
                "policy": f"dual_occ_ge_{th:g}",
                "active": active,
                "passive": passive,
                "description": f"Scheme A occupation-threshold projector: NO occupation >= {th:g}.",
            }
        )

    rows = []
    for pol in policies:
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
        correction = sub["correction_total"]
        E_total = case["E_obs"] + correction
        residual = E_total - args.reference_energy if args.reference_energy is not None else None
        rows.append(
            {
                "case": case["label"],
                "path": case["path"],
                "nobs_energy": nobs,
                "nri": nri,
                "nqubits_energy": 2 * nobs,
                "policy": pol["policy"],
                "description": pol["description"],
                "active_indices": pol["active"],
                "active_labels": [labels[i] for i in pol["active"]],
                "active_size": len(pol["active"]),
                "passive_size": len(pol["passive"]),
                "active_trace_dm1": sub["active_trace_dm1"],
                "active_trace_dm2": sub["active_trace_dm2"],
                "missing_trace_dm1": 2.0 - sub["active_trace_dm1"],
                "E_obs_energy": case["E_obs"],
                "delta_E_r12": correction,
                "E_total": E_total,
                "reference_energy": args.reference_energy,
                "residual_to_reference": residual,
                "abs_residual_to_reference_mEh": None if residual is None else abs(residual) * 1000.0,
                "V": sub["V_total"],
                "B": sub["B_total"],
                "X": sub["X_total"],
                "Delta": sub["Delta_total"],
            }
        )
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "case",
        "nqubits_energy",
        "policy",
        "active_size",
        "passive_size",
        "active_trace_dm1",
        "missing_trace_dm1",
        "E_obs_energy",
        "delta_E_r12",
        "E_total",
        "residual_to_reference",
        "abs_residual_to_reference_mEh",
        "V",
        "B",
        "X",
        "Delta",
        "active_labels",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            out["active_labels"] = ";".join(row["active_labels"])
            writer.writerow(out)


def write_outputs(args, rows: List[Dict[str, Any]]) -> None:
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    write_csv(args.out_csv, rows)

    lines = []
    lines.append("=" * 116)
    lines.append("Step 7h | Scheme A dual-space ECG-NO R12 projector prototype")
    lines.append("=" * 116)
    lines.append(f"reference = {args.reference_energy:.14f} Eh")
    lines.append(f"fixed core = {args.core}")
    lines.append("")
    lines.append("case                         qubits policy                         Tr1        dE_R12(mEh)    E_total             resid_ref(mEh)")
    for row in rows:
        lines.append(
            f"{row['case']:<28s} {row['nqubits_energy']:>6d} "
            f"{row['policy']:<30s} {row['active_trace_dm1']:.8f} "
            f"{1000.0 * row['delta_E_r12']: .8f} {row['E_total']: .14f} "
            f"{1000.0 * row['residual_to_reference']: .8f}"
        )
    lines.append("")
    lines.append("[Interpretation Guardrail]")
    lines.append("Rows with active_trace_dm1 < 2 are dual-space projector prototypes, not final variational energies.")
    lines.append("They test whether low-occupation ECG-NO orbitals should be excluded from the R12 projector subtraction.")
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
    rows = []
    for path in args.inputs:
        case = load_case(path)
        labels = load_labels(path, case["nobs"])
        rows.extend(policy_rows_for_case(case, labels, args))
    write_outputs(args, rows)


if __name__ == "__main__":
    main()
