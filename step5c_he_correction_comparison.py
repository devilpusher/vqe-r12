#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5c: Relate prototype R12 corrections to standard external-space FCI/PT2 diagnostics.

Purpose
-------
This script summarizes the results from Step 4b and Step 5b into a compact
energy/error table.

It does NOT implement the final literature-level [2]R12 contraction.  It is a
diagnostic bridge that clarifies the relation among:

1. OBS-FCI reference energy;
2. full parent-basis FCI target energy;
3. ordinary external Q-pair correction;
4. diagonal EN-like PT2 correction;
5. prototype fixed-amplitude and one-vector optimized R12/F12 directions.

Inputs
------
Default Step-4b file:
    he_ccpvdz_step4b_obs_fci_rdm.npz

Default Step-5b file:
    he_ccpvdz_step5b_r12_prototype_correction.npz

Outputs
-------
    he_ccpvdz_step5c_correction_comparison.csv
    he_ccpvdz_step5c_correction_comparison_summary.txt
    he_ccpvdz_step5c_correction_comparison.json

Interpretation
--------------
Let

    E_obs  = OBS-FCI energy
    E_full = full parent-basis FCI energy
    Delta_full = E_full - E_obs

Then each correction method has

    Delta_method = E_method - E_obs
    residual_to_full = E_method - E_full
    recovery_ratio = Delta_method / Delta_full

For a good parent-basis external-space correction, Delta_method should be close
to Delta_full and residual_to_full should be small.

Important
---------
The present R12-like quantities are prototype diagnostics with a one-Gaussian
testing factor unless the input Step-4b file used a proper fitted Slater
Gaussian expansion.  They should not be reported as final [2]R12 numerical
results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step4b", default="he_ccpvdz_step4b_obs_fci_rdm.npz", help="Step-4b npz input.")
    p.add_argument("--step5b", default="he_ccpvdz_step5b_r12_prototype_correction.npz", help="Step-5b npz input.")
    p.add_argument("--csv", default="he_ccpvdz_step5c_correction_comparison.csv", help="Output CSV.")
    p.add_argument("--json", default="he_ccpvdz_step5c_correction_comparison.json", help="Output JSON.")
    p.add_argument("--summary", default="he_ccpvdz_step5c_correction_comparison_summary.txt", help="Output text summary.")
    return p.parse_args()


def load_meta(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    if "metadata_json" not in data:
        raise RuntimeError(f"{npz_path} does not contain metadata_json")
    return json.loads(str(data["metadata_json"]))


def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def add_row(rows: List[Dict[str, Any]], method: str, energy: Optional[float], E_obs: float, E_full: float,
            note: str, category: str = "diagnostic"):
    Delta_full = E_full - E_obs
    if energy is None:
        delta = None
        residual = None
        ratio = None
        abs_residual_mEh = None
    else:
        delta = energy - E_obs
        residual = energy - E_full
        ratio = delta / Delta_full if abs(Delta_full) > 0.0 else None
        abs_residual_mEh = abs(residual) * 1000.0

    rows.append({
        "category": category,
        "method": method,
        "energy_Eh": energy,
        "delta_from_OBS_Eh": delta,
        "residual_to_full_parent_FCI_Eh": residual,
        "abs_residual_to_full_mEh": abs_residual_mEh,
        "recovery_ratio_vs_full_truncation": ratio,
        "note": note,
    })


def fmt_float(x: Any, width: int = 18, prec: int = 12) -> str:
    if x is None:
        return "None".rjust(width)
    try:
        return f"{float(x): {width}.{prec}f}"
    except Exception:
        return str(x).rjust(width)


def fmt_sci(x: Any, width: int = 14, prec: int = 6) -> str:
    if x is None:
        return "None".rjust(width)
    try:
        return f"{float(x): {width}.{prec}e}"
    except Exception:
        return str(x).rjust(width)


def make_markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| category | method | E / Eh | Δ from OBS / Eh | residual to full / Eh | "
        "|residual| / mEh | recovery ratio | note |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['category']} | {r['method']} | "
            f"{fmt_float(r['energy_Eh'], width=0, prec=12)} | "
            f"{fmt_sci(r['delta_from_OBS_Eh'], width=0, prec=6)} | "
            f"{fmt_sci(r['residual_to_full_parent_FCI_Eh'], width=0, prec=6)} | "
            f"{fmt_float(r['abs_residual_to_full_mEh'], width=0, prec=6)} | "
            f"{fmt_float(r['recovery_ratio_vs_full_truncation'], width=0, prec=6)} | "
            f"{r['note']} |\n"
        )
    return "".join(lines)


def main():
    args = parse_args()

    meta4 = load_meta(args.step4b)
    meta5 = load_meta(args.step5b)

    # Core energies from Step 4b.
    E_obs = as_float(meta4.get("E_obs_fci"))
    E_full = as_float(meta4.get("E_full_parent_fci"))
    E_scf = as_float(meta4.get("E_scf_parent"))
    E_obs_rdm = as_float(meta4.get("E_obs_rdm"))
    E_ri_rdm = as_float(meta4.get("E_ri_embedded_rdm"))

    if E_obs is None or E_full is None:
        raise RuntimeError("Step-4b metadata must contain E_obs_fci and E_full_parent_fci")

    Delta_full = E_full - E_obs

    # Energies from Step 5b.
    energies5 = meta5.get("prototype_corrected_total_energies", {})
    one_vector = meta5.get("one_vector", {})
    full_q = meta5.get("full_Q_pair_diagnostic", {})
    alignment = meta5.get("alignment", {})
    pair_norms = meta5.get("pair_block_norms", {})

    rows: List[Dict[str, Any]] = []
    add_row(rows, "RHF parent", E_scf, E_obs, E_full, "Parent-basis Hartree--Fock reference.", "reference")
    add_row(rows, "OBS-FCI", E_obs, E_obs, E_full, "Truncated OBS variational reference.", "reference")
    add_row(rows, "OBS-RDM check", E_obs_rdm, E_obs, E_full, "RDM reconstruction in OBS.", "check")
    add_row(rows, "RI-embedded RDM check", E_ri_rdm, E_obs, E_full, "OBS RDM embedded in RI basis.", "check")
    add_row(rows, "Full parent FCI", E_full, E_obs, E_full, "Target within the single parent AO basis.", "target")

    add_row(
        rows,
        "Raw F12 fixed amplitude",
        as_float(energies5.get("E_raw_fixed")),
        E_obs,
        E_full,
        "Fixed A_raw_Q amplitude; not variationally optimized and not physically meaningful.",
        "prototype",
    )
    add_row(
        rows,
        "Raw F12 1D optimized",
        as_float(energies5.get("E_raw_1D_opt")),
        E_obs,
        E_full,
        "Scalar-optimized one-vector correction along A_raw_Q.",
        "prototype",
    )
    add_row(
        rows,
        "SP-F12 fixed amplitude",
        as_float(energies5.get("E_sp_fixed")),
        E_obs,
        E_full,
        "Fixed A_sp_Q amplitude; diagnostic only.",
        "prototype",
    )
    add_row(
        rows,
        "SP-F12 1D optimized",
        as_float(energies5.get("E_sp_1D_opt")),
        E_obs,
        E_full,
        "Scalar-optimized one-vector correction along A_sp_Q.",
        "prototype",
    )
    add_row(
        rows,
        "Full Q-pair solve",
        as_float(energies5.get("E_full_Q_pair")),
        E_obs,
        E_full,
        "Ordinary external Q-pair solve in RI pair space; diagnostic target for external correction.",
        "external-space",
    )
    add_row(
        rows,
        "Diagonal EN-like PT2",
        as_float(energies5.get("E_diag_EN_like")),
        E_obs,
        E_full,
        "Diagonal approximation to Q-pair solve; analogous to EN-PT2 diagnostic.",
        "external-space",
    )

    # Sort rows in intended order already inserted.

    # Write CSV.
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category",
            "method",
            "energy_Eh",
            "delta_from_OBS_Eh",
            "residual_to_full_parent_FCI_Eh",
            "abs_residual_to_full_mEh",
            "recovery_ratio_vs_full_truncation",
            "note",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Add higher-level diagnostics.
    comparison = {
        "inputs": {
            "step4b": args.step4b,
            "step5b": args.step5b,
        },
        "core_energies": {
            "E_obs_fci": E_obs,
            "E_full_parent_fci": E_full,
            "Delta_full_parent_minus_OBS": Delta_full,
            "abs_Delta_full_mEh": abs(Delta_full) * 1000.0,
            "E_scf_parent": E_scf,
        },
        "correction_rows": rows,
        "one_vector_details": {
            "A_raw_Q": one_vector.get("A_raw_Q"),
            "A_sp_Q": one_vector.get("A_sp_Q"),
        },
        "full_Q_pair_details": full_q,
        "alignment": alignment,
        "pair_block_norms": pair_norms,
        "interpretation": {
            "main_point": (
                "The prototype one-vector optimized F12 directions and the full Q-pair solve "
                "recover nearly the same external-space correction in this minimal He test. "
                "This validates the sign, Q projection, and pair-space convention, but it is not "
                "yet the final [2]R12 contraction."
            ),
            "limitation": (
                "The current correlation factor is a testing Gaussian surrogate unless the input "
                "Step-4b file was generated with a proper fitted Slater expansion."
            ),
        },
    }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Build human-readable summary.
    table = make_markdown_table(rows)

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("Step 5c | Prototype correction vs external-space FCI/PT2 relation")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("[Core energies]")
    summary_lines.append(f"E_OBS-FCI              = {E_obs: .14f} Eh")
    summary_lines.append(f"E_full_parent_FCI      = {E_full: .14f} Eh")
    summary_lines.append(f"Delta_full = E_full - E_OBS = {Delta_full: .14e} Eh")
    summary_lines.append(f"|Delta_full|           = {abs(Delta_full) * 1000.0:.8f} mEh")
    summary_lines.append("")
    summary_lines.append("[Correction table]")
    summary_lines.append(table)
    summary_lines.append("")
    summary_lines.append("[Key diagnostics from Step 5b]")
    try:
        raw = one_vector.get("A_raw_Q", {})
        sp = one_vector.get("A_sp_Q", {})
        summary_lines.append(f"DeltaE raw 1D optimized = {raw.get('deltaE_1D_opt')}")
        summary_lines.append(f"DeltaE SP  1D optimized = {sp.get('deltaE_1D_opt')}")
        summary_lines.append(f"DeltaE full Q-pair      = {full_q.get('deltaE_full_Q_pair_diagnostic')}")
        summary_lines.append(f"DeltaE diagonal EN-like = {full_q.get('deltaE_diag_EN_like')}")
        summary_lines.append(f"min eig(H_QQ-E0)        = {full_q.get('min_eig_HQQ_minus_E')}")
        summary_lines.append(f"n eig <= threshold      = {full_q.get('n_eig_le_thresh')}")
        summary_lines.append(f"cos(A_raw_Q, x_Q)       = {alignment.get('A_raw_Q_vs_x_Q', {}).get('cosine')}")
        summary_lines.append(f"cos(A_sp_Q,  x_Q)       = {alignment.get('A_sp_Q_vs_x_Q', {}).get('cosine')}")
    except Exception as exc:
        summary_lines.append(f"Could not summarize Step-5b diagnostics: {repr(exc)}")

    summary_lines.append("")
    summary_lines.append("[Interpretation]")
    summary_lines.append(
        "In this minimal He parent-basis test, the optimized one-vector prototype "
        "corrections and the full Q-pair solve recover the OBS-to-full-parent "
        "FCI gap almost completely. This shows that the present Q-space, pair "
        "Hamiltonian, sign convention, and F12-generated direction are internally "
        "consistent."
    )
    summary_lines.append(
        "The fixed-amplitude rows should not be interpreted as physical energy "
        "corrections. They are included only to show why Hylleraas optimization "
        "or an equivalent amplitude treatment is required."
    )
    summary_lines.append(
        "The diagonal EN-like row is a conventional external-space PT2 analogue. "
        "Its comparison with the full Q-pair solve gives a useful scale for the "
        "error introduced by diagonal denominators."
    )
    summary_lines.append(
        "This step still does not constitute the final article-level [2]R12 "
        "correction because the current correlation factor is a one-Gaussian "
        "testing surrogate and the full approximation-C/SP tensor contraction "
        "has not yet been implemented."
    )

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        f.write("\n")

    print("=" * 80)
    print("Step 5c | Prototype correction comparison")
    print("=" * 80)
    print(f"E_OBS-FCI         = {E_obs: .14f} Eh")
    print(f"E_full_parent_FCI = {E_full: .14f} Eh")
    print(f"Delta_full        = {Delta_full: .8e} Eh ({abs(Delta_full)*1000.0:.6f} mEh)")
    print("")
    print(table)
    print("[Saved]")
    print(" ", args.csv)
    print(" ", args.json)
    print(" ", args.summary)


if __name__ == "__main__":
    main()
