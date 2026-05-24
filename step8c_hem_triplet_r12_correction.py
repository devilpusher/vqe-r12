#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8c: First HEM triplet SF-[2]R12 candidate correction."""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict

import numpy as np

from r12_correction import compute_he_sf2r12_correction, validate_correction_result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="step8b_hem_triplet_sp_s01_p01_fitN7_step4b_like.npz")
    p.add_argument("--scale-f12", type=float, default=1.0)
    p.add_argument("--prefix", default="step8c_hem_triplet_sp_s01_p01_fitN7")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--skip-full-parent-target", action="store_true")
    return p.parse_args()


def same_spin_pair_fci_target(inp: str) -> Dict[str, float]:
    """Dense same-spin pair diagonalization for the parent/RI space."""
    data = np.load(inp, allow_pickle=True)
    h = np.array(data["h_ri"], dtype=float)
    eri = np.array(data["eri_ri"], dtype=float)
    n = h.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    H = np.zeros((len(pairs), len(pairs)), dtype=float)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            one = h[i, k] * (j == l) + h[j, l] * (i == k) - h[i, l] * (j == k) - h[j, k] * (i == l)
            H[a, b] = one + eri[i, k, j, l] - eri[i, l, j, k]
    H = 0.5 * (H + H.T)
    evals = np.linalg.eigvalsh(H)
    E_full = float(evals[0])
    E_obs = float(data["E_obs_fci"])
    return {
        "E_full_parent_triplet_pair_fci": E_full,
        "full_parent_gap": E_full - E_obs,
        "full_parent_gap_mEh": 1000.0 * (E_full - E_obs),
        "pair_dimension": len(pairs),
        "hamiltonian_symmetry_error": float(np.max(np.abs(H - H.T))),
    }


def write_outputs(args, result: Dict[str, Any], target: Dict[str, Any] | None) -> None:
    args.out_json = args.out_json or f"{args.prefix}_sf2r12_correction.json"
    args.out_csv = args.out_csv or f"{args.prefix}_sf2r12_correction.csv"
    args.summary = args.summary or f"{args.prefix}_sf2r12_correction_summary.txt"

    payload = {
        "pipeline": {
            "source": "Step8b HEM triplet selected-OBS bridge",
            "input": args.inp,
            "scale_f12": args.scale_f12,
            "note": "First triplet candidate reuses the audited spin-free CABS-only contraction; open-shell normalization remains an explicit audit item.",
        },
        "result": result,
        "full_parent_triplet_target": target,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    comp = result["components"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "state",
                "nobs",
                "nri",
                "ncabs",
                "E_obs_fci",
                "delta_E_r12",
                "E_total",
                "V",
                "B",
                "X",
                "Delta",
                "E_full_parent_triplet_pair_fci",
                "full_parent_gap_mEh",
                "residual_to_full_parent_mEh",
            ],
        )
        writer.writeheader()
        residual = None
        if target is not None:
            residual = 1000.0 * (float(result["E_total"]) - float(target["E_full_parent_triplet_pair_fci"]))
        writer.writerow(
            {
                "method": result["method"],
                "state": "He metastable triplet Ms=1",
                "nobs": result["nobs"],
                "nri": result["nri"],
                "ncabs": result["ncabs"],
                "E_obs_fci": result["E_obs_fci"],
                "delta_E_r12": result["delta_E_r12"],
                "E_total": result["E_total"],
                "V": comp["V"],
                "B": comp["B"],
                "X": comp["X"],
                "Delta": comp["Delta"],
                "E_full_parent_triplet_pair_fci": "" if target is None else target["E_full_parent_triplet_pair_fci"],
                "full_parent_gap_mEh": "" if target is None else target["full_parent_gap_mEh"],
                "residual_to_full_parent_mEh": "" if residual is None else residual,
            }
        )

    checks = result["diagnostics"]["energy_checks"]
    rdm_obs = result["diagnostics"]["rdm_diagnostics"]["obs"]
    lines = [
        "=" * 100,
        "Step 8c | HEM triplet selected-OBS SF-[2]R12 candidate correction",
        "=" * 100,
        f"input        = {args.inp}",
        f"method       = {result['method']} ({result['fock_model']})",
        f"passive      = {result['passive_space']}",
        f"nobs/ncabs/nri = {result['nobs']}/{result['ncabs']}/{result['nri']}",
        "",
        "[Energy]",
        f"E_HEM_triplet_OBS_FCI    = {result['E_obs_fci']:.14f} Eh",
        f"DeltaE_R12               = {result['delta_E_r12']:.12e} Eh",
        f"E_HEM_triplet_OBS_plus_R12 = {result['E_total']:.14f} Eh",
        "",
        "[Components]",
        f"V      = {comp['V']:.12e} Eh",
        f"B      = {comp['B']:.12e} Eh",
        f"X      = {comp['X']:.12e} Eh",
        f"Delta  = {comp['Delta']:.12e} Eh",
        "",
    ]
    if target is not None:
        residual = 1000.0 * (float(result["E_total"]) - float(target["E_full_parent_triplet_pair_fci"]))
        lines.extend(
            [
                "[Full Parent Triplet Pair-FCI Target]",
                f"pair dimension             = {target['pair_dimension']}",
                f"E_full_parent_triplet      = {target['E_full_parent_triplet_pair_fci']:.14f} Eh",
                f"OBS -> full parent gap     = {target['full_parent_gap_mEh']:.6f} mEh",
                f"R12 residual to target     = {residual:.6f} mEh",
                "",
            ]
        )
    lines.extend(
        [
        "[Diagnostics]",
        f"Delta OBS-RDM minus FCI = {checks['delta_obs_rdm_minus_fci']:.3e} Eh",
        f"Delta RI-RDM minus FCI  = {checks['delta_ri_rdm_minus_fci']:.3e} Eh",
        f"Tr(dm1_obs), Tr(dm2_obs) = {rdm_obs['trace_dm1']:.12f}, {rdm_obs['trace_dm2']:.12f}",
        "",
        "[Saved]",
        f"  {args.out_json}",
        f"  {args.out_csv}",
        f"  {args.summary}",
        ]
    )
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    result = compute_he_sf2r12_correction(args.inp, step4b_path=None, scale_f12=args.scale_f12)
    validate_correction_result(result)
    target = None if args.skip_full_parent_target else same_spin_pair_fci_target(args.inp)
    write_outputs(args, result, target)


if __name__ == "__main__":
    main()
