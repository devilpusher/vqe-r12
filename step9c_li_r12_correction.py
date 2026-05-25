#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 9c: first Li small-space SF-[2]R12 correction audit.

This reuses the audited CABS-only spin-free contraction machinery, but validates
the Li three-electron RDM convention explicitly: Tr(dm1)=3 and Tr(dm2)=6.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict

from r12_correction import compute_he_sf2r12_correction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="step9b_li_sp_s01_p0_fitN7_step4b_like.npz")
    p.add_argument("--scale-f12", type=float, default=1.0)
    p.add_argument("--prefix", default="step9c_li_sp_s01_p0_fitN7")
    p.add_argument("--nelec", type=int, default=3)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def validate_li_correction_result(result: Dict[str, Any], nelec: int, tol: float = 1e-10) -> None:
    checks = result["diagnostics"]["energy_checks"]
    if abs(checks["delta_obs_rdm_minus_fci"]) > tol:
        raise ValueError("OBS RDM energy reconstruction check failed")
    if abs(checks["delta_ri_rdm_minus_fci"]) > tol:
        raise ValueError("RI-embedded RDM energy reconstruction check failed")
    for name, diag in result["diagnostics"]["tensor_diagnostics"].items():
        if diag["has_nan"] or diag["has_inf"]:
            raise ValueError(f"{name} contains NaN or Inf")

    expected_dm2_trace = float(nelec * (nelec - 1))
    for name, diag in result["diagnostics"]["rdm_diagnostics"].items():
        if abs(diag["trace_dm1"] - nelec) > 1e-8:
            raise ValueError(f"{name} dm1 trace check failed")
        if abs(diag["trace_dm2"] - expected_dm2_trace) > 1e-8:
            raise ValueError(f"{name} dm2 trace check failed")


def write_outputs(args, result: Dict[str, Any]) -> None:
    args.out_json = args.out_json or f"{args.prefix}_sf2r12_correction.json"
    args.out_csv = args.out_csv or f"{args.prefix}_sf2r12_correction.csv"
    args.summary = args.summary or f"{args.prefix}_sf2r12_correction_summary.txt"

    payload = {
        "pipeline": {
            "source": "Step9b Li selected-OBS bridge",
            "input": args.inp,
            "scale_f12": args.scale_f12,
            "nelec": args.nelec,
            "note": (
                "First Li audit reuses the audited spin-free CABS-only SF-[2]R12 "
                "contraction.  It is a small-space candidate, not yet a calibrated "
                "Li/Be physical law."
            ),
        },
        "result": result,
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": result["method"],
                "state": "Li ground-state doublet, selected ECG-NO s01+p0",
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
            }
        )

    checks = result["diagnostics"]["energy_checks"]
    rdm_obs = result["diagnostics"]["rdm_diagnostics"]["obs"]
    tensor = result["diagnostics"]["tensor_diagnostics"]
    fock = result["diagnostics"]["fock_diagnostics"]
    lines = [
        "=" * 100,
        "Step 9c | Li selected-OBS SF-[2]R12 correction audit",
        "=" * 100,
        f"input        = {args.inp}",
        f"method       = {result['method']} ({result['fock_model']})",
        f"passive      = {result['passive_space']}",
        f"nobs/ncabs/nri = {result['nobs']}/{result['ncabs']}/{result['nri']}",
        "",
        "[Energy]",
        f"E_Li_OBS_FCI             = {result['E_obs_fci']:.14f} Eh",
        f"DeltaE_R12               = {result['delta_E_r12']:.12e} Eh",
        f"E_Li_OBS_plus_R12        = {result['E_total']:.14f} Eh",
        "",
        "[Components]",
        f"V      = {comp['V']:.12e} Eh",
        f"B      = {comp['B']:.12e} Eh",
        f"X      = {comp['X']:.12e} Eh",
        f"Delta  = {comp['Delta']:.12e} Eh",
        "",
        "[Diagnostics]",
        f"Delta OBS-RDM minus FCI = {checks['delta_obs_rdm_minus_fci']:.3e} Eh",
        f"Delta RI-RDM minus FCI  = {checks['delta_ri_rdm_minus_fci']:.3e} Eh",
        f"Tr(dm1_obs), Tr(dm2_obs) = {rdm_obs['trace_dm1']:.12f}, {rdm_obs['trace_dm2']:.12f}",
        f"eri/f12 bra-ket error    = {tensor['eri_phys']['bra_ket_error']:.3e}, {tensor['f12_phys']['bra_ket_error']:.3e}",
        f"Step5a Fock available    = {fock['fock_step5a_available']}",
        "",
        "[Caution]",
        "This is a Li small-space correction audit using the He/ECG-NO SF-[2]R12 contraction form.",
        "Do not promote it as a calibrated Li physical law before selected-space and Be consistency checks.",
        "",
        "[Saved]",
        f"  {args.out_json}",
        f"  {args.out_csv}",
        f"  {args.summary}",
    ]
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    result = compute_he_sf2r12_correction(args.inp, step4b_path=None, scale_f12=args.scale_f12)
    validate_li_correction_result(result, nelec=args.nelec)
    write_outputs(args, result)


if __name__ == "__main__":
    main()
