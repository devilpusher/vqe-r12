#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 10c: first Be small-space SF-[2]R12 correction audit."""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict

from r12_correction import compute_he_sf2r12_correction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="step10b_be_sp_s012_p0_fitN7_step4b_like.npz")
    p.add_argument("--scale-f12", type=float, default=1.0)
    p.add_argument("--prefix", default="step10c_be_sp_s012_p0_fitN7")
    p.add_argument("--nelec", type=int, default=4)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def validate_result(result: Dict[str, Any], nelec: int, tol: float = 1e-10) -> None:
    checks = result["diagnostics"]["energy_checks"]
    if abs(checks["delta_obs_rdm_minus_fci"]) > tol:
        raise ValueError("OBS RDM energy reconstruction check failed")
    if abs(checks["delta_ri_rdm_minus_fci"]) > tol:
        raise ValueError("RI-embedded RDM energy reconstruction check failed")
    expected_dm2_trace = float(nelec * (nelec - 1))
    for name, diag in result["diagnostics"]["rdm_diagnostics"].items():
        if abs(diag["trace_dm1"] - nelec) > 1e-8:
            raise ValueError(f"{name} dm1 trace check failed")
        if abs(diag["trace_dm2"] - expected_dm2_trace) > 1e-8:
            raise ValueError(f"{name} dm2 trace check failed")
    for name, diag in result["diagnostics"]["tensor_diagnostics"].items():
        if diag["has_nan"] or diag["has_inf"]:
            raise ValueError(f"{name} contains NaN or Inf")


def write_outputs(args, result: Dict[str, Any]) -> None:
    args.out_json = args.out_json or f"{args.prefix}_sf2r12_correction.json"
    args.out_csv = args.out_csv or f"{args.prefix}_sf2r12_correction.csv"
    args.summary = args.summary or f"{args.prefix}_sf2r12_correction_summary.txt"
    payload = {
        "pipeline": {
            "source": "Step10b Be selected-OBS bridge",
            "input": args.inp,
            "scale_f12": args.scale_f12,
            "nelec": args.nelec,
            "note": "First Be audit reuses the spin-free CABS-only SF-[2]R12 contraction; data are from early exnot13.f90 NOs.",
        },
        "result": result,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    comp = result["components"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "state", "nobs", "nri", "ncabs", "E_obs_fci", "delta_E_r12", "E_total", "V", "B", "X", "Delta"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": result["method"],
                "state": "Be ground-state singlet, selected ECG-NO",
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
    lines = [
        "=" * 100,
        "Step 10c | Be selected-OBS SF-[2]R12 correction audit",
        "=" * 100,
        f"input        = {args.inp}",
        f"method       = {result['method']} ({result['fock_model']})",
        f"passive      = {result['passive_space']}",
        f"nobs/ncabs/nri = {result['nobs']}/{result['ncabs']}/{result['nri']}",
        "",
        "[Energy]",
        f"E_Be_OBS_FCI             = {result['E_obs_fci']:.14f} Eh",
        f"DeltaE_R12               = {result['delta_E_r12']:.12e} Eh",
        f"E_Be_OBS_plus_R12        = {result['E_total']:.14f} Eh",
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
        "",
        "[Caution]",
        "This is a first Be correction audit using early exnot13.f90 NO data and an s,p parent CABS space.",
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
    validate_result(result, args.nelec)
    write_outputs(args, result)


if __name__ == "__main__":
    main()
