#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7d: Apply the audited SF-[2]R12 correction to ECG-NO Step7c input.

Input is the Step7c Step-4b-like `.npz`, which already contains the selected
ECG-NO OBS RDMs, CABS+/RI tensors, and energy reconstruction checks.  This
script applies the same paper/Tequila spin-free [2]R12 candidate used by Step6m.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict

from r12_correction import compute_he_sf2r12_correction, validate_correction_result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="step7c_ecg_no_sp_s012_p0_fitN7_step4b_like.npz")
    p.add_argument("--scale-f12", type=float, default=1.0)
    p.add_argument("--prefix", default="step7d_ecg_no_sp_s012_p0_fitN7")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def fmt(x: Any, prec: int = 12) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{prec}e}"
    except Exception:
        return str(x)


def write_outputs(args, result: Dict[str, Any]) -> None:
    args.out_json = args.out_json or f"{args.prefix}_sf2r12_correction.json"
    args.out_csv = args.out_csv or f"{args.prefix}_sf2r12_correction.csv"
    args.summary = args.summary or f"{args.prefix}_sf2r12_correction_summary.txt"

    payload = {
        "pipeline": {
            "source": "Step7c ECG-NO selected-OBS bridge",
            "input": args.inp,
            "scale_f12": args.scale_f12,
        },
        "result": result,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    components = result["components"]
    fieldnames = [
        "method",
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
    ]
    row = {
        "method": result["method"],
        "nobs": result["nobs"],
        "nri": result["nri"],
        "ncabs": result["ncabs"],
        "E_obs_fci": result["E_obs_fci"],
        "delta_E_r12": result["delta_E_r12"],
        "E_total": result["E_total"],
        "V": components["V"],
        "B": components["B"],
        "X": components["X"],
        "Delta": components["Delta"],
    }
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    checks = result["diagnostics"]["energy_checks"]
    rdm_obs = result["diagnostics"]["rdm_diagnostics"]["obs"]
    tensor = result["diagnostics"]["tensor_diagnostics"]
    fock = result["diagnostics"]["fock_diagnostics"]
    lines = []
    lines.append("=" * 100)
    lines.append("Step 7d | ECG-NO selected-OBS SF-[2]R12 correction")
    lines.append("=" * 100)
    lines.append(f"input        = {args.inp}")
    lines.append(f"method       = {result['method']} ({result['fock_model']})")
    lines.append(f"passive      = {result['passive_space']}")
    lines.append(f"nobs/ncabs/nri = {result['nobs']}/{result['ncabs']}/{result['nri']}")
    lines.append("")
    lines.append("[Energy]")
    lines.append(f"E_ECG_NO_OBS_FCI        = {result['E_obs_fci']:.14f} Eh")
    lines.append(f"DeltaE_R12              = {result['delta_E_r12']:.12e} Eh")
    lines.append(f"E_ECG_NO_OBS_plus_R12   = {result['E_total']:.14f} Eh")
    lines.append("full-parent target      = not available for Step7c ECG-NO bridge")
    lines.append("")
    lines.append("[Components]")
    lines.append(f"V      = {components['V']:.12e} Eh")
    lines.append(f"B      = {components['B']:.12e} Eh")
    lines.append(f"X      = {components['X']:.12e} Eh")
    lines.append(f"Delta  = {components['Delta']:.12e} Eh")
    lines.append("")
    lines.append("[Diagnostics]")
    lines.append(f"Delta OBS-RDM minus FCI = {checks['delta_obs_rdm_minus_fci']:.3e} Eh")
    lines.append(f"Delta RI-RDM minus FCI  = {checks['delta_ri_rdm_minus_fci']:.3e} Eh")
    lines.append(f"Tr(dm1_obs), Tr(dm2_obs) = {rdm_obs['trace_dm1']:.12f}, {rdm_obs['trace_dm2']:.12f}")
    lines.append(f"eri/f12 bra-ket error    = {tensor['eri_phys']['bra_ket_error']:.3e}, {tensor['f12_phys']['bra_ket_error']:.3e}")
    lines.append(f"Step5a Fock available    = {fock['fock_step5a_available']}")
    lines.append(f"max|F_tequila-F_step5a|  = {fmt(fock['maxabs_fock_tequila_minus_step5a'], 3)}")
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
    result = compute_he_sf2r12_correction(args.inp, step4b_path=None, scale_f12=args.scale_f12)
    validate_correction_result(result)
    write_outputs(args, result)


if __name__ == "__main__":
    main()
