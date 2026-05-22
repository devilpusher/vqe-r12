#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6m: Formal He parent-basis SF-[2]R12 correction pipeline.

This is the clean workflow entry point for the stabilized He/Psi4 OBS path:

    fitted Slater corr -> Step 4b -> Step 5a -> paper/Tequila SF-[2]R12 correction

The older Step-6f candidate ledger and Step-6g/6k audits remain useful for
formula debugging, but this script emits only the selected correction path.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import step6a_fit_slater_corr as fit6a
from r12_correction import compute_he_sf2r12_correction, validate_correction_result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parent-basis", default="cc-pvdz")
    p.add_argument("--nobs", type=int, default=2)
    p.add_argument("--fitN", type=int, default=7, help="Number of Gaussian terms for local fitted Slater corr.")
    p.add_argument("--corr", default=None, help="Explicit Psi4 corr string alpha,c;alpha,c. Overrides --fitN fitting.")
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--alpha-min", type=float, default=0.08)
    p.add_argument("--alpha-max", type=float, default=60.0)
    p.add_argument("--rmin", type=float, default=0.0)
    p.add_argument("--rmax", type=float, default=8.0)
    p.add_argument("--ngrid", type=int, default=2000)
    p.add_argument("--grid", choices=["linear", "quadratic"], default="quadratic")
    p.add_argument("--weight", choices=["flat", "r2", "short", "relative"], default="short")
    p.add_argument("--ridge", type=float, default=0.0)
    p.add_argument("--nonpositive-coeff", action="store_true")
    p.add_argument("--scale-f12", type=float, default=1.0)
    p.add_argument("--prefix", default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--force", action="store_true", help="Re-run Step 4b/5a even if outputs exist.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def safe_label(s: str) -> str:
    return s.lower().replace("*", "s").replace("+", "p").replace("-", "").replace("_", "")


def default_prefix(parent_basis: str, nobs: int, fitN: int, corr: str | None) -> str:
    corr_label = f"fitN{fitN}" if corr is None else "customcorr"
    return f"he_{safe_label(parent_basis)}_nobs{nobs}_{corr_label}"


def fit_corr(args) -> Dict[str, Any] | None:
    if args.corr is not None:
        return None
    ns = argparse.Namespace(
        gamma=args.gamma,
        nterms=args.fitN,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        rmin=args.rmin,
        rmax=args.rmax,
        ngrid=args.ngrid,
        grid=args.grid,
        weight=args.weight,
        ridge=args.ridge,
        nonpositive_coeff=args.nonpositive_coeff,
    )
    return fit6a.fit_gaussian_expansion(ns)


def output_names(prefix: str) -> Dict[str, str]:
    return {
        "step4b": f"{prefix}_step4b_obs_fci_rdm.npz",
        "step4b_summary": f"{prefix}_step4b_obs_fci_rdm_summary.txt",
        "step5a": f"{prefix}_step5a_r12_intermediates.npz",
        "step5a_summary": f"{prefix}_step5a_r12_intermediates_summary.txt",
        "json": f"{prefix}_step6m_sf2r12_correction.json",
        "csv": f"{prefix}_step6m_sf2r12_correction.csv",
        "summary": f"{prefix}_step6m_sf2r12_correction_summary.txt",
    }


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_inputs(args, names: Dict[str, str], corr: str) -> None:
    if args.force or not Path(names["step4b"]).exists():
        run_command(
            [
                args.python,
                "step4b_he_parent_obs_fci_rdm_check.py",
                "--parent-basis",
                args.parent_basis,
                "--nobs",
                str(args.nobs),
                "--corr",
                corr,
                "--out",
                names["step4b"],
                "--summary",
                names["step4b_summary"],
            ],
            args.dry_run,
        )
    if args.force or not Path(names["step5a"]).exists():
        run_command(
            [
                args.python,
                "step5a_he_r12_intermediate_check.py",
                "--inp",
                names["step4b"],
                "--out",
                names["step5a"],
                "--summary",
                names["step5a_summary"],
            ],
            args.dry_run,
        )


def fmt(x: Any, prec: int = 12) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{prec}e}"
    except Exception:
        return str(x)


def write_outputs(args, names: Dict[str, str], result: Dict[str, Any], fit: Dict[str, Any] | None, corr: str) -> None:
    payload = {
        "pipeline": {
            "parent_basis": args.parent_basis,
            "nobs": args.nobs,
            "fitN": None if args.corr else args.fitN,
            "corr_source": "explicit --corr" if args.corr else "local least-squares fitted Slater Gaussian expansion",
            "corr_string": corr,
            "step4b": names["step4b"],
            "step5a": names["step5a"],
        },
        "fit": fit,
        "result": result,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    components = result["components"]
    fieldnames = [
        "method",
        "parent_basis",
        "nobs",
        "nri",
        "ncabs",
        "fitN",
        "E_obs_fci",
        "E_full_parent_fci",
        "delta_E_r12",
        "E_total",
        "residual_to_full_parent_FCI",
        "abs_residual_to_full_mEh",
        "recovery_ratio",
        "V",
        "B",
        "X",
        "Delta",
    ]
    row = {
        "method": result["method"],
        "parent_basis": args.parent_basis,
        "nobs": result["nobs"],
        "nri": result["nri"],
        "ncabs": result["ncabs"],
        "fitN": None if args.corr else args.fitN,
        "E_obs_fci": result["E_obs_fci"],
        "E_full_parent_fci": result["E_full_parent_fci"],
        "delta_E_r12": result["delta_E_r12"],
        "E_total": result["E_total"],
        "residual_to_full_parent_FCI": result["residual_to_full_parent_FCI"],
        "abs_residual_to_full_mEh": result["abs_residual_to_full_mEh"],
        "recovery_ratio": result["recovery_ratio"],
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
    lines = []
    lines.append("=" * 100)
    lines.append("Step 6m | Formal He parent-basis SF-[2]R12 correction")
    lines.append("=" * 100)
    lines.append(f"parent basis / nobs / nri = {args.parent_basis} / {result['nobs']} / {result['nri']}")
    lines.append(f"method       = {result['method']} ({result['fock_model']})")
    lines.append(f"passive      = {result['passive_space']}")
    lines.append(f"corr source  = {'explicit --corr' if args.corr else 'local fitted Slater Gaussian expansion'}")
    if fit is not None:
        m = fit["metrics"]
        lines.append(f"fit          = N={args.fitN}, RMS={m['rms_abs']:.8e}, relRMS={m['rel_rms']:.8e}, f0err={m['f0_error']:.8e}")
    lines.append("")
    lines.append("[Energy]")
    lines.append(f"E_OBS_FCI              = {result['E_obs_fci']:.14f} Eh")
    if result["E_full_parent_fci"] is not None:
        lines.append(f"E_full_parent_FCI      = {result['E_full_parent_fci']:.14f} Eh")
        lines.append(f"OBS-to-full gap        = {result['full_parent_gap']:.12e} Eh ({abs(result['full_parent_gap']) * 1000.0:.6f} mEh)")
    lines.append(f"DeltaE_R12             = {result['delta_E_r12']:.12e} Eh")
    lines.append(f"E_OBS_plus_R12         = {result['E_total']:.14f} Eh")
    lines.append(f"residual_to_full       = {fmt(result['residual_to_full_parent_FCI'])} Eh ({fmt(result['abs_residual_to_full_mEh'], 8)} mEh)")
    lines.append(f"recovery_ratio         = {fmt(result['recovery_ratio'], 8)}")
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
    lines.append(f"Tr(dm1_obs), Tr(dm2_obs) = {result['diagnostics']['rdm_diagnostics']['obs']['trace_dm1']:.12f}, {result['diagnostics']['rdm_diagnostics']['obs']['trace_dm2']:.12f}")
    lines.append(f"f12 phys bra-ket error   = {result['diagnostics']['tensor_diagnostics']['f12_phys']['bra_ket_error']:.3e}")
    lines.append(f"max|F_tequila-F_step5a|  = {result['diagnostics']['fock_diagnostics']['maxabs_fock_tequila_minus_step5a']:.3e}")
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
    prefix = args.prefix or default_prefix(args.parent_basis, args.nobs, args.fitN, args.corr)
    names = output_names(prefix)
    args.out_json = args.out_json or names["json"]
    args.out_csv = args.out_csv or names["csv"]
    args.summary = args.summary or names["summary"]

    fit = fit_corr(args)
    corr = args.corr if args.corr is not None else fit["corr_string"]
    ensure_inputs(args, names, corr)
    if args.dry_run:
        return

    result = compute_he_sf2r12_correction(names["step5a"], names["step4b"], scale_f12=args.scale_f12)
    validate_correction_result(result)
    write_outputs(args, names, result, fit, corr)


if __name__ == "__main__":
    main()
