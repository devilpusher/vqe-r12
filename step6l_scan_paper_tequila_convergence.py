#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6l: Scan paper_tequila_sf2r12 convergence for fitN and basis/OBS size."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

import step6a_fit_slater_corr as fit6a


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nterms", default="3,5,7,9", help="Comma-separated Gaussian fit sizes.")
    p.add_argument(
        "--cases",
        default="cc-pvdz:2,cc-pvdz:3,cc-pvtz:2",
        help="Comma-separated parent-basis:nobs cases.",
    )
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
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--fock-model",
        default="tequila_fock_from_paper_formula",
        help="paper_tequila_sf2r12 sub-row to collect; use 'all' to keep comparison rows.",
    )
    p.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-json", default="step6l_paper_tequila_convergence_scan.json")
    p.add_argument("--out-csv", default="step6l_paper_tequila_convergence_scan.csv")
    p.add_argument("--summary", default="step6l_paper_tequila_convergence_scan_summary.txt")
    return p.parse_args()


def safe_label(s: str) -> str:
    return s.lower().replace("*", "s").replace("+", "p").replace("-", "").replace("_", "")


def parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_cases(s: str) -> list[tuple[str, int]]:
    out = []
    for part in s.split(","):
        if not part.strip():
            continue
        basis, nobs = part.split(":")
        out.append((basis.strip(), int(nobs.strip())))
    return out


def run_command(cmd: list[str], dry_run: bool = False):
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def fit_corr(args, nterms: int) -> Dict[str, Any]:
    ns = argparse.Namespace(
        gamma=args.gamma,
        nterms=nterms,
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


def output_names(parent_basis: str, nobs: int, nterms: int) -> Dict[str, str]:
    prefix = f"he_{safe_label(parent_basis)}_nobs{nobs}_fitN{nterms}"
    return {
        "prefix": prefix,
        "step4b": f"{prefix}_step4b_obs_fci_rdm.npz",
        "step4b_summary": f"{prefix}_step4b_obs_fci_rdm_summary.txt",
        "step5a": f"{prefix}_step5a_r12_intermediates.npz",
        "step5a_summary": f"{prefix}_step5a_r12_intermediates_summary.txt",
        "step6f_json": f"{prefix}_step6f_he_r12_candidate_energy.json",
        "step6f_csv": f"{prefix}_step6f_he_r12_candidate_energy.csv",
        "step6f_summary": f"{prefix}_step6f_he_r12_candidate_energy_summary.txt",
    }


def ensure_pipeline(args, parent_basis: str, nobs: int, nterms: int, corr: str) -> Dict[str, str]:
    names = output_names(parent_basis, nobs, nterms)
    need_step4b = args.force or not Path(names["step4b"]).exists()
    need_step5a = args.force or not Path(names["step5a"]).exists()
    need_step6f = args.force or not Path(names["step6f_json"]).exists()

    if need_step4b:
        run_command(
            [
                args.python,
                "step4b_he_parent_obs_fci_rdm_check.py",
                "--parent-basis",
                parent_basis,
                "--nobs",
                str(nobs),
                "--corr",
                corr,
                "--out",
                names["step4b"],
                "--summary",
                names["step4b_summary"],
            ],
            args.dry_run,
        )
    if need_step5a:
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
    if need_step6f:
        run_command(
            [
                args.python,
                "step6f_he_r12_candidate_energy.py",
                "--inp",
                names["step5a"],
                "--step4b",
                names["step4b"],
                "--out",
                names["step6f_json"],
                "--csv",
                names["step6f_csv"],
                "--summary",
                names["step6f_summary"],
            ],
            args.dry_run,
        )
    return names


def load_metadata_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if "metadata_json" not in data:
        return {}
    return json.loads(str(data["metadata_json"]))


def load_paper_rows(
    names: Dict[str, str],
    fit: Dict[str, Any],
    parent_basis: str,
    nobs: int,
    nterms: int,
    fock_model: str,
) -> list[Dict[str, Any]]:
    if not Path(names["step6f_json"]).exists():
        return []
    with open(names["step6f_json"], "r", encoding="utf-8") as f:
        data = json.load(f)
    step4b_meta = load_metadata_npz(names["step4b"])
    rows = []
    for row in data.get("paper_tequila_sf2r12", []):
        if fock_model != "all" and row.get("fock_model") != fock_model:
            continue
        out = {
            "parent_basis": parent_basis,
            "nobs": nobs,
            "nterms": nterms,
            "nri": data.get("nri"),
            "ncabs": None if data.get("nri") is None else int(data["nri"]) - int(nobs),
            "fock_model": row.get("fock_model"),
            "delta_E": row.get("delta_E"),
            "residual_to_full_parent_FCI": row.get("residual_to_full_parent_FCI"),
            "abs_residual_to_full_mEh": row.get("abs_residual_to_full_mEh"),
            "recovery_ratio": row.get("recovery_ratio"),
            "E_obs_fci": data.get("E_obs_fci"),
            "E_full_parent_fci": data.get("E_full_parent_fci"),
            "full_parent_gap": data.get("full_parent_gap"),
            "V_component": row.get("V_component"),
            "B_component": row.get("B_component"),
            "X_component": row.get("X_component"),
            "Delta_component": row.get("Delta_component"),
            "fit_RMS_error": fit["metrics"]["rms_abs"],
            "fit_max_error": fit["metrics"]["max_abs"],
            "fit_relative_RMS_error": fit["metrics"]["rel_rms"],
            "fit_short_range_RMS_error": fit["metrics"]["region_metrics"]["short_r_le_1"]["rms_abs"],
            "fit_short_range_max_error": fit["metrics"]["region_metrics"]["short_r_le_1"]["max_abs"],
            "fit_f0_error": fit["metrics"]["f0_error"],
            "step4b_ri_orth_error": step4b_meta.get("cabs_info", {}).get("ri_orth_error"),
            "step4b_delta_obs_rdm_minus_fci": step4b_meta.get("delta_obs_rdm_minus_fci"),
            "step4b_delta_ri_rdm_minus_fci": step4b_meta.get("delta_ri_rdm_minus_fci"),
            "step6f_json": names["step6f_json"],
        }
        rows.append(out)
    return rows


def fmt(x: Any, prec: int = 8) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{prec}e}"
    except Exception:
        return str(x)


def write_outputs(args, rows: list[Dict[str, Any]], fits: Dict[int, Dict[str, Any]], cases: list[tuple[str, int]]):
    payload = {
        "scan": {
            "nterms": sorted(fits),
            "cases": [{"parent_basis": b, "nobs": n} for b, n in cases],
            "fock_model": args.fock_model,
            "note": "Rows are the paper_tequila_sf2r12 candidate from Step 6f.",
        },
        "fits": fits,
        "rows": rows,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "parent_basis",
        "nobs",
        "nterms",
        "nri",
        "ncabs",
        "fock_model",
        "delta_E",
        "abs_residual_to_full_mEh",
        "recovery_ratio",
        "full_parent_gap",
        "V_component",
        "B_component",
        "X_component",
        "Delta_component",
        "fit_RMS_error",
        "fit_max_error",
        "fit_relative_RMS_error",
        "fit_short_range_RMS_error",
        "fit_short_range_max_error",
        "fit_f0_error",
        "step4b_ri_orth_error",
        "step4b_delta_obs_rdm_minus_fci",
        "step4b_delta_ri_rdm_minus_fci",
        "step6f_json",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6l | paper_tequila_sf2r12 fit/basis convergence scan")
    lines.append("=" * 100)
    lines.append("[Fit metrics]")
    lines.append("| fitN | RMS | max | rel RMS | f0 err |")
    lines.append("|---:|---:|---:|---:|---:|")
    for n in sorted(fits):
        m = fits[n]["metrics"]
        lines.append(
            f"| {n} | {m['rms_abs']:.8e} | {m['max_abs']:.8e} | "
            f"{m['rel_rms']:.8e} | {m['f0_error']:.8e} |"
        )
    lines.append("")
    for basis, nobs in cases:
        lines.append(f"[Case {basis} / nobs={nobs}]")
        lines.append("| fitN | nri | fock model | DeltaE / mEh | recovery | residual / mEh | V / mEh | B / mEh | X / mEh |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        subset = [r for r in rows if r["parent_basis"] == basis and int(r["nobs"]) == int(nobs)]
        subset.sort(key=lambda r: (int(r["nterms"]), str(r["fock_model"])))
        for r in subset:
            lines.append(
                f"| {r['nterms']} | {r['nri']} | {r['fock_model']} "
                f"| {1000.0 * r['delta_E']:.8f} | {fmt(r['recovery_ratio'], 8)} "
                f"| {fmt(r['abs_residual_to_full_mEh'], 8)} "
                f"| {1000.0 * r['V_component']:.8f} | {1000.0 * r['B_component']:.8f} "
                f"| {1000.0 * r['X_component']:.8f} |"
            )
        lines.append("")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))
    print("\n[Saved]")
    print(f"  {args.out_json}")
    print(f"  {args.out_csv}")
    print(f"  {args.summary}")


def main():
    args = parse_args()
    nterms_list = parse_ints(args.nterms)
    cases = parse_cases(args.cases)
    fits = {n: fit_corr(args, n) for n in nterms_list}

    rows = []
    for basis, nobs in cases:
        for n in nterms_list:
            fit = fits[n]
            names = ensure_pipeline(args, basis, nobs, n, fit["corr_string"])
            if not args.dry_run:
                rows.extend(load_paper_rows(names, fit, basis, nobs, n, args.fock_model))

    if not args.dry_run:
        write_outputs(args, rows, fits, cases)


if __name__ == "__main__":
    main()
