#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6a: Fit Slater-type F12 correlation factor by Gaussian expansion.

Target
------
The article uses a Slater-type geminal

    f12(r) = -1/gamma * exp(-gamma r)

Psi4's F12 integral API expects a Gaussian correlation list

    corr = [(alpha_1, c_1), (alpha_2, c_2), ...]

with convention established in Step 3c:

    f12(r) ~= sum_k c_k exp(-alpha_k r^2)

where alpha_k is the Gaussian exponent and c_k is the coefficient.

This script constructs a controllable least-squares Gaussian expansion for the
Slater-type factor and prints a corr string that can be passed directly to the
previous scripts:

    --corr "alpha1,c1;alpha2,c2;..."

It can also optionally run the existing Step 4b -> 5a -> 5b -> 5c pipeline with
the fitted corr string.

Important
---------
A finite Gaussian expansion cannot exactly reproduce the electron-electron
cusp because d/dr exp(-alpha r^2)|_{r=0}=0.  This script provides a controlled
and reproducible approximation for testing the pipeline.  A publication-level
implementation should either use a validated fitted-Slater expansion from
Psi4/Libint/Tequila or document the fitting protocol and convergence with
respect to nterms, alpha range, and radial fitting interval.

Usage
-----
Fit only:

    python step6a_fit_slater_corr.py --nterms 6

Run the full existing pipeline with the fitted factor:

    python step6a_fit_slater_corr.py --nterms 6 --run-pipeline

Use a wider exponent range:

    python step6a_fit_slater_corr.py --nterms 8 --alpha-min 0.05 --alpha-max 80 --rmax 8.0

Outputs
-------
    step6a_slater_fit_corr.json
    step6a_slater_fit_corr.csv
    step6a_slater_fit_corr.txt

If --run-pipeline is used, it also creates files with prefix:

    he_ccpvdz_fitN<terms>_...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=1.4, help="Slater exponent gamma.")
    p.add_argument("--nterms", type=int, default=6, help="Number of Gaussian terms.")
    p.add_argument("--alpha-min", type=float, default=0.08, help="Smallest Gaussian exponent.")
    p.add_argument("--alpha-max", type=float, default=60.0, help="Largest Gaussian exponent.")
    p.add_argument("--rmin", type=float, default=0.0, help="Smallest r in fitting grid.")
    p.add_argument("--rmax", type=float, default=8.0, help="Largest r in fitting grid.")
    p.add_argument("--ngrid", type=int, default=2000, help="Number of radial grid points.")
    p.add_argument(
        "--grid",
        choices=["linear", "quadratic"],
        default="quadratic",
        help="Radial grid distribution. quadratic puts more points near r=0.",
    )
    p.add_argument(
        "--weight",
        choices=["flat", "r2", "short", "relative"],
        default="short",
        help=(
            "Least-squares weights. flat: uniform; r2: radial volume weight; "
            "short: emphasizes short/intermediate range; relative: relative-error fit."
        ),
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Optional Tikhonov ridge regularization for coefficients.",
    )
    p.add_argument(
        "--nonpositive-coeff",
        action="store_true",
        help="Force coefficients to be nonpositive by fitting positive amplitudes to -f(r). "
             "Uses a simple active-set clipping iteration, no scipy required.",
    )
    p.add_argument("--parent-basis", default="cc-pvdz", help="Parent basis if running pipeline.")
    p.add_argument("--nobs", type=int, default=2, help="OBS orbitals if running pipeline.")
    p.add_argument("--prefix", default=None, help="Output prefix. Default depends on nterms.")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-txt", default=None)
    p.add_argument("--run-pipeline", action="store_true", help="Run Step 4b, 5a, 5b, 5c with fitted corr.")
    p.add_argument("--python", default=sys.executable, help="Python executable for subprocess pipeline.")
    p.add_argument("--dry-run", action="store_true", help="Print commands but do not run pipeline.")
    return p.parse_args()


def target_slater(r: np.ndarray, gamma: float) -> np.ndarray:
    return -(1.0 / gamma) * np.exp(-gamma * r)


def make_grid(rmin: float, rmax: float, ngrid: int, kind: str) -> np.ndarray:
    if ngrid < 10:
        raise ValueError("--ngrid should be at least 10")
    if kind == "linear":
        r = np.linspace(rmin, rmax, ngrid)
    elif kind == "quadratic":
        x = np.linspace(0.0, 1.0, ngrid)
        r = rmin + (rmax - rmin) * x * x
    else:
        raise ValueError(kind)
    return r


def make_weights(r: np.ndarray, f: np.ndarray, mode: str, gamma: float) -> np.ndarray:
    if mode == "flat":
        w = np.ones_like(r)
    elif mode == "r2":
        # Volume-like radial weight; avoid exactly zero weight at r=0.
        w = np.maximum(r * r, 1e-8)
    elif mode == "short":
        # Emphasize the chemically relevant short/intermediate range.
        # The factor keeps the long-range tail from dominating the least-squares norm.
        w = np.exp(-0.5 * gamma * r) + 0.05
    elif mode == "relative":
        # Approximate relative-error fit.  Avoid blow-up in the tail.
        floor = max(1e-6, 1e-4 / gamma)
        w = 1.0 / np.maximum(np.abs(f), floor)
    else:
        raise ValueError(mode)
    return np.sqrt(w)


def geometric_exponents(alpha_min: float, alpha_max: float, nterms: int) -> np.ndarray:
    if nterms < 1:
        raise ValueError("--nterms must be positive")
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValueError("Gaussian exponents must be positive")
    if nterms == 1:
        return np.array([math.sqrt(alpha_min * alpha_max)], dtype=float)
    return np.geomspace(alpha_min, alpha_max, nterms)


def solve_linear_coeffs(A: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    if ridge <= 0.0:
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return c
    ATA = A.T @ A
    ATy = A.T @ y
    return np.linalg.solve(ATA + ridge * np.eye(ATA.shape[0]), ATy)


def solve_nonpositive_coeffs(A: np.ndarray, y: np.ndarray, ridge: float = 0.0, max_iter: int = 30) -> np.ndarray:
    """
    Fit y with A c subject to c <= 0.

    Since y is negative, define b=-c >=0 and fit -y ~= A b.
    A simple active-set clipping loop is used to avoid scipy dependency.
    """
    yy = -y
    active = np.ones(A.shape[1], dtype=bool)
    b = np.zeros(A.shape[1], dtype=float)

    for _ in range(max_iter):
        if not np.any(active):
            break
        b_active = solve_linear_coeffs(A[:, active], yy, ridge=ridge)
        bad = b_active < 0.0
        if not np.any(bad):
            b[active] = b_active
            break
        active_idx = np.where(active)[0]
        # Deactivate the most negative coefficient and repeat.
        worst_local = int(np.argmin(b_active))
        active[active_idx[worst_local]] = False
        b[:] = 0.0
    b = np.maximum(b, 0.0)
    return -b


def fit_gaussian_expansion(args) -> Dict[str, Any]:
    r = make_grid(args.rmin, args.rmax, args.ngrid, args.grid)
    f = target_slater(r, args.gamma)
    alpha = geometric_exponents(args.alpha_min, args.alpha_max, args.nterms)
    G = np.exp(-np.outer(r * r, alpha))

    w = make_weights(r, f, args.weight, args.gamma)
    Aw = G * w[:, None]
    yw = f * w

    if args.nonpositive_coeff:
        coeff = solve_nonpositive_coeffs(Aw, yw, ridge=args.ridge)
    else:
        coeff = solve_linear_coeffs(Aw, yw, ridge=args.ridge)

    singular_values = np.linalg.svd(Aw, compute_uv=False)
    if singular_values.size and singular_values[-1] > 0.0:
        design_condition = float(singular_values[0] / singular_values[-1])
    else:
        design_condition = None

    f_fit = G @ coeff
    err = f_fit - f

    # Metrics on the fitting grid.
    rms = float(np.sqrt(np.mean(err * err)))
    max_abs = float(np.max(np.abs(err)))
    target_norm = float(np.sqrt(np.mean(f * f)))
    rel_rms = float(rms / target_norm) if target_norm != 0.0 else None

    # Short-range and tail metrics.
    masks = {
        "short_r_le_1": r <= 1.0,
        "mid_1_lt_r_le_3": (r > 1.0) & (r <= 3.0),
        "tail_r_gt_3": r > 3.0,
    }
    region_metrics = {}
    for name, m in masks.items():
        if not np.any(m):
            continue
        em = err[m]
        fm = f[m]
        rms_m = float(np.sqrt(np.mean(em * em)))
        norm_m = float(np.sqrt(np.mean(fm * fm)))
        region_metrics[name] = {
            "rms_abs": rms_m,
            "max_abs": float(np.max(np.abs(em))),
            "rel_rms": float(rms_m / norm_m) if norm_m != 0.0 else None,
        }

    # Cusp diagnostics.  Gaussian expansion derivative at r=0 is exactly zero.
    f0 = float(target_slater(np.array([0.0]), args.gamma)[0])
    ffit0 = float(np.sum(coeff))
    derivative_target_0 = 1.0  # d[-1/gamma exp(-gamma r)]/dr at r=0
    derivative_fit_0 = 0.0

    corr = [(float(a), float(c)) for a, c in zip(alpha, coeff)]
    corr_string = ";".join(f"{a:.16g},{c:.16g}" for a, c in corr)

    # Sample values for text inspection.
    sample_r = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, args.rmax])
    sample_f = target_slater(sample_r, args.gamma)
    sample_G = np.exp(-np.outer(sample_r * sample_r, alpha))
    sample_fit = sample_G @ coeff
    samples = []
    for rr, yy, yyfit in zip(sample_r, sample_f, sample_fit):
        samples.append({
            "r": float(rr),
            "target": float(yy),
            "fit": float(yyfit),
            "error": float(yyfit - yy),
        })

    return {
        "gamma": args.gamma,
        "target": "f(r) = -1/gamma * exp(-gamma*r)",
        "gaussian_form": "sum_k c_k exp(-alpha_k*r^2)",
        "corr_convention": "Psi4 corr = [(alpha_k, c_k), ...]",
        "fit_protocol": "local least-squares Gaussian expansion; not Psi4's official fitted Slater object",
        "nterms": args.nterms,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "rmin": args.rmin,
        "rmax": args.rmax,
        "ngrid": args.ngrid,
        "grid": args.grid,
        "weight": args.weight,
        "ridge": args.ridge,
        "nonpositive_coeff": bool(args.nonpositive_coeff),
        "corr": corr,
        "alpha_list": [float(x) for x in alpha],
        "coefficient_list": [float(x) for x in coeff],
        "corr_string": corr_string,
        "metrics": {
            "rms_abs": rms,
            "max_abs": max_abs,
            "target_rms_norm": target_norm,
            "rel_rms": rel_rms,
            "design_condition_number": design_condition,
            "coefficient_l2_norm": float(np.linalg.norm(coeff)),
            "n_positive_coefficients": int(np.sum(coeff > 0.0)),
            "has_nan": bool(np.isnan(coeff).any() or np.isnan(f_fit).any()),
            "has_inf": bool(np.isinf(coeff).any() or np.isinf(f_fit).any()),
            "region_metrics": region_metrics,
            "f0_target": f0,
            "f0_fit": ffit0,
            "f0_error": ffit0 - f0,
            "derivative_target_at_0": derivative_target_0,
            "derivative_fit_at_0": derivative_fit_0,
        },
        "samples": samples,
    }


def write_outputs(args, fit: Dict[str, Any]):
    prefix = args.prefix or f"step6a_slater_fit_N{args.nterms}"
    out_json = args.out_json or f"{prefix}.json"
    out_csv = args.out_csv or f"{prefix}.csv"
    out_txt = args.out_txt or f"{prefix}.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(fit, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "alpha", "coefficient"])
        writer.writeheader()
        for k, (a, c) in enumerate(fit["corr"]):
            writer.writerow({"k": k, "alpha": a, "coefficient": c})

    lines = []
    lines.append("=" * 80)
    lines.append("Step 6a | Fitted Slater-type F12 Gaussian expansion")
    lines.append("=" * 80)
    lines.append(f"target        : {fit['target']}")
    lines.append(f"gaussian form : {fit['gaussian_form']}")
    lines.append(f"gamma         : {fit['gamma']}")
    lines.append(f"nterms        : {fit['nterms']}")
    lines.append(f"alpha range   : [{fit['alpha_min']}, {fit['alpha_max']}]")
    lines.append(f"r range/grid  : [{fit['rmin']}, {fit['rmax']}], ngrid={fit['ngrid']}, grid={fit['grid']}")
    lines.append(f"weight        : {fit['weight']}")
    lines.append(f"protocol      : {fit['fit_protocol']}")
    lines.append("")
    lines.append("[Psi4 corr string]")
    lines.append(fit["corr_string"])
    lines.append("")
    lines.append("[Gaussian terms]")
    for k, (a, c) in enumerate(fit["corr"]):
        lines.append(f"{k:3d}  alpha={a: .16e}  coefficient={c: .16e}")
    lines.append("")
    lines.append("[Metrics]")
    m = fit["metrics"]
    lines.append(f"RMS abs error        = {m['rms_abs']:.12e}")
    lines.append(f"Max abs error        = {m['max_abs']:.12e}")
    lines.append(f"Relative RMS error   = {m['rel_rms']:.12e}")
    lines.append(f"Design cond. number  = {m['design_condition_number']}")
    lines.append(f"Coefficient L2 norm  = {m['coefficient_l2_norm']:.12e}")
    lines.append(f"Positive coeff count = {m['n_positive_coefficients']}")
    lines.append(f"has NaN / has Inf    = {m['has_nan']} / {m['has_inf']}")
    lines.append(f"f(0) target/fit/err  = {m['f0_target']:.12e} / {m['f0_fit']:.12e} / {m['f0_error']:.12e}")
    lines.append(f"f'(0) target/fit     = {m['derivative_target_at_0']:.12e} / {m['derivative_fit_at_0']:.12e}")
    lines.append("")
    lines.append("[Region metrics]")
    for name, rm in m["region_metrics"].items():
        lines.append(f"{name}: rms={rm['rms_abs']:.12e}, max={rm['max_abs']:.12e}, rel_rms={rm['rel_rms']:.12e}")
    lines.append("")
    lines.append("[Samples]")
    for s in fit["samples"]:
        lines.append(f"r={s['r']: .4f}  target={s['target']: .12e}  fit={s['fit']: .12e}  err={s['error']: .12e}")
    lines.append("")
    lines.append("[Note]")
    lines.append("This is a controllable Gaussian fit for testing. It does not exactly reproduce the cusp.")
    lines.append("Converge nterms, alpha range, and radial fit interval before using quantitative results.")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return out_json, out_csv, out_txt


def run_command(cmd: List[str], dry_run: bool = False):
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_pipeline(args, fit: Dict[str, Any]):
    basis_label = args.parent_basis.replace("-", "").lower()
    prefix = args.prefix or f"he_{basis_label}_nobs{args.nobs}_fitN{args.nterms}"
    corr = fit["corr_string"]

    step4b_out = f"{prefix}_step4b_obs_fci_rdm.npz"
    step4b_sum = f"{prefix}_step4b_obs_fci_rdm_summary.txt"
    step5a_out = f"{prefix}_step5a_r12_intermediates.npz"
    step5a_sum = f"{prefix}_step5a_r12_intermediates_summary.txt"
    step5b_out = f"{prefix}_step5b_r12_prototype_correction.npz"
    step5b_sum = f"{prefix}_step5b_r12_prototype_correction_summary.txt"
    step5c_csv = f"{prefix}_step5c_correction_comparison.csv"
    step5c_json = f"{prefix}_step5c_correction_comparison.json"
    step5c_sum = f"{prefix}_step5c_correction_comparison_summary.txt"

    commands = [
        [
            args.python, "step4b_he_parent_obs_fci_rdm_check.py",
            "--parent-basis", args.parent_basis,
            "--nobs", str(args.nobs),
            "--corr", corr,
            "--out", step4b_out,
            "--summary", step4b_sum,
        ],
        [
            args.python, "step5a_he_r12_intermediate_check.py",
            "--inp", step4b_out,
            "--out", step5a_out,
            "--summary", step5a_sum,
        ],
        [
            args.python, "step5b_he_r12_prototype_correction.py",
            "--inp", step5a_out,
            "--out", step5b_out,
            "--summary", step5b_sum,
        ],
        [
            args.python, "step5c_he_correction_comparison.py",
            "--step4b", step4b_out,
            "--step5b", step5b_out,
            "--csv", step5c_csv,
            "--json", step5c_json,
            "--summary", step5c_sum,
        ],
    ]

    print("\n[Pipeline commands]")
    for cmd in commands:
        run_command(cmd, dry_run=args.dry_run)

    return {
        "step4b_out": step4b_out,
        "step4b_summary": step4b_sum,
        "step5a_out": step5a_out,
        "step5a_summary": step5a_sum,
        "step5b_out": step5b_out,
        "step5b_summary": step5b_sum,
        "step5c_csv": step5c_csv,
        "step5c_json": step5c_json,
        "step5c_summary": step5c_sum,
    }


def main():
    args = parse_args()

    fit = fit_gaussian_expansion(args)
    out_json, out_csv, out_txt = write_outputs(args, fit)

    print("=" * 80)
    print("Step 6a | fitted Slater-type Gaussian expansion")
    print("=" * 80)
    print(f"nterms = {fit['nterms']}, gamma = {fit['gamma']}")
    print(f"corr string:\n{fit['corr_string']}")
    print("")
    print("[Fit metrics]")
    print(f"RMS abs error      = {fit['metrics']['rms_abs']:.6e}")
    print(f"Max abs error      = {fit['metrics']['max_abs']:.6e}")
    print(f"Relative RMS error = {fit['metrics']['rel_rms']:.6e}")
    print(f"Design cond. no.   = {fit['metrics']['design_condition_number']}")
    print(f"Positive coeffs    = {fit['metrics']['n_positive_coefficients']}")
    print(f"f(0) target/fit    = {fit['metrics']['f0_target']:.6e} / {fit['metrics']['f0_fit']:.6e}")
    print("")
    print("[Saved]")
    print(" ", out_json)
    print(" ", out_csv)
    print(" ", out_txt)

    if args.run_pipeline:
        outputs = run_pipeline(args, fit)
        print("\n[Pipeline outputs]")
        for k, v in outputs.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
