#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6b: Scan fitted Slater Gaussian expansions and optional pipeline response.

Step 6a turns one Slater-type geminal into a Gaussian corr list.  Step 6b
compares several fit protocols so the next R12 development step is not tied to
one arbitrary N=6 expansion.

By default this script performs fit-only scans.  With --run-pipeline it also
runs Step 4b -> 5a -> 5b -> 5c for each fit and records the final Step-5c
energy diagnostics.

Examples
--------
Fit-only scan:

    python step6b_scan_slater_pipeline.py

Run selected fits through the existing correction diagnostics:

    python step6b_scan_slater_pipeline.py --nterms-list 4,6,8 --run-pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from step6a_fit_slater_corr import fit_gaussian_expansion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--nterms-list", default="4,6,8,10")
    p.add_argument("--alpha-min", type=float, default=0.08)
    p.add_argument("--alpha-max", type=float, default=60.0)
    p.add_argument("--rmin", type=float, default=0.0)
    p.add_argument("--rmax", type=float, default=8.0)
    p.add_argument("--ngrid", type=int, default=2000)
    p.add_argument("--grid", choices=["linear", "quadratic"], default="quadratic")
    p.add_argument("--weights", default="short,relative")
    p.add_argument("--include-unconstrained", action="store_true")
    p.add_argument("--ridge", type=float, default=0.0)
    p.add_argument("--parent-basis", default="cc-pvdz")
    p.add_argument("--nobs", type=int, default=2)
    p.add_argument("--run-pipeline", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out-prefix", default="step6b_slater_scan")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_int_list(text: str) -> List[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("empty integer list")
    return vals


def parse_str_list(text: str) -> List[str]:
    vals = [part.strip() for part in text.split(",") if part.strip()]
    if not vals:
        raise ValueError("empty string list")
    return vals


def namespace_for_fit(args, nterms: int, weight: str, nonpositive: bool):
    class FitArgs:
        pass

    fit_args = FitArgs()
    fit_args.gamma = args.gamma
    fit_args.nterms = nterms
    fit_args.alpha_min = args.alpha_min
    fit_args.alpha_max = args.alpha_max
    fit_args.rmin = args.rmin
    fit_args.rmax = args.rmax
    fit_args.ngrid = args.ngrid
    fit_args.grid = args.grid
    fit_args.weight = weight
    fit_args.ridge = args.ridge
    fit_args.nonpositive_coeff = nonpositive
    return fit_args


def run_command(cmd: List[str], dry_run: bool = False):
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def load_step5c_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("correction_rows", [])
    out: Dict[str, Any] = {}
    for row in rows:
        method = row.get("method")
        if method in {
            "Full parent FCI",
            "Raw F12 1D optimized",
            "SP-F12 1D optimized",
            "Full Q-pair solve",
            "Diagonal EN-like PT2",
        }:
            key = method.lower().replace(" ", "_").replace("-", "_")
            out[f"{key}_energy"] = row.get("energy")
            out[f"{key}_delta_from_obs"] = row.get("delta_from_obs")
            out[f"{key}_residual_to_full"] = row.get("residual_to_full")
            out[f"{key}_recovery_ratio"] = row.get("recovery_ratio")

    core = data.get("core_energies", {})
    out["delta_full"] = core.get("delta_full")
    return out


def run_existing_pipeline(args, tag: str, corr_string: str) -> Dict[str, Any]:
    prefix = f"he_{args.parent_basis.replace('-', '').lower()}_{tag}"
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
            args.python,
            "step4b_he_parent_obs_fci_rdm_check.py",
            "--parent-basis",
            args.parent_basis,
            "--nobs",
            str(args.nobs),
            "--corr",
            corr_string,
            "--out",
            step4b_out,
            "--summary",
            step4b_sum,
        ],
        [
            args.python,
            "step5a_he_r12_intermediate_check.py",
            "--inp",
            step4b_out,
            "--out",
            step5a_out,
            "--summary",
            step5a_sum,
        ],
        [
            args.python,
            "step5b_he_r12_prototype_correction.py",
            "--inp",
            step5a_out,
            "--out",
            step5b_out,
            "--summary",
            step5b_sum,
        ],
        [
            args.python,
            "step5c_he_correction_comparison.py",
            "--step4b",
            step4b_out,
            "--step5b",
            step5b_out,
            "--csv",
            step5c_csv,
            "--json",
            step5c_json,
            "--summary",
            step5c_sum,
        ],
    ]

    for cmd in commands:
        run_command(cmd, dry_run=args.dry_run)

    outputs = {
        "pipeline_prefix": prefix,
        "step4b_out": step4b_out,
        "step5a_out": step5a_out,
        "step5b_out": step5b_out,
        "step5c_json": step5c_json,
        "step5c_summary": step5c_sum,
    }
    if not args.dry_run:
        outputs.update(load_step5c_summary(Path(step5c_json)))
    return outputs


def flatten_fit_metrics(fit: Dict[str, Any]) -> Dict[str, Any]:
    metrics = fit["metrics"]
    regions = metrics["region_metrics"]
    row: Dict[str, Any] = {
        "gamma": fit["gamma"],
        "nterms": fit["nterms"],
        "weight": fit["weight"],
        "nonpositive_coeff": fit["nonpositive_coeff"],
        "alpha_min": fit["alpha_min"],
        "alpha_max": fit["alpha_max"],
        "rmax": fit["rmax"],
        "rms_abs": metrics["rms_abs"],
        "max_abs": metrics["max_abs"],
        "rel_rms": metrics["rel_rms"],
        "f0_error": metrics["f0_error"],
        "corr_string": fit["corr_string"],
    }
    for name, vals in regions.items():
        row[f"{name}_rms_abs"] = vals["rms_abs"]
        row[f"{name}_max_abs"] = vals["max_abs"]
        row[f"{name}_rel_rms"] = vals["rel_rms"]
    return row


def write_outputs(prefix: str, rows: List[Dict[str, Any]]):
    json_path = Path(f"{prefix}.json")
    csv_path = Path(f"{prefix}.csv")
    txt_path = Path(f"{prefix}.txt")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("=" * 80)
    lines.append("Step 6b | Slater Gaussian fit scan")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| nterms | weight | constrained | rel_rms | f0_error | max_abs | pipeline residual / Eh | recovery |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        residual = row.get("full_q_pair_solve_residual_to_full")
        recovery = row.get("full_q_pair_solve_recovery_ratio")
        residual_text = "" if residual is None else f"{residual:.6e}"
        recovery_text = "" if recovery is None else f"{recovery:.6f}"
        lines.append(
            f"| {row['nterms']} | {row['weight']} | {int(row['nonpositive_coeff'])} "
            f"| {row['rel_rms']:.6e} | {row['f0_error']:.6e} | {row['max_abs']:.6e} "
            f"| {residual_text} | {recovery_text} |"
        )
    lines.append("")
    lines.append("Use fit-only metrics to select stable Gaussian expansions before interpreting pipeline energies.")

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return json_path, csv_path, txt_path


def main():
    args = parse_args()
    nterms_list = parse_int_list(args.nterms_list)
    weights = parse_str_list(args.weights)
    constraints = [True]
    if args.include_unconstrained:
        constraints.insert(0, False)

    rows: List[Dict[str, Any]] = []
    for nterms in nterms_list:
        for weight in weights:
            for nonpositive in constraints:
                tag = f"fitN{nterms}_{weight}"
                if nonpositive:
                    tag += "_nonpos"
                print(f"\n[Fit] {tag}")
                fit_args = namespace_for_fit(args, nterms, weight, nonpositive)
                fit = fit_gaussian_expansion(fit_args)
                row = flatten_fit_metrics(fit)
                row["tag"] = tag

                if args.run_pipeline:
                    row.update(run_existing_pipeline(args, tag, fit["corr_string"]))

                rows.append(row)

    json_path, csv_path, txt_path = write_outputs(args.out_prefix, rows)
    print("\n[Saved]")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
