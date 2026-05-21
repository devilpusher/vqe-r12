#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6b: Collect fitted-Slater convergence diagnostics.

This script runs, or reuses, Step 6a fitted Slater Gaussian expansions for a
set of nterms and collects both fit metrics and Step-5c prototype pair-space
diagnostics.  The full Q-pair solve remains a diagnostic reference; it is not
the final approximation-C/SP [2]R12 contraction.

Default pipeline:

    python step6b_collect_fit_convergence.py

Equivalent explicit Step-6a commands:

    python step6a_fit_slater_corr.py --nterms 3 --run-pipeline
    python step6a_fit_slater_corr.py --nterms 5 --run-pipeline
    python step6a_fit_slater_corr.py --nterms 7 --run-pipeline
    python step6a_fit_slater_corr.py --nterms 9 --run-pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nterms-list", default="3,5,7,9")
    p.add_argument("--parent-basis", default="cc-pvdz")
    p.add_argument("--nobs", type=int, default=2)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--force", action="store_true", help="Re-run Step 6a even if outputs exist.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-prefix", default=None)
    return p.parse_args()


def safe_basis_label(basis: str) -> str:
    return basis.lower().replace("*", "s").replace("+", "p").replace("-", "").replace("_", "")


def parse_nterms(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("--nterms-list is empty")
    return vals


def run_command(cmd: List[str], dry_run: bool = False):
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def expected_paths(parent_basis: str, nobs: int, nterms: int) -> Dict[str, Path]:
    basis = safe_basis_label(parent_basis)
    prefix = f"he_{basis}_nobs{nobs}_fitN{nterms}"
    return {
        "fit_json": Path(f"step6a_slater_fit_N{nterms}.json"),
        "fit_txt": Path(f"step6a_slater_fit_N{nterms}.txt"),
        "step5c_json": Path(f"{prefix}_step5c_correction_comparison.json"),
        "step5c_csv": Path(f"{prefix}_step5c_correction_comparison.csv"),
        "step5c_summary": Path(f"{prefix}_step5c_correction_comparison_summary.txt"),
    }


def ensure_pipeline(args, nterms: int, paths: Dict[str, Path]):
    needed = [paths["fit_json"], paths["step5c_json"], paths["step5c_csv"], paths["step5c_summary"]]
    if not args.force and all(p.exists() for p in needed):
        print(f"[reuse] fitN{nterms} outputs already exist")
        return
    cmd = [
        args.python,
        "step6a_fit_slater_corr.py",
        "--gamma",
        str(args.gamma),
        "--nterms",
        str(nterms),
        "--parent-basis",
        args.parent_basis,
        "--nobs",
        str(args.nobs),
        "--run-pipeline",
    ]
    run_command(cmd, dry_run=args.dry_run)


def row_for_method(rows: List[Dict[str, Any]], method: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def energy(rows: List[Dict[str, Any]], method: str) -> Optional[float]:
    row = row_for_method(rows, method)
    return None if row is None else row.get("energy_Eh")


def collect_row(nterms: int, paths: Dict[str, Path]) -> Dict[str, Any]:
    with paths["fit_json"].open("r", encoding="utf-8") as f:
        fit = json.load(f)
    with paths["step5c_json"].open("r", encoding="utf-8") as f:
        step5c = json.load(f)

    metrics = fit["metrics"]
    regions = metrics.get("region_metrics", {})
    short = regions.get("short_r_le_1", {})
    rows = step5c.get("correction_rows", [])
    core = step5c.get("core_energies", {})
    full_q = row_for_method(rows, "Full Q-pair solve") or {}

    return {
        "nterms": nterms,
        "fit_RMS_error": metrics.get("rms_abs"),
        "fit_relative_RMS_error": metrics.get("rel_rms"),
        "fit_max_error": metrics.get("max_abs"),
        "fit_short_range_RMS_error": short.get("rms_abs"),
        "fit_short_range_max_error": short.get("max_abs"),
        "f0_error": metrics.get("f0_error"),
        "design_condition_number": metrics.get("design_condition_number"),
        "n_positive_coefficients": metrics.get("n_positive_coefficients"),
        "E_OBS_FCI": core.get("E_obs_fci"),
        "E_full_parent_FCI": core.get("E_full_parent_fci"),
        "E_raw_1D_opt": energy(rows, "Raw F12 1D optimized"),
        "E_sp_1D_opt": energy(rows, "SP-F12 1D optimized"),
        "E_full_Q_pair": energy(rows, "Full Q-pair solve"),
        "E_diag_EN_like": energy(rows, "Diagonal EN-like PT2"),
        "residual_to_full_parent_FCI": full_q.get("residual_to_full_parent_FCI_Eh"),
        "recovery_ratio": full_q.get("recovery_ratio_vs_full_truncation"),
        "corr_string": fit.get("corr_string"),
        "step5c_json": str(paths["step5c_json"]),
        "step5c_summary": str(paths["step5c_summary"]),
    }


def write_outputs(prefix: str, rows: List[Dict[str, Any]]):
    json_path = Path(f"{prefix}.json")
    csv_path = Path(f"{prefix}.csv")
    txt_path = Path(f"{prefix}.txt")

    payload = {
        "warning": (
            "These are fitted Gaussian expansion and prototype pair-space diagnostics. "
            "They are not final article-level [2]R12 approximation-C/SP corrections."
        ),
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6b | fitted Slater convergence through prototype pair-space diagnostics")
    lines.append("=" * 100)
    lines.append("")
    lines.append("| N | fit RMS | fit max | f0 err | E raw 1D | E SP 1D | E full Q | E diag EN | residual full Q | recovery |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['nterms']} | {r['fit_RMS_error']:.6e} | {r['fit_max_error']:.6e} "
            f"| {r['f0_error']:.6e} | {r['E_raw_1D_opt']:.12f} | {r['E_sp_1D_opt']:.12f} "
            f"| {r['E_full_Q_pair']:.12f} | {r['E_diag_EN_like']:.12f} "
            f"| {r['residual_to_full_parent_FCI']:.6e} | {r['recovery_ratio']:.6f} |"
        )
    lines.append("")
    lines.append("Reminder: Gaussian expansions have zero cusp derivative at r=0 and cannot exactly reproduce the Slater cusp.")
    lines.append("Fixed-amplitude energies are diagnostic only; use optimized/full-Q rows only as prototype checks.")

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return json_path, csv_path, txt_path


def main():
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for nterms in parse_nterms(args.nterms_list):
        paths = expected_paths(args.parent_basis, args.nobs, nterms)
        ensure_pipeline(args, nterms, paths)
        if not args.dry_run:
            rows.append(collect_row(nterms, paths))

    if args.dry_run:
        return

    basis = safe_basis_label(args.parent_basis)
    out_prefix = args.out_prefix or f"step6b_he_{basis}_nobs{args.nobs}_fit_convergence"
    json_path, csv_path, txt_path = write_outputs(out_prefix, rows)
    print("[Saved]")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
