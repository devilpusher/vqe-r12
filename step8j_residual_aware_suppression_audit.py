#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8j: Residual-aware audit for HEM same-spin suppression rules.

Step8i showed that scalar rules such as

    lambda_ss = 2 * sqrt(tail occupation)
    lambda_ss = 3 * sqrt(residual pair weight)

work well in the smallest HEM space but drift across larger even s/p spaces.
This script quantifies whether that drift can be repaired by a single
residual-aware scalar, or whether the rule must become channel/pair resolved.

Rows labeled `target_fit` or `oracle_*` use the dense same-spin parent pair-FCI
gap and are diagnostics only.  They are not production formulas.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scan-csv", default="step8i_hem_triplet_open_shell_space_scan.csv")
    p.add_argument("--ensure-scan", action="store_true", help="Run Step8i first if --scan-csv is missing.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--prefix", default="step8j_hem_triplet_residual_aware_audit")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def read_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in [
            "nobs",
            "lambda",
            "tail_sqrt",
            "pair_sqrt",
            "c_tail_equiv",
            "c_pair_equiv",
            "E_obs",
            "E_parent",
            "target_gap_mEh",
            "correction_mEh",
            "residual_mEh",
            "V_mEh",
            "B_mEh",
            "X_mEh",
            "Delta_mEh",
        ]:
            r[key] = float(r[key])
        r["nobs"] = int(r["nobs"])
    return rows


def grouped(rows: List[Dict[str, Any]]) -> Dict[tuple[str, str], Dict[str, Dict[str, Any]]]:
    out: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        out[(r["case"], r["fock_model"])][r["rule"]] = r
    return out


def unit_coefficients(row: Dict[str, Any]) -> tuple[float, float]:
    """Return V_unit, Q_unit for correction(lambda)=lambda*V_unit+lambda^2*Q_unit."""
    lam = row["lambda"]
    q_mEh = row["B_mEh"] + row["X_mEh"] + row["Delta_mEh"]
    return row["V_mEh"] / lam, q_mEh / (lam * lam)


def eval_lambda(row: Dict[str, Any], lam: float) -> Dict[str, float]:
    v_unit, q_unit = unit_coefficients(row)
    v = lam * v_unit
    q = lam * lam * q_unit
    corr = v + q
    target = row["target_gap_mEh"]
    return {
        "lambda": lam,
        "V_mEh": v,
        "Q_mEh": q,
        "correction_mEh": corr,
        "target_gap_mEh": target,
        "residual_mEh": corr - target,
    }


def scalar_constant(rows: List[Dict[str, Any]], fock_model: str, feature: str, leave_out: str | None = None) -> float:
    key = "c_tail_equiv" if feature == "tail" else "c_pair_equiv"
    vals = [
        r[key]
        for r in rows
        if r["fock_model"] == fock_model and r["rule"] == "target_fit" and r["case"] != leave_out
    ]
    return float(np.mean(vals))


def build_audit_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    g = grouped(rows)
    audit: List[Dict[str, Any]] = []
    for (case, fock), rules in sorted(g.items()):
        target = rules["target_fit"]
        for base_rule, feature in [("tail2", "tail"), ("pair3", "pair")]:
            base = rules[base_rule]
            base_lam = base["lambda"]
            target_lam = target["lambda"]
            oracle_factor = target_lam / base_lam
            audit.append(
                {
                    "case": case,
                    "nobs": base["nobs"],
                    "fock_model": fock,
                    "family": feature,
                    "model": base_rule,
                    "lambda": base_lam,
                    "q_factor": 1.0,
                    "target_lambda": target_lam,
                    "target_q_factor": oracle_factor,
                    "target_gap_mEh": base["target_gap_mEh"],
                    "correction_mEh": base["correction_mEh"],
                    "residual_mEh": base["residual_mEh"],
                    "c_tail_equiv": base["c_tail_equiv"],
                    "c_pair_equiv": base["c_pair_equiv"],
                    "note": "Original scalar rule from Step8i.",
                }
            )

            ev = eval_lambda(base, target_lam)
            audit.append(
                {
                    "case": case,
                    "nobs": base["nobs"],
                    "fock_model": fock,
                    "family": feature,
                    "model": f"oracle_residual_{feature}",
                    "lambda": ev["lambda"],
                    "q_factor": oracle_factor,
                    "target_lambda": target_lam,
                    "target_q_factor": oracle_factor,
                    "target_gap_mEh": ev["target_gap_mEh"],
                    "correction_mEh": ev["correction_mEh"],
                    "residual_mEh": ev["residual_mEh"],
                    "c_tail_equiv": ev["lambda"] / base["tail_sqrt"],
                    "c_pair_equiv": ev["lambda"] / base["pair_sqrt"],
                    "note": "Diagnostic upper bound using parent pair-FCI residual.",
                }
            )

            c_loo = scalar_constant(rows, fock, feature, leave_out=case)
            lam_loo = c_loo * (base["tail_sqrt"] if feature == "tail" else base["pair_sqrt"])
            ev = eval_lambda(base, lam_loo)
            audit.append(
                {
                    "case": case,
                    "nobs": base["nobs"],
                    "fock_model": fock,
                    "family": feature,
                    "model": f"leave_one_space_out_{feature}",
                    "lambda": ev["lambda"],
                    "q_factor": lam_loo / base_lam,
                    "target_lambda": target_lam,
                    "target_q_factor": oracle_factor,
                    "target_gap_mEh": ev["target_gap_mEh"],
                    "correction_mEh": ev["correction_mEh"],
                    "residual_mEh": ev["residual_mEh"],
                    "c_tail_equiv": ev["lambda"] / base["tail_sqrt"],
                    "c_pair_equiv": ev["lambda"] / base["pair_sqrt"],
                    "note": "Tests whether one scalar c transfers when this space is withheld.",
                }
            )
    return audit


def summarize(audit: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for model in sorted({r["model"] for r in audit}):
        sub = [r for r in audit if r["model"] == model]
        abs_res = np.array([abs(float(r["residual_mEh"])) for r in sub], dtype=float)
        out[model] = {
            "mean_abs_residual_mEh": float(np.mean(abs_res)),
            "max_abs_residual_mEh": float(np.max(abs_res)),
        }
    return out


def write_outputs(args, scan_rows: List[Dict[str, Any]], audit: List[Dict[str, Any]]) -> None:
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    summary = summarize(audit)
    target_rows = [r for r in scan_rows if r["rule"] == "target_fit"]
    payload = {
        "step": "8j",
        "scan_csv": args.scan_csv,
        "model_summary": summary,
        "target_fit_constants": target_rows,
        "rows": audit,
        "guardrail": "oracle_residual rows use the parent pair-FCI target and are diagnostics only.",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
        writer.writeheader()
        writer.writerows(audit)

    lines = [
        "=" * 112,
        "Step 8j | HEM residual-aware same-spin suppression audit",
        "=" * 112,
        f"scan_csv = {args.scan_csv}",
        "",
        "[Model residual summary]",
    ]
    for model, vals in summary.items():
        lines.append(
            f"{model:<32s} mean_abs={vals['mean_abs_residual_mEh']:.6f} mEh  "
            f"max_abs={vals['max_abs_residual_mEh']:.6f} mEh"
        )
    lines.extend(
        [
            "",
            "[Per-space rows]",
            f"{'case':<18s} {'fock':<8s} {'family':<6s} {'model':<28s} {'q':>9s} {'dE':>12s} {'resid':>12s}",
            "-" * 112,
        ]
    )
    for r in audit:
        lines.append(
            f"{r['case']:<18s} {r['fock_model']:<8s} {r['family']:<6s} {r['model']:<28s} "
            f"{float(r['q_factor']):9.5f} {float(r['correction_mEh']):12.6f} {float(r['residual_mEh']):12.6f}"
        )
    lines.extend(
        [
            "",
            "[Conclusion]",
            "The oracle residual-aware q_factor can, by construction, map each scalar",
            "tail/pair family onto the parent pair-FCI residual.  But leave-one-space-out",
            "scalar constants fail badly for the s-expanded and fully expanded spaces.",
            "This means the next useful rule should not be a single scalar c.  It should",
            "resolve at least s/p channel saturation or pair-channel coupling.",
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
    if args.ensure_scan and not Path(args.scan_csv).exists():
        subprocess.run([args.python, "step8i_scan_hem_open_shell_rule_spaces.py"], check=True)
    scan_rows = read_rows(args.scan_csv)
    audit = build_audit_rows(scan_rows)
    write_outputs(args, scan_rows, audit)


if __name__ == "__main__":
    main()
