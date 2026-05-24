#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8g: Scan HEM same-spin suppression rules over fitted-Slater fitN.

This step keeps the HEM active space fixed at the externally generated
s01+p01 RDM space and tests whether the Pauli-suppression constants inferred in
Step8f are stable as the fitted Slater Gaussian expansion changes.

The scanned rules are diagnostics, not production formulas:

    lambda_ss = c_tail * sqrt(sum occ beyond the dominant two alpha NOs)
    lambda_ss = c_pair * sqrt(1 - leading alpha-alpha pair weight)

For each fitN, the script also reports the target-fit constants that would
exactly reproduce the dense same-spin parent pair-FCI gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from step8c_hem_triplet_r12_correction import same_spin_pair_fci_target
from step8e_audit_hem_same_spin_failure_source import load_case, scale_to_target_mEh
from step8f_hem_pauli_suppressed_geminal_audit import components, occupation_descriptors, pair_descriptors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitNs", default="3,5,7,9")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--prefix", default="step8g_hem_triplet_suppression_scan")
    p.add_argument("--no-dir", default="local_external/he-meta")
    p.add_argument("--rdm-inp", default="step8a_hem_triplet_ecg_no_fci_rdm_export.npz")
    p.add_argument("--pair-inp", default="step8a_hem_triplet_ecg_no_fci_rdm_export.npz")
    p.add_argument("--channels", default="s,p")
    p.add_argument("--s-pick", default="0,1")
    p.add_argument("--p-pick", default="0,1")
    p.add_argument("--memory", default="4 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run(cmd: List[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def step8b_path(fitn: int) -> Dict[str, str]:
    tag = f"sp_s01_p01_fitN{fitn}"
    return {
        "tag": tag,
        "npz": f"step8b_hem_triplet_{tag}_step4b_like.npz",
        "json": f"step8b_hem_triplet_{tag}_step4b_like.json",
        "summary": f"step8b_hem_triplet_{tag}_step4b_like_summary.txt",
    }


def ensure_step8b(args, fitn: int) -> str:
    paths = step8b_path(fitn)
    if args.force or not Path(paths["npz"]).exists():
        run(
            [
                args.python,
                "step8b_build_hem_triplet_step4b_like.py",
                "--no-dir",
                args.no_dir,
                "--rdm-inp",
                args.rdm_inp,
                "--channels",
                args.channels,
                "--s-pick",
                args.s_pick,
                "--p-pick",
                args.p_pick,
                "--fitN",
                str(fitn),
                "--memory",
                args.memory,
                "--nthreads",
                str(args.nthreads),
                "--r12-only",
                "--out",
                paths["npz"],
                "--json",
                paths["json"],
                "--summary",
                paths["summary"],
            ],
            args.dry_run,
        )
    return paths["npz"]


def add_rule_row(
    rows: List[Dict[str, Any]],
    fitn: int,
    fock_model: str,
    rule: str,
    lam: float,
    case: Dict[str, Any],
    target: Dict[str, Any],
    tail_sqrt: float,
    pair_sqrt: float,
) -> None:
    direct = lam / 8.0
    exchange = -lam / 8.0
    comp = components(case, direct, exchange, fock_model)
    corr_mEh = 1000.0 * comp["correction"]
    target_mEh = float(target["full_parent_gap_mEh"])
    rows.append(
        {
            "fitN": fitn,
            "fock_model": fock_model,
            "rule": rule,
            "lambda": lam,
            "direct": direct,
            "exchange": exchange,
            "tail_sqrt": tail_sqrt,
            "pair_sqrt": pair_sqrt,
            "c_tail_equiv": lam / tail_sqrt if tail_sqrt > 0.0 else None,
            "c_pair_equiv": lam / pair_sqrt if pair_sqrt > 0.0 else None,
            "V_mEh": 1000.0 * comp["V"],
            "B_mEh": 1000.0 * comp["B"],
            "X_mEh": 1000.0 * comp["X"],
            "Delta_mEh": 1000.0 * comp["Delta"],
            "correction_mEh": corr_mEh,
            "target_gap_mEh": target_mEh,
            "residual_mEh": corr_mEh - target_mEh,
            "same_sign_as_gap": bool(np.sign(corr_mEh) == np.sign(target_mEh)),
        }
    )


def scan_point(args, fitn: int, pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    inp = ensure_step8b(args, fitn)
    if args.dry_run:
        return []
    case = load_case(inp)
    target = same_spin_pair_fci_target(inp)
    desc = occupation_descriptors(case["dm1"])
    tail_sqrt = float(desc["sqrt_tail_occupation_sum_after_two"])
    pair_sqrt = float(pair["sqrt_residual_pair_weight"])
    rows: List[Dict[str, Any]] = []
    for fock_model in ["spinfree", "alpha"]:
        base = components(case, 1.0 / 8.0, -1.0 / 8.0, fock_model)
        V_mEh = 1000.0 * base["V"]
        Q_mEh = 1000.0 * (base["B"] + base["X"] + base["Delta"])
        target_lam = scale_to_target_mEh(V_mEh, Q_mEh, float(target["full_parent_gap_mEh"]))
        rules = [
            ("sqrt_tail_occ", tail_sqrt),
            ("2sqrt_tail_occ", 2.0 * tail_sqrt),
            ("sqrt_residual_pair_weight", pair_sqrt),
            ("3sqrt_residual_pair_weight", 3.0 * pair_sqrt),
        ]
        if target_lam is not None:
            rules.append(("target_fit_lambda", target_lam))
        for rule, lam in rules:
            add_rule_row(rows, fitn, fock_model, rule, float(lam), case, target, tail_sqrt, pair_sqrt)
    return rows


def summarize_constants(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for fock_model in sorted({r["fock_model"] for r in rows}):
        target_rows = [r for r in rows if r["fock_model"] == fock_model and r["rule"] == "target_fit_lambda"]
        if not target_rows:
            continue
        c_tail = np.array([float(r["c_tail_equiv"]) for r in target_rows], dtype=float)
        c_pair = np.array([float(r["c_pair_equiv"]) for r in target_rows], dtype=float)
        out[fock_model] = {
            "c_tail_mean": float(np.mean(c_tail)),
            "c_tail_std": float(np.std(c_tail)),
            "c_tail_min": float(np.min(c_tail)),
            "c_tail_max": float(np.max(c_tail)),
            "c_pair_mean": float(np.mean(c_pair)),
            "c_pair_std": float(np.std(c_pair)),
            "c_pair_min": float(np.min(c_pair)),
            "c_pair_max": float(np.max(c_pair)),
        }
    return out


def write_outputs(args, rows: List[Dict[str, Any]], constants: Dict[str, Any], pair: Dict[str, Any]) -> None:
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    payload = {
        "step": "8g",
        "fitNs": parse_int_list(args.fitNs),
        "pair_descriptors": pair,
        "constant_summary": constants,
        "rows": rows,
        "guardrail": "target_fit_lambda and fitted c constants are diagnostics, not final formulas.",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "=" * 108,
        "Step 8g | HEM same-spin suppression rule scan",
        "=" * 108,
        f"fitNs             = {args.fitNs}",
        f"space             = channels={args.channels}, s={args.s_pick}, p={args.p_pick}",
        f"leading pair      = {pair['leading_pair']['label_p']} ^ {pair['leading_pair']['label_q']} weight={pair['leading_pair']['weight']:.12e}",
        f"sqrt residual pair= {pair['sqrt_residual_pair_weight']:.12e}",
        "",
        "[Target-fit constant stability]",
    ]
    for fock_model, vals in constants.items():
        lines.extend(
            [
                f"{fock_model}:",
                f"  c_tail mean/std/min/max = {vals['c_tail_mean']:.8f} / {vals['c_tail_std']:.3e} / {vals['c_tail_min']:.8f} / {vals['c_tail_max']:.8f}",
                f"  c_pair mean/std/min/max = {vals['c_pair_mean']:.8f} / {vals['c_pair_std']:.3e} / {vals['c_pair_min']:.8f} / {vals['c_pair_max']:.8f}",
            ]
        )
    lines.extend(
        [
            "",
            "[Rows]",
            f"{'fitN':>4s} {'fock':<8s} {'rule':<28s} {'lambda':>10s} {'dE/mEh':>12s} {'resid/mEh':>12s} {'c_tail':>10s} {'c_pair':>10s}",
            "-" * 108,
        ]
    )
    for r in rows:
        lines.append(
            f"{int(r['fitN']):4d} {r['fock_model']:<8s} {r['rule']:<28s} "
            f"{float(r['lambda']):10.6f} {float(r['correction_mEh']):12.6f} {float(r['residual_mEh']):12.6f} "
            f"{float(r['c_tail_equiv']):10.6f} {float(r['c_pair_equiv']):10.6f}"
        )
    lines.extend(
        [
            "",
            "[Interpretation]",
            "For the current fixed HEM s01+p01 RDM, target-fit c constants are expected",
            "to vary only through the fitted-Slater f12 tensor.  Strong stability across",
            "fitN supports a state-defined suppression rule; large drift would mean the",
            "rule is absorbing fit error rather than physical same-spin Pauli suppression.",
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
    fitns = parse_int_list(args.fitNs)
    pair = pair_descriptors(args.pair_inp)
    rows: List[Dict[str, Any]] = []
    for fitn in fitns:
        rows.extend(scan_point(args, fitn, pair))
    if args.dry_run:
        return
    constants = summarize_constants(rows)
    write_outputs(args, rows, constants, pair)


if __name__ == "__main__":
    main()
