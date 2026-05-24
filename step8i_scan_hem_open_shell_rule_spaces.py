#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8i: Test HEM open-shell same-spin suppression rules across spaces."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from step8c_hem_triplet_r12_correction import same_spin_pair_fci_target
from step8e_audit_hem_same_spin_failure_source import load_case, scale_to_target_mEh
from step8f_hem_pauli_suppressed_geminal_audit import components, occupation_descriptors, pair_descriptors


@dataclass
class Case:
    name: str
    s_pick: str
    p_pick: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--prefix", default="step8i_hem_triplet_open_shell_space_scan")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--memory", default="4 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def cases() -> List[Case]:
    return [
        Case("sp_s01_p01", "0,1", "0,1"),
        Case("sp_s0123_p01", "0,1,2,3", "0,1"),
        Case("sp_s012345_p01", "0,1,2,3,4,5", "0,1"),
        Case("sp_s01_p0123", "0,1", "0,1,2,3"),
        Case("sp_s01_p012345", "0,1", "0,1,2,3,4,5"),
        Case("sp_s0123_p0123", "0,1,2,3", "0,1,2,3"),
        Case("sp_s012345_p0123", "0,1,2,3,4,5", "0,1,2,3"),
        Case("sp_s012345_p012345", "0,1,2,3,4,5", "0,1,2,3,4,5"),
    ]


def run(cmd: List[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def paths(case: Case, fitn: int) -> Dict[str, str]:
    return {
        "rdm_prefix": f"step8h_hem_triplet_{case.name}",
        "rdm_npz": f"step8h_hem_triplet_{case.name}_rdm_export.npz",
        "rdm_json": f"step8h_hem_triplet_{case.name}_rdm_export.json",
        "rdm_summary": f"step8h_hem_triplet_{case.name}_rdm_export_summary.txt",
        "bridge_npz": f"step8b_hem_triplet_{case.name}_fitN{fitn}_step4b_like.npz",
        "bridge_json": f"step8b_hem_triplet_{case.name}_fitN{fitn}_step4b_like.json",
        "bridge_summary": f"step8b_hem_triplet_{case.name}_fitN{fitn}_step4b_like_summary.txt",
    }


def ensure_case(args, case: Case) -> Dict[str, str]:
    p = paths(case, args.fitN)
    if args.force or not Path(p["rdm_npz"]).exists():
        run(
            [
                args.python,
                "step8h_generate_hem_triplet_rdm_space.py",
                "--s-pick",
                case.s_pick,
                "--p-pick",
                case.p_pick,
                "--prefix",
                p["rdm_prefix"],
                "--out",
                p["rdm_npz"],
                "--json",
                p["rdm_json"],
                "--summary",
                p["rdm_summary"],
            ],
            args.dry_run,
        )
    if args.force or not Path(p["bridge_npz"]).exists():
        run(
            [
                args.python,
                "step8b_build_hem_triplet_step4b_like.py",
                "--rdm-inp",
                p["rdm_npz"],
                "--s-pick",
                case.s_pick,
                "--p-pick",
                case.p_pick,
                "--fitN",
                str(args.fitN),
                "--memory",
                args.memory,
                "--nthreads",
                str(args.nthreads),
                "--r12-only",
                "--out",
                p["bridge_npz"],
                "--json",
                p["bridge_json"],
                "--summary",
                p["bridge_summary"],
            ],
            args.dry_run,
        )
    return p


def add_rule(rows: List[Dict[str, Any]], case_name: str, rule: str, lam: float, fock_model: str, case_data, target, desc, pair) -> None:
    comp = components(case_data, lam / 8.0, -lam / 8.0, fock_model)
    target_mEh = float(target["full_parent_gap_mEh"])
    rows.append(
        {
            "case": case_name,
            "nobs": case_data["nobs"],
            "fock_model": fock_model,
            "rule": rule,
            "lambda": lam,
            "tail_sqrt": desc["sqrt_tail_occupation_sum_after_two"],
            "pair_sqrt": pair["sqrt_residual_pair_weight"],
            "c_tail_equiv": lam / desc["sqrt_tail_occupation_sum_after_two"] if desc["sqrt_tail_occupation_sum_after_two"] > 0 else None,
            "c_pair_equiv": lam / pair["sqrt_residual_pair_weight"] if pair["sqrt_residual_pair_weight"] > 0 else None,
            "E_obs": case_data["E_obs"],
            "E_parent": target["E_full_parent_triplet_pair_fci"],
            "target_gap_mEh": target_mEh,
            "correction_mEh": 1000.0 * comp["correction"],
            "residual_mEh": 1000.0 * comp["correction"] - target_mEh,
            "V_mEh": 1000.0 * comp["V"],
            "B_mEh": 1000.0 * comp["B"],
            "X_mEh": 1000.0 * comp["X"],
            "Delta_mEh": 1000.0 * comp["Delta"],
        }
    )


def evaluate(args, case: Case, p: Dict[str, str]) -> List[Dict[str, Any]]:
    data = load_case(p["bridge_npz"])
    target = same_spin_pair_fci_target(p["bridge_npz"])
    desc = occupation_descriptors(data["dm1"])
    pair = pair_descriptors(p["rdm_npz"])
    rows: List[Dict[str, Any]] = []
    for fock_model in ["spinfree", "alpha"]:
        base = components(data, 1.0 / 8.0, -1.0 / 8.0, fock_model)
        V_mEh = 1000.0 * base["V"]
        Q_mEh = 1000.0 * (base["B"] + base["X"] + base["Delta"])
        target_lam = scale_to_target_mEh(V_mEh, Q_mEh, float(target["full_parent_gap_mEh"]))
        rules = [
            ("tail2", 2.0 * desc["sqrt_tail_occupation_sum_after_two"]),
            ("pair3", 3.0 * pair["sqrt_residual_pair_weight"]),
        ]
        if target_lam is not None:
            rules.append(("target_fit", target_lam))
        for rule, lam in rules:
            add_rule(rows, case.name, rule, float(lam), fock_model, data, target, desc, pair)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    target_rows = [r for r in rows if r["rule"] == "target_fit"]
    for fock_model in sorted({r["fock_model"] for r in target_rows}):
        sub = [r for r in target_rows if r["fock_model"] == fock_model]
        for key in ["c_tail_equiv", "c_pair_equiv"]:
            vals = np.array([float(r[key]) for r in sub], dtype=float)
            out[f"{fock_model}_{key}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return out


def write_outputs(args, rows: List[Dict[str, Any]]) -> None:
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    const = summarize(rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"step": "8i", "fitN": args.fitN, "constant_summary": const, "rows": rows}, f, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [
        "=" * 112,
        "Step 8i | HEM open-shell same-spin rule scan across even s/p spaces",
        "=" * 112,
        f"fitN = {args.fitN}",
        "",
        "[Target-fit constant summary]",
    ]
    for k, v in const.items():
        lines.append(f"{k}: mean/std/min/max = {v['mean']:.6f} / {v['std']:.3e} / {v['min']:.6f} / {v['max']:.6f}")
    lines.extend(
        [
            "",
            "[Rows]",
            f"{'case':<18s} {'nobs':>4s} {'fock':<8s} {'rule':<10s} {'lambda':>10s} {'gap':>12s} {'dE':>12s} {'resid':>12s} {'c_tail':>10s} {'c_pair':>10s}",
            "-" * 112,
        ]
    )
    for r in rows:
        lines.append(
            f"{r['case']:<18s} {int(r['nobs']):4d} {r['fock_model']:<8s} {r['rule']:<10s} "
            f"{float(r['lambda']):10.6f} {float(r['target_gap_mEh']):12.6f} {float(r['correction_mEh']):12.6f} "
            f"{float(r['residual_mEh']):12.6f} {float(r['c_tail_equiv']):10.6f} {float(r['c_pair_equiv']):10.6f}"
        )
    lines.extend(
        [
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
    rows: List[Dict[str, Any]] = []
    for case in cases():
        p = ensure_case(args, case)
        if not args.dry_run:
            rows.extend(evaluate(args, case, p))
    if not args.dry_run:
        write_outputs(args, rows)


if __name__ == "__main__":
    main()
