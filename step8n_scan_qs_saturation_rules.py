#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8n: Scan HEM same-spin q_s_saturation rules.

We audit rules of the form

    lambda_ss = 2 * sqrt(tail_occ) * q_s_saturation

for the metastable He triplet same-spin pair.  The goal is to find a compact
space-aware saturation factor before promoting any rule into production R12.
Oracle/reference rows are diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from step8c_hem_triplet_r12_correction import same_spin_pair_fci_target
from step8e_audit_hem_same_spin_failure_source import load_case
from step8f_hem_pauli_suppressed_geminal_audit import occupation_descriptors, pair_descriptors
from step8l_channel_resolved_same_spin_candidates import (
    add_row,
    compute,
    cases,
    paths,
    radial_counts,
    root_for_family,
    vss_unit,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--prefix", default="step8n_hem_qs_saturation_rules")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def q_count_balance(labels: List[str]) -> float:
    """Simple active radial count balance: s expansion boosts, p expansion damps."""
    counts = radial_counts(labels)
    ns = max(1, counts["s"])
    np_ = max(1, counts["p"])
    return float(np.sqrt(ns / np_))


def q_count_balance_cross(labels: List[str]) -> float:
    """Count balance plus extra damping when both s and p spaces are enlarged."""
    counts = radial_counts(labels)
    ns = max(1, counts["s"])
    np_ = max(1, counts["p"])
    q = q_count_balance(labels)
    if ns > 2 and np_ > 2:
        q *= 2.0 / ns
    return float(q)


def q_vsat_reference(labels: List[str], ref_vss: float, this_vss: float) -> float:
    """Reference audit: reproduce the successful Step8l V-saturation family."""
    counts = radial_counts(labels)
    ns = max(1, counts["s"])
    np_ = max(1, counts["p"])
    q = np.sqrt(abs(ref_vss) / max(abs(this_vss), 1e-30))
    if np_ > 2:
        q *= np.sqrt(2.0 / np_)
    if np_ > 2 and ns > 2:
        q *= 2.0 / ns
    return float(q)


def q_vsat_self_count(labels: List[str], ref_vss: float, this_vss: float) -> float:
    """Blend V-saturation with count balance, reducing dependence on one case."""
    return float(np.sqrt(max(q_vsat_reference(labels, ref_vss, this_vss), 0.0) * max(q_count_balance_cross(labels), 0.0)))


def build_rows(fitn: int) -> List[Dict[str, Any]]:
    loaded = {}
    for case in cases():
        p = paths(case, fitn)
        if not Path(p["rdm_npz"]).exists() or not Path(p["bridge_npz"]).exists():
            raise FileNotFoundError(f"Missing prerequisite files for {case}; run make step8i first.")
        data = load_case(p["bridge_npz"])
        labels = [str(x) for x in np.load(p["bridge_npz"], allow_pickle=True)["labels"]]
        loaded[case] = {
            "paths": p,
            "data": data,
            "labels": labels,
            "target": same_spin_pair_fci_target(p["bridge_npz"]),
            "desc": occupation_descriptors(data["dm1"]),
            "pair": pair_descriptors(p["rdm_npz"]),
            "vss_unit": vss_unit(data, labels),
        }

    ref_vss = loaded["sp_s01_p01"]["vss_unit"]
    rows: List[Dict[str, Any]] = []
    for case, item in loaded.items():
        data = item["data"]
        labels = item["labels"]
        target = item["target"]
        desc = item["desc"]
        pair = item["pair"]
        tail2 = 2.0 * desc["sqrt_tail_occupation_sum_after_two"]
        target_mEh = float(target["full_parent_gap_mEh"])
        q_oracle = root_for_family(data, labels, {"s-s": 1.0}, "alpha", target_mEh) / tail2
        q_rules = [
            ("q_count_balance_ss_only", q_count_balance(labels)),
            ("q_count_balance_cross_ss_only", q_count_balance_cross(labels)),
            ("q_vsat_reference_ss_only", q_vsat_reference(labels, ref_vss, item["vss_unit"])),
            ("q_vsat_self_count_ss_only", q_vsat_self_count(labels, ref_vss, item["vss_unit"])),
            ("q_oracle_ss_only", q_oracle),
        ]
        for name, q in q_rules:
            add_row(rows, case, data, labels, target, desc, pair, name, {"s-s": tail2 * q}, "alpha")
            rows[-1]["q_s_saturation"] = float(q)
            rows[-1]["tail2"] = float(tail2)
            rows[-1]["ns_radial"] = radial_counts(labels)["s"]
            rows[-1]["np_radial"] = radial_counts(labels)["p"]
        # The same q applied also to s-p is retained only as a cancellation audit.
        q = q_count_balance_cross(labels)
        add_row(rows, case, data, labels, target, desc, pair, "q_count_balance_cross_ss_sp", {"s-s": tail2 * q, "p-s": tail2 * q}, "alpha")
        rows[-1]["q_s_saturation"] = float(q)
        rows[-1]["tail2"] = float(tail2)
        rows[-1]["ns_radial"] = radial_counts(labels)["s"]
        rows[-1]["np_radial"] = radial_counts(labels)["p"]
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        abs_res = np.array([abs(float(r["residual_mEh"])) for r in sub])
        out[model] = {
            "mean_abs_residual_mEh": float(np.mean(abs_res)),
            "max_abs_residual_mEh": float(np.max(abs_res)),
        }
    return out


def main():
    args = parse_args()
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    rows = build_rows(args.fitN)
    summary = summarize(rows)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"step": "8n", "fitN": args.fitN, "model_summary": summary, "rows": rows}, f, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=" * 112,
        "Step 8n | HEM lambda_ss = 2*sqrt(tail_occ)*q_s_saturation scan",
        "=" * 112,
        f"fitN = {args.fitN}",
        "",
        "[Model residual summary]",
    ]
    for model, vals in summary.items():
        lines.append(f"{model:<36s} mean_abs={vals['mean_abs_residual_mEh']:.6f} mEh  max_abs={vals['max_abs_residual_mEh']:.6f} mEh")
    lines.extend(
        [
            "",
            "[Rows]",
            f"{'case':<18s} {'model':<34s} {'q_sat':>8s} {'tail2':>8s} {'ss':>9s} {'gap':>11s} {'dE':>11s} {'resid':>11s}",
            "-" * 112,
        ]
    )
    for r in rows:
        lines.append(
            f"{r['case']:<18s} {r['model']:<34s} {float(r['q_s_saturation']):8.4f} {float(r['tail2']):8.5f} "
            f"{float(r['scale_ss']):9.5f} {float(r['target_gap_mEh']):11.6f} {float(r['correction_mEh']):11.6f} {float(r['residual_mEh']):11.6f}"
        )
    lines.extend(
        [
            "",
            "[Interpretation]",
            "q_count_balance is fully internal and uses only active radial counts.",
            "q_vsat_reference is a reference-space audit, not a final production rule.",
            "q_oracle uses the parent target and is only the ideal scale diagnostic.",
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


if __name__ == "__main__":
    main()
