#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8p: Named HEM same-spin R12 candidate.

This freezes the conservative worst-case Step8o rule as a named HEM candidate:

    lambda_ss = 2 * sqrt(tail_occ) * q_hem_worst_case

    q_hem_worst_case = (n_s / 2)^0.80 * max((2 / n_p)^0.35, 0.76)

    if n_s > 2 and n_p > 2:
        q_hem_worst_case *= (2 / n_s)^1.00

The rule is still HEM-specific and should be treated as a production candidate,
not a universal R12 formula.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict, List

import numpy as np

from step8e_audit_hem_same_spin_failure_source import scale_to_target_mEh
from step8n_scan_qs_saturation_rules import q_count_balance_cross
from step8o_scan_internal_q_shape_rules import add_scaled_row, load_items, q_shape, summarize


HEM_WORST_CASE_PARAMS = {"a": 0.80, "b": 0.35, "p_floor": 0.76, "c": 1.00}
HEM_WORST_CASE_MODEL = "hem_q_shape_worst_case_ss_only"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--prefix", default="step8p_hem_same_spin_candidate")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def add_oracle_row(rows: List[Dict[str, Any]], case: str, item: Dict[str, Any]) -> None:
    target_mEh = float(item["target"]["full_parent_gap_mEh"])
    unit = item["unit"]
    v_mEh = 1000.0 * unit["V"]
    q_mEh = 1000.0 * (unit["B"] + unit["X"] + unit["Delta"])
    lam = scale_to_target_mEh(v_mEh, q_mEh, target_mEh)
    if lam is not None:
        add_scaled_row(rows, case, item, "q_oracle_ss_only", lam / item["tail2"], None, unit)


def build_rows(fitn: int) -> List[Dict[str, Any]]:
    loaded = load_items(fitn)
    rows: List[Dict[str, Any]] = []
    for case, item in loaded.items():
        unit = item["unit"]
        q_candidate = q_shape(item["labels"], **HEM_WORST_CASE_PARAMS)
        add_scaled_row(rows, case, item, HEM_WORST_CASE_MODEL, q_candidate, HEM_WORST_CASE_PARAMS, unit)
        add_scaled_row(rows, case, item, "q_count_balance_cross_ss_only", q_count_balance_cross(item["labels"]), None, unit)
        add_oracle_row(rows, case, item)
    return rows


def main():
    args = parse_args()
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"

    rows = build_rows(args.fitN)
    summary = summarize(rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "step": "8p",
                "fitN": args.fitN,
                "hem_worst_case_model": HEM_WORST_CASE_MODEL,
                "hem_worst_case_params": HEM_WORST_CASE_PARAMS,
                "model_summary": summary,
                "rows": rows,
            },
            f,
            indent=2,
        )
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=" * 112,
        "Step 8p | HEM same-spin R12 named candidate",
        "=" * 112,
        f"fitN = {args.fitN}",
        f"model = {HEM_WORST_CASE_MODEL}",
        f"params = {HEM_WORST_CASE_PARAMS}",
        "",
        "[Model residual summary]",
    ]
    for model in [HEM_WORST_CASE_MODEL, "q_count_balance_cross_ss_only", "q_oracle_ss_only"]:
        vals = summary.get(model)
        if vals:
            lines.append(
                f"{model:<36s} mean_abs={vals['mean_abs_residual_mEh']:.6f} mEh  "
                f"max_abs={vals['max_abs_residual_mEh']:.6f} mEh  rms={vals['rms_residual_mEh']:.6f} mEh"
            )
    lines.extend(
        [
            "",
            "[Rows]",
            f"{'case':<20s} {'model':<34s} {'q_sat':>8s} {'gap':>11s} {'dE':>11s} {'resid':>11s}",
            "-" * 112,
        ]
    )
    for r in rows:
        if r["model"] in {HEM_WORST_CASE_MODEL, "q_count_balance_cross_ss_only"}:
            lines.append(
                f"{r['case']:<20s} {r['model']:<34s} {float(r['q_s_saturation']):8.4f} "
                f"{float(r['target_gap_mEh']):11.6f} {float(r['correction_mEh']):11.6f} {float(r['residual_mEh']):11.6f}"
            )
    lines.extend(
        [
            "",
            "[Interpretation]",
            "This is the conservative worst-case Step8o HEM candidate.",
            "The q_oracle row is retained only as a diagnostic and is not a production formula.",
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
