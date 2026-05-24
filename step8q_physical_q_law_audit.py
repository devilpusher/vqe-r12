#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8q: Physical interpretation audit for the HEM q-shape rule.

This is not another parameter scan.  It decomposes the named Step8p HEM
candidate into interpretable factors:

* s_boost: radial s-space cusp/contact resolution;
* p_screen: p-space same-spin exchange-hole screening, with saturation;
* cross_cancel: extra s-p cancellation when both channels are enlarged.

The goal is to make the conservative Step8p rule inspectable before it is
promoted beyond HEM.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict, List

import numpy as np

from step8l_channel_resolved_same_spin_candidates import radial_counts
from step8p_hem_same_spin_candidate import HEM_WORST_CASE_MODEL, HEM_WORST_CASE_PARAMS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--inp", default=None)
    p.add_argument("--prefix", default="step8q_hem_physical_q_law_audit")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def factors(ns: int, np_: int) -> Dict[str, float]:
    a = HEM_WORST_CASE_PARAMS["a"]
    b = HEM_WORST_CASE_PARAMS["b"]
    p_floor = HEM_WORST_CASE_PARAMS["p_floor"]
    c = HEM_WORST_CASE_PARAMS["c"]
    s_boost = (ns / 2.0) ** a
    p_power = (2.0 / np_) ** b
    p_screen = max(p_power, p_floor)
    cross_cancel = (2.0 / ns) ** c if ns > 2 and np_ > 2 else 1.0
    return {
        "s_boost": float(s_boost),
        "p_power": float(p_power),
        "p_screen": float(p_screen),
        "p_floor_active": bool(p_screen == p_floor),
        "cross_cancel": float(cross_cancel),
        "q_product": float(s_boost * p_screen * cross_cancel),
    }


def branch(ns: int, np_: int) -> str:
    if ns > 2 and np_ > 2:
        return "mixed_s_p"
    if ns > 2:
        return "s_growth"
    if np_ > 2:
        return "p_screening"
    return "reference"


def load_rows(inp: str) -> List[Dict[str, Any]]:
    data = json.load(open(inp, encoding="utf-8"))
    return [r for r in data["rows"] if r["model"] == HEM_WORST_CASE_MODEL]


def summarize_by_branch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for br in sorted({r["branch"] for r in rows}):
        sub = [r for r in rows if r["branch"] == br]
        abs_res = np.array([abs(float(r["residual_mEh"])) for r in sub])
        out[br] = {
            "count": len(sub),
            "mean_abs_residual_mEh": float(np.mean(abs_res)),
            "max_abs_residual_mEh": float(np.max(abs_res)),
            "mean_q": float(np.mean([float(r["q_s_saturation"]) for r in sub])),
            "mean_unit_v_mEh_per_lambda": float(np.mean([float(r["unit_v_mEh_per_lambda"]) for r in sub])),
        }
    return out


def build_rows(args) -> List[Dict[str, Any]]:
    inp = args.inp or f"step8p_hem_same_spin_candidate_fitN{args.fitN}.json"
    rows = []
    for r in load_rows(inp):
        ns = int(r["ns_radial"])
        np_ = int(r["np_radial"])
        fac = factors(ns, np_)
        scale = float(r["scale_ss"])
        unit_v = float(r["V_mEh"]) / scale if abs(scale) > 0.0 else None
        row = {
            "case": r["case"],
            "nobs": r["nobs"],
            "ns_radial": ns,
            "np_radial": np_,
            "branch": branch(ns, np_),
            "q_s_saturation": r["q_s_saturation"],
            "tail2": r["tail2"],
            "scale_ss": r["scale_ss"],
            "unit_v_mEh_per_lambda": unit_v,
            "target_gap_mEh": r["target_gap_mEh"],
            "correction_mEh": r["correction_mEh"],
            "residual_mEh": r["residual_mEh"],
            **fac,
        }
        rows.append(row)
    return rows


def main():
    args = parse_args()
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    rows = build_rows(args)
    by_branch = summarize_by_branch(rows)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "step": "8q",
                "fitN": args.fitN,
                "model": HEM_WORST_CASE_MODEL,
                "params": HEM_WORST_CASE_PARAMS,
                "branch_summary": by_branch,
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
        "Step 8q | HEM physical q-law audit",
        "=" * 112,
        f"fitN = {args.fitN}",
        f"model = {HEM_WORST_CASE_MODEL}",
        f"params = {HEM_WORST_CASE_PARAMS}",
        "",
        "[Physical reading]",
        "s_boost grows sublinearly with selected s radial count: more s functions resolve the same-spin cusp/contact channel.",
        "p_screen damps same-spin s-s correction as p space opens exchange-hole relaxation, but it saturates at a floor.",
        "cross_cancel removes the s_boost when both s and p are enlarged, representing observed s-p cancellation.",
        "",
        "[Branch summary]",
    ]
    for br, vals in by_branch.items():
        lines.append(
            f"{br:<12s} count={vals['count']} mean_abs={vals['mean_abs_residual_mEh']:.6f} mEh "
            f"max_abs={vals['max_abs_residual_mEh']:.6f} mEh mean_q={vals['mean_q']:.4f} "
            f"mean_unitV={vals['mean_unit_v_mEh_per_lambda']:.4f}"
        )
    lines.extend(
        [
            "",
            "[Rows]",
            f"{'case':<20s} {'branch':<12s} {'s_boost':>8s} {'p_scr':>8s} {'cross':>8s} {'q':>8s} {'unitV':>10s} {'resid':>10s}",
            "-" * 112,
        ]
    )
    for r in rows:
        lines.append(
            f"{r['case']:<20s} {r['branch']:<12s} {float(r['s_boost']):8.4f} {float(r['p_screen']):8.4f} "
            f"{float(r['cross_cancel']):8.4f} {float(r['q_s_saturation']):8.4f} "
            f"{float(r['unit_v_mEh_per_lambda']):10.4f} {float(r['residual_mEh']):10.6f}"
        )
    lines.extend(
        [
            "",
            "[Interpretation]",
            "The rule can be read as cusp resolution times exchange-hole screening times mixed-channel cancellation.",
            "It is physically constrained but remains an HEM calibration until tested outside this state/space family.",
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
