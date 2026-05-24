#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8o: Scan internal HEM same-spin q-shape rules.

This continues Step8n after the 4-to-6 radial selected-space check exposed
that a plain count-balance rule is fitN-stable but too rigid.  The scanned
family remains fully internal:

    lambda_ss = 2 * sqrt(tail_occ) * q_s_saturation

    q = (n_s / 2)^a * max((2 / n_p)^b, p_floor) * cross

where cross is applied only when both s and p spaces are enlarged:

    cross = (2 / n_s)^c

Oracle/reference rows are diagnostics only and are not used by the formula.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from step8c_hem_triplet_r12_correction import same_spin_pair_fci_target
from step8e_audit_hem_same_spin_failure_source import load_case, scale_to_target_mEh
from step8f_hem_pauli_suppressed_geminal_audit import occupation_descriptors, pair_descriptors
from step8l_channel_resolved_same_spin_candidates import cases, compute, paths, radial_counts
from step8n_scan_qs_saturation_rules import q_count_balance_cross


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--a-grid", default="0.70,0.75,0.80,0.85")
    p.add_argument("--b-grid", default="0.35,0.45,0.50")
    p.add_argument("--p-floor-grid", default="0.74,0.76,0.78")
    p.add_argument("--c-grid", default="1.00,1.25,1.50")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--prefix", default="step8o_hem_internal_q_shape_rules")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def q_shape(labels: List[str], a: float, b: float, p_floor: float, c: float) -> float:
    counts = radial_counts(labels)
    ns = max(1, counts["s"])
    np_ = max(1, counts["p"])
    q = (ns / 2.0) ** a
    q *= max((2.0 / np_) ** b, p_floor)
    if ns > 2 and np_ > 2:
        q *= (2.0 / ns) ** c
    return float(q)


def model_name(a: float, b: float, p_floor: float, c: float) -> str:
    return f"q_shape_a{a:.2f}_b{b:.2f}_pf{p_floor:.2f}_c{c:.2f}"


def iter_params(args) -> Iterable[Dict[str, float]]:
    for a in parse_float_list(args.a_grid):
        for b in parse_float_list(args.b_grid):
            for p_floor in parse_float_list(args.p_floor_grid):
                for c in parse_float_list(args.c_grid):
                    yield {"a": a, "b": b, "p_floor": p_floor, "c": c}


def scaled_components(unit: Dict[str, float], scale: float) -> Dict[str, float]:
    return {
        "V": scale * unit["V"],
        "B": scale * scale * unit["B"],
        "X": scale * scale * unit["X"],
        "Delta": scale * scale * unit["Delta"],
    }


def add_scaled_row(
    rows: List[Dict[str, Any]],
    case: str,
    item: Dict[str, Any],
    model: str,
    q: float,
    params: Dict[str, float] | None,
    unit: Dict[str, float],
) -> None:
    tail2 = float(item["tail2"])
    scale = tail2 * q
    comp = scaled_components(unit, scale)
    corr_mEh = 1000.0 * (comp["V"] + comp["B"] + comp["X"] + comp["Delta"])
    target_mEh = float(item["target"]["full_parent_gap_mEh"])
    counts = radial_counts(item["labels"])
    row = {
        "case": case,
        "nobs": item["data"]["nobs"],
        "fock_model": "alpha",
        "model": model,
        "q_s_saturation": float(q),
        "tail2": tail2,
        "scale_ss": float(scale),
        "ns_radial": int(counts["s"]),
        "np_radial": int(counts["p"]),
        "target_gap_mEh": target_mEh,
        "correction_mEh": corr_mEh,
        "residual_mEh": corr_mEh - target_mEh,
        "V_mEh": 1000.0 * comp["V"],
        "B_mEh": 1000.0 * comp["B"],
        "X_mEh": 1000.0 * comp["X"],
        "Delta_mEh": 1000.0 * comp["Delta"],
        "tail_sqrt": item["desc"]["sqrt_tail_occupation_sum_after_two"],
        "pair_sqrt": item["pair"]["sqrt_residual_pair_weight"],
        "a": None,
        "b": None,
        "p_floor": None,
        "c": None,
    }
    if params is not None:
        row.update(params)
    rows.append(row)


def load_items(fitn: int) -> Dict[str, Dict[str, Any]]:
    loaded = {}
    for case in cases():
        p = paths(case, fitn)
        if not Path(p["rdm_npz"]).exists() or not Path(p["bridge_npz"]).exists():
            raise FileNotFoundError(f"Missing prerequisite files for {case}; run Step8i for fitN={fitn}.")
        data = load_case(p["bridge_npz"])
        labels = [str(x) for x in np.load(p["bridge_npz"], allow_pickle=True)["labels"]]
        desc = occupation_descriptors(data["dm1"])
        loaded[case] = {
            "paths": p,
            "data": data,
            "labels": labels,
            "target": same_spin_pair_fci_target(p["bridge_npz"]),
            "desc": desc,
            "pair": pair_descriptors(p["rdm_npz"]),
            "tail2": 2.0 * desc["sqrt_tail_occupation_sum_after_two"],
            "unit": compute(data, labels, {"s-s": 1.0}, "alpha"),
        }
    return loaded


def build_rows(args) -> List[Dict[str, Any]]:
    loaded = load_items(args.fitN)
    params_grid = list(iter_params(args))
    rows: List[Dict[str, Any]] = []
    for case, item in loaded.items():
        target_mEh = float(item["target"]["full_parent_gap_mEh"])
        unit = item["unit"]
        v_mEh = 1000.0 * unit["V"]
        q_mEh = 1000.0 * (unit["B"] + unit["X"] + unit["Delta"])
        lam_oracle = scale_to_target_mEh(v_mEh, q_mEh, target_mEh)
        if lam_oracle is not None:
            add_scaled_row(rows, case, item, "q_oracle_ss_only", lam_oracle / item["tail2"], None, unit)
        add_scaled_row(rows, case, item, "q_count_balance_cross_ss_only", q_count_balance_cross(item["labels"]), None, unit)
        for params in params_grid:
            q = q_shape(item["labels"], params["a"], params["b"], params["p_floor"], params["c"])
            add_scaled_row(rows, case, item, model_name(**params), q, params, unit)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        abs_res = np.array([abs(float(r["residual_mEh"])) for r in sub])
        out[model] = {
            "mean_abs_residual_mEh": float(np.mean(abs_res)),
            "max_abs_residual_mEh": float(np.max(abs_res)),
            "rms_residual_mEh": float(np.sqrt(np.mean(abs_res * abs_res))),
        }
    return out


def best_models(summary: Dict[str, Any], top: int) -> List[tuple[str, Dict[str, Any]]]:
    candidates = [(k, v) for k, v in summary.items() if not k.startswith("q_oracle")]
    return sorted(candidates, key=lambda kv: (kv[1]["max_abs_residual_mEh"], kv[1]["mean_abs_residual_mEh"]))[:top]


def main():
    args = parse_args()
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"

    rows = build_rows(args)
    summary = summarize(rows)
    best = best_models(summary, args.top)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"step": "8o", "fitN": args.fitN, "model_summary": summary, "best_models": best, "rows": rows}, f, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=" * 118,
        "Step 8o | HEM internal q-shape scan for lambda_ss = 2*sqrt(tail_occ)*q",
        "=" * 118,
        f"fitN = {args.fitN}",
        f"a_grid = {args.a_grid}",
        f"b_grid = {args.b_grid}",
        f"p_floor_grid = {args.p_floor_grid}",
        f"c_grid = {args.c_grid}",
        "",
        "[Best non-oracle models]",
    ]
    for model, vals in best:
        lines.append(
            f"{model:<42s} mean_abs={vals['mean_abs_residual_mEh']:.6f} mEh  "
            f"max_abs={vals['max_abs_residual_mEh']:.6f} mEh  rms={vals['rms_residual_mEh']:.6f} mEh"
        )
    lines.extend(
        [
            "",
            "[Baseline / oracle]",
        ]
    )
    for model in ["q_count_balance_cross_ss_only", "q_oracle_ss_only"]:
        vals = summary.get(model)
        if vals:
            lines.append(
                f"{model:<42s} mean_abs={vals['mean_abs_residual_mEh']:.6f} mEh  "
                f"max_abs={vals['max_abs_residual_mEh']:.6f} mEh  rms={vals['rms_residual_mEh']:.6f} mEh"
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


if __name__ == "__main__":
    main()
