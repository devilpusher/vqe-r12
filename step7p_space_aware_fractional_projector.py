#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7p: Space-aware tensor fractional NO projector audit for He singlet.

Step7o showed that a local q(n) can restore the 18q correction, but the same
local rule may overcorrect when more low-occupation ECG-NO orbitals are present.
This script keeps the tensor-level weighted contraction but adds a space-aware
cap on the total OBS fractional complement strength:

    q_obs_raw = n / (n + alpha)
    q_obs     = q_obs_raw * min(1, q_obs_cap / sum(q_obs_raw))
    q_cabs    = 1

The cap is a He-singlet self-consistency audit parameter, not a final universal
formula.  It tests whether controlling the total fractional OBS complement is
the missing ingredient before moving to triplet He and larger atoms.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from step7h_dual_space_projector_prototype import fixed_core_indices, load_labels, parse_core_tokens
from step7i_residual_weighted_dual_space import DEFAULT_REFERENCE_ECG14
from step7j_scan_residual_weights import parse_fitn
from step7m_strict_projector_partition_audit import DEFAULT_HE_EXACT_NONREL, case_family
from step7o_tensor_fractional_projector_audit import parse_float_list, sorted_inputs, tensor_fractional_components
from step7g_audit_ecg_no_r12_subterms import load_case


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=None)
    p.add_argument("--glob", default="step7c_*fitN*_r12only_step4b_like.npz")
    p.add_argument("--core", default="s0,s1,s2,p0")
    p.add_argument("--alphas", default="1e-4,2.5e-4,5e-4")
    p.add_argument("--q-caps", default="0.5,0.75,1.0,1.25,1.5")
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--exact-energy", type=float, default=DEFAULT_HE_EXACT_NONREL)
    p.add_argument("--out-json", default="step7p_space_aware_fractional_projector.json")
    p.add_argument("--out-csv", default="step7p_space_aware_fractional_projector.csv")
    p.add_argument("--stats-csv", default="step7p_space_aware_fractional_projector_stats.csv")
    p.add_argument("--summary", default="step7p_space_aware_fractional_projector_summary.txt")
    return p.parse_args()


def q_space_aware(occ: np.ndarray, nobs: int, nri: int, core: List[int], alpha: float, cap: float) -> tuple[np.ndarray, float, float]:
    q = np.ones(nri, dtype=float)
    q[:nobs] = 0.0
    core_set = set(core)
    noncore = [i for i in range(nobs) if i not in core_set]
    raw = occ[noncore] / (occ[noncore] + alpha)
    raw_sum = float(np.sum(raw))
    scale = 1.0 if raw_sum <= cap or raw_sum <= 0.0 else cap / raw_sum
    q[noncore] = np.clip(raw * scale, 0.0, 1.0)
    q[core] = 0.0
    return q, raw_sum, scale


def audit_case(path: str, args) -> List[Dict[str, Any]]:
    case = load_case(path)
    nobs = case["nobs"]
    nri = case["nri"]
    labels = load_labels(path, nobs)
    core = fixed_core_indices(labels, parse_core_tokens(args.core))
    if not core:
        core = list(range(min(6, nobs)))
    occ = np.array(case["natural_occupations"], dtype=float)
    rows = []
    for alpha in parse_float_list(args.alphas):
        for cap in parse_float_list(args.q_caps):
            q, raw_sum, scale = q_space_aware(occ, nobs, nri, core, alpha, cap)
            comp = tensor_fractional_components(case["g_phys"], case["r_phys"], case["fock"], case["dm1_obs"], case["dm2_obs"], nobs, nri, q)
            delta = comp["correction"]
            E_total = case["E_obs"] + delta
            rows.append(
                {
                    "family": case_family(case["label"]),
                    "case": case["label"],
                    "path": path,
                    "fitN": parse_fitn(path),
                    "nobs": nobs,
                    "nqubits": 2 * nobs,
                    "alpha": alpha,
                    "q_cap": cap,
                    "q_obs_raw_sum": raw_sum,
                    "q_obs_scale": scale,
                    "q_obs_sum": float(np.sum(q[:nobs])),
                    "q_obs_max": float(np.max(q[:nobs])) if nobs else 0.0,
                    "E_obs": case["E_obs"],
                    "delta_E_r12": delta,
                    "delta_E_r12_mEh": 1000.0 * delta,
                    "E_total": E_total,
                    "residual_to_reference_mEh": 1000.0 * (E_total - args.reference_energy),
                    "abs_residual_to_reference_mEh": 1000.0 * abs(E_total - args.reference_energy),
                    "above_exact_mEh": 1000.0 * (E_total - args.exact_energy),
                    "overcorrects_reference": E_total < args.reference_energy - 1.0e-10,
                    "below_exact_nonrel": E_total < args.exact_energy,
                    "V_mEh": 1000.0 * comp["V"],
                    "B_mEh": 1000.0 * comp["B"],
                    "X_mEh": 1000.0 * comp["X"],
                    "Delta_mEh": 1000.0 * comp["Delta"],
                }
            )
    return rows


def build_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, float, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["alpha"], row["q_cap"])].append(row)
    stats = []
    for (family, alpha, cap), group in sorted(grouped.items()):
        deltas = np.array([r["delta_E_r12_mEh"] for r in group], dtype=float)
        residuals = np.array([r["abs_residual_to_reference_mEh"] for r in group], dtype=float)
        stats.append(
            {
                "family": family,
                "alpha": alpha,
                "q_cap": cap,
                "npoints": len(group),
                "fitNs": ",".join(str(r["fitN"]) for r in sorted(group, key=lambda x: x["fitN"])),
                "delta_mean_mEh": float(np.mean(deltas)),
                "delta_std_mEh": float(np.std(deltas)),
                "delta_min_mEh": float(np.min(deltas)),
                "delta_max_mEh": float(np.max(deltas)),
                "delta_span_mEh": float(np.max(deltas) - np.min(deltas)),
                "abs_residual_mean_mEh": float(np.mean(residuals)),
                "abs_residual_max_mEh": float(np.max(residuals)),
                "q_obs_sum_mean": float(np.mean([r["q_obs_sum"] for r in group])),
                "q_obs_scale_mean": float(np.mean([r["q_obs_scale"] for r in group])),
                "any_overcorrects_reference": any(r["overcorrects_reference"] for r in group),
                "any_below_exact_nonrel": any(r["below_exact_nonrel"] for r in group),
            }
        )
    return stats


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args, inputs: List[str], rows: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> None:
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": inputs,
                "reference_energy": args.reference_energy,
                "exact_energy": args.exact_energy,
                "core": args.core,
                "alphas": parse_float_list(args.alphas),
                "q_caps": parse_float_list(args.q_caps),
                "rows": rows,
                "stats": stats,
                "notes": [
                    "This is a He-singlet space-aware cap audit.",
                    "q_obs_cap controls the total fractional OBS complement strength.",
                    "The cap is not a final cross-system formula until a state-aware occupancy-domain rule is defined.",
                ],
            },
            f,
            indent=2,
        )
    write_csv(args.out_csv, rows)
    write_csv(args.stats_csv, stats)

    lines = []
    lines.append("=" * 132)
    lines.append("Step 7p | He-singlet space-aware tensor fractional projector audit")
    lines.append("=" * 132)
    lines.append("family                   alpha     q_cap  dE_mean  dE_span  resid_max qsum_mean overRef belowExact")
    for st in stats:
        lines.append(
            f"{st['family']:<24s} {st['alpha']:>8.1e} {st['q_cap']:>7.2f} "
            f"{st['delta_mean_mEh']:>8.3f} {st['delta_span_mEh']:>8.3f} "
            f"{st['abs_residual_max_mEh']:>9.3f} {st['q_obs_sum_mean']:>9.3f} "
            f"{str(st['any_overcorrects_reference']):>7s} {str(st['any_below_exact_nonrel']):>10s}"
        )
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out_json}")
    lines.append(f"  {args.out_csv}")
    lines.append(f"  {args.stats_csv}")
    lines.append(f"  {args.summary}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    inputs = sorted_inputs(args)
    if not inputs:
        raise SystemExit(f"No inputs matched {args.glob!r}")
    rows = []
    for path in inputs:
        rows.extend(audit_case(path, args))
    stats = build_stats(rows)
    write_outputs(args, inputs, rows, stats)


if __name__ == "__main__":
    main()
