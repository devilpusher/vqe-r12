#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7n: Fractional natural-occupation projector audit.

This is Route 1 after Step7m.  The audit tests whether a continuous
natural-occupation projector can interpolate between the two hard limits:

    full_active_cabs_only
    strict_fixed_core_lowNO_plus_CABS

without using an external refscale.  The current implementation is an
occupation-only interpolation diagnostic:

    Delta(w) = (1 - lambda) * Delta_fixed_core + lambda * Delta_full_active

where lambda is computed from the occupations of the ECG-NO orbitals outside the
fixed core.  It is not yet the final tensor formula; it is the first sanity
check for whether route 1 has the correct numerical shape.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from step7i_residual_weighted_dual_space import DEFAULT_REFERENCE_ECG14
from step7j_scan_residual_weights import parse_fitn
from step7m_strict_projector_partition_audit import DEFAULT_HE_EXACT_NONREL, case_family


BASE_MODELS = [
    "linear_occ_fraction",
    "sqrt_occ_fraction",
    "square_occ_fraction",
    "logistic_occ_fraction",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step7m-json", default="step7m_strict_projector_partition_audit.json")
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--exact-energy", type=float, default=DEFAULT_HE_EXACT_NONREL)
    p.add_argument("--core-policy", default="strict_fixed_core_lowNO_plus_CABS")
    p.add_argument("--full-policy", default="full_active_cabs_only")
    p.add_argument("--logistic-center", type=float, default=2.5e-4)
    p.add_argument("--logistic-width", type=float, default=7.5e-5)
    p.add_argument("--rational-alphas", default="1e-4,2.5e-4,5e-4,1e-3")
    p.add_argument("--out-json", default="step7n_fractional_occupation_projector_audit.json")
    p.add_argument("--out-csv", default="step7n_fractional_occupation_projector_audit.csv")
    p.add_argument("--stats-csv", default="step7n_fractional_occupation_projector_audit_stats.csv")
    p.add_argument("--summary", default="step7n_fractional_occupation_projector_audit_summary.txt")
    return p.parse_args()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def model_names(args) -> List[str]:
    names = list(BASE_MODELS)
    names.extend(f"rational_occ_alpha_{alpha:g}" for alpha in parse_float_list(args.rational_alphas))
    return names


def load_step7m_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing {path}; run step7m_strict_projector_partition_audit.py first")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["rows"]


def low_occ_sum_from_row(row: Dict[str, Any]) -> float:
    return float(row["missing_trace_dm1"])


def lambda_models(low_occ_sum: float, max_low_occ_sum: float, args) -> Dict[str, float]:
    frac = 0.0 if max_low_occ_sum <= 0.0 else min(1.0, max(0.0, low_occ_sum / max_low_occ_sum))
    z = (low_occ_sum - args.logistic_center) / max(args.logistic_width, 1.0e-15)
    logistic = 1.0 / (1.0 + float(np.exp(-z)))
    out = {
        "linear_occ_fraction": frac,
        "sqrt_occ_fraction": float(np.sqrt(frac)),
        "square_occ_fraction": frac * frac,
        "logistic_occ_fraction": logistic,
    }
    for alpha in parse_float_list(args.rational_alphas):
        out[f"rational_occ_alpha_{alpha:g}"] = low_occ_sum / (low_occ_sum + alpha)
    return out


def pair_limits(rows: List[Dict[str, Any]], full_policy: str, core_policy: str) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
    by_key = {(r["case"], r["policy"]): r for r in rows}
    pairs = []
    for case in sorted({r["case"] for r in rows}):
        full = by_key.get((case, full_policy))
        core = by_key.get((case, core_policy))
        if full is not None and core is not None:
            pairs.append((full, core))
    return pairs


def build_rows(step7m_rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    pairs = pair_limits(step7m_rows, args.full_policy, args.core_policy)
    max_low_occ = max(low_occ_sum_from_row(core) for _, core in pairs)
    out: List[Dict[str, Any]] = []
    for full, core in pairs:
        low_occ = low_occ_sum_from_row(core)
        lambdas = lambda_models(low_occ, max_low_occ, args)
        delta_full = float(full["delta_E_r12"])
        delta_core = float(core["delta_E_r12"])
        for model in model_names(args):
            lam = lambdas[model]
            delta = (1.0 - lam) * delta_core + lam * delta_full
            E_total = float(full["E_obs"]) + delta
            resid_ref = E_total - args.reference_energy
            above_exact = E_total - args.exact_energy
            out.append(
                {
                    "family": full["family"],
                    "case": full["case"],
                    "fitN": parse_fitn(full["path"]),
                    "nqubits": full["nqubits"],
                    "model": model,
                    "lambda_full_active": lam,
                    "low_occ_sum": low_occ,
                    "max_low_occ_sum": max_low_occ,
                    "E_obs": full["E_obs"],
                    "delta_full_active_mEh": 1000.0 * delta_full,
                    "delta_fixed_core_mEh": 1000.0 * delta_core,
                    "delta_fractional_mEh": 1000.0 * delta,
                    "E_total": E_total,
                    "residual_to_reference_mEh": 1000.0 * resid_ref,
                    "abs_residual_to_reference_mEh": 1000.0 * abs(resid_ref),
                    "above_exact_mEh": 1000.0 * above_exact,
                    "overcorrects_reference": E_total < args.reference_energy - 1.0e-10,
                    "below_exact_nonrel": E_total < args.exact_energy,
                    "active_trace_fixed_core": core["active_trace_dm1"],
                    "active_size_fixed_core": core["active_size"],
                    "passive_obs_count_fixed_core": core["passive_obs_count"],
                }
            )
    return out


def build_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["model"])].append(row)
    stats = []
    for (family, model), group in sorted(grouped.items()):
        deltas = np.array([r["delta_fractional_mEh"] for r in group], dtype=float)
        residuals = np.array([r["abs_residual_to_reference_mEh"] for r in group], dtype=float)
        lambdas = np.array([r["lambda_full_active"] for r in group], dtype=float)
        stats.append(
            {
                "family": family,
                "model": model,
                "npoints": len(group),
                "fitNs": ",".join(str(r["fitN"]) for r in sorted(group, key=lambda x: x["fitN"])),
                "delta_mean_mEh": float(np.mean(deltas)),
                "delta_std_mEh": float(np.std(deltas)),
                "delta_min_mEh": float(np.min(deltas)),
                "delta_max_mEh": float(np.max(deltas)),
                "delta_span_mEh": float(np.max(deltas) - np.min(deltas)),
                "abs_residual_mean_mEh": float(np.mean(residuals)),
                "abs_residual_max_mEh": float(np.max(residuals)),
                "lambda_mean": float(np.mean(lambdas)),
                "lambda_std": float(np.std(lambdas)),
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


def write_outputs(args, rows: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> None:
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": args.step7m_json,
                "reference_energy": args.reference_energy,
                "exact_energy": args.exact_energy,
                "models": model_names(args),
                "base_models": BASE_MODELS,
                "rational_alphas": parse_float_list(args.rational_alphas),
                "rows": rows,
                "stats": stats,
                "notes": [
                    "This is a route-1 audit, not the final tensor formula.",
                    "linear/sqrt/square models are scan-normalized diagnostics.",
                    "rational_occ_alpha models use only an absolute NO-occupation scale alpha.",
                    "A successful model should avoid full-active quenching and fixed-core overcorrection.",
                ],
            },
            f,
            indent=2,
        )
    write_csv(args.out_csv, rows)
    write_csv(args.stats_csv, stats)

    lines = []
    lines.append("=" * 132)
    lines.append("Step 7n | Fractional natural-occupation projector audit")
    lines.append("=" * 132)
    lines.append(f"source = {args.step7m_json}")
    lines.append("")
    lines.append("[FitN stability by family/model]")
    lines.append("family                   model                     dE_mean  dE_span  resid_max lambda_mean overRef belowExact")
    for st in stats:
        lines.append(
            f"{st['family']:<24s} {st['model']:<25s} "
            f"{st['delta_mean_mEh']:>8.3f} {st['delta_span_mEh']:>8.3f} "
            f"{st['abs_residual_max_mEh']:>9.3f} {st['lambda_mean']:>11.4f} "
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
    step7m_rows = load_step7m_rows(args.step7m_json)
    rows = build_rows(step7m_rows, args)
    stats = build_stats(rows)
    write_outputs(args, rows, stats)


if __name__ == "__main__":
    main()
