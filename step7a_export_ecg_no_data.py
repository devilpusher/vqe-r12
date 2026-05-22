#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7a: Export ECG-NO FCI RDM data for the parent-basis R12 bridge.

This script is intentionally limited to the RDM/data-convention boundary.  It
does not build Psi4 parent integrals and does not call R12 yet.  The goal is to
turn the local `he_2rdm_compare.py` output into a clean, versioned handoff file:

    pair_coeff_ab -> spin-free dm1_obs/dm2_obs

The spin-free convention matches Step 1/4b:

    dm1[p,q] = sum_sigma <a^+_{p sigma} a_{q sigma}>
    dm2[p,q,r,s] = sum_{sigma,tau}<a^+_{p sigma}a^+_{r tau}a_{s tau}a_{q sigma}>
    E2 = 1/2 * einsum("pqrs,pqrs", eri, dm2)

The external ECG-NO generator and its raw outputs remain local-only assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from r12_common import maxabs, rdm_diagnostics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inp",
        default="local_external/he-sin/he_2rdm_compare_matrices.npz",
        help="NPZ produced by local he_2rdm_compare.py.",
    )
    p.add_argument("--method", default="ECG-NO-FCI", help="Method label to export from the NPZ.")
    p.add_argument("--out", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--trace-tol", type=float, default=1e-10)
    return p.parse_args()


def safe_label(s: str) -> str:
    return s.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def load_metrics(x: Any) -> Dict[str, Any]:
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if not isinstance(x, str):
        x = str(x)
    return json.loads(x)


def select_method(data: np.lib.npyio.NpzFile, method: str) -> int:
    methods = [str(x) for x in data["methods"]]
    if method not in methods:
        raise ValueError(f"method {method!r} not found. Available methods: {methods}")
    return methods.index(method)


def rdms_from_ab_pair(Cab: np.ndarray):
    """Rebuild Step1/4b spin-free RDMs from an ordered alpha-beta pair matrix."""
    dm1a = Cab @ Cab.T
    dm1b = Cab.T @ Cab
    dm1 = dm1a + dm1b
    dm2 = np.einsum("pr,qs->pqrs", Cab, Cab, optimize=True)
    dm2 += np.einsum("rp,sq->pqrs", Cab, Cab, optimize=True)
    return dm1, dm2, dm1a, dm1b


def validate_export(Cab: np.ndarray, dm1: np.ndarray, dm2: np.ndarray, gamma_ref: np.ndarray, trace_tol: float) -> Dict[str, Any]:
    diag = rdm_diagnostics(dm1, dm2)
    metrics = {
        "pair_coeff_norm": float(np.linalg.norm(Cab)),
        "pair_coeff_symmetry_error": maxabs(Cab - Cab.T),
        "dm1_vs_gamma_spatial_maxabs": maxabs(dm1 - gamma_ref),
        "dm1_trace_error": float(diag["trace_dm1"] - 2.0),
        "dm2_trace_error": float(diag["trace_dm2"] - 2.0),
        "dm1_asym_error": diag["max_dm1_asym"],
        "dm2_bra_ket_error": diag["max_dm2_bra_ket_error"],
        "natural_occupations": diag["natural_occupations"],
    }
    failures = []
    if abs(metrics["pair_coeff_norm"] - 1.0) > trace_tol:
        failures.append("pair_coeff_norm")
    if abs(metrics["dm1_trace_error"]) > trace_tol:
        failures.append("dm1_trace")
    if abs(metrics["dm2_trace_error"]) > trace_tol:
        failures.append("dm2_trace")
    if metrics["dm1_vs_gamma_spatial_maxabs"] > 1e-10:
        failures.append("dm1_vs_gamma_spatial")
    if metrics["dm2_bra_ket_error"] > 1e-10:
        failures.append("dm2_bra_ket")
    metrics["passed"] = not failures
    metrics["failures"] = failures
    return metrics


def main():
    args = parse_args()
    inp = Path(args.inp)
    if args.out is None:
        args.out = f"step7a_{safe_label(args.method)}_ecg_no_export.npz"
    if args.summary is None:
        args.summary = f"step7a_{safe_label(args.method)}_ecg_no_export_summary.txt"
    if args.json is None:
        args.json = f"step7a_{safe_label(args.method)}_ecg_no_export.json"

    data = np.load(inp, allow_pickle=True)
    idx = select_method(data, args.method)

    methods = [str(x) for x in data["methods"]]
    labels = np.array(data["labels"][idx], dtype=object)
    Cab = np.array(data["pair_coeff_ab"][idx], dtype=float)
    gamma_spatial = np.array(data["gamma_spatial"][idx], dtype=float)
    cumulant_ab = np.array(data["cumulant_ab"][idx], dtype=float)
    metrics_in = load_metrics(data["metrics_json"][idx])
    ao_channels = np.array(data["ao_channels"], dtype=object) if "ao_channels" in data else np.array([], dtype=object)

    dm1, dm2, dm1a, dm1b = rdms_from_ab_pair(Cab)
    checks = validate_export(Cab, dm1, dm2, gamma_spatial, args.trace_tol)

    metadata = {
        "source": "he_2rdm_compare.py output",
        "source_npz": str(inp),
        "method": args.method,
        "available_methods": methods,
        "method_index": idx,
        "nobs": int(Cab.shape[0]),
        "nelec": 2,
        "rdm_convention": {
            "dm1": "dm1[p,q] = sum_sigma <a^+_{p sigma} a_{q sigma}>",
            "dm2": "dm2[p,q,r,s] = sum_{sigma,tau}<a^+_{p sigma}a^+_{r tau}a_{s tau}a_{q sigma}>",
            "energy": "E = einsum(h,dm1) + 0.5*einsum(eri,dm2) + Enuc",
        },
        "notes": [
            "This file exports RDMs only; C_obs and parent-basis integrals are Step 7b responsibilities.",
            "External ECG-NO source files and generated raw NO data are local-only assets.",
        ],
        "input_metrics": metrics_in,
        "checks": checks,
    }

    np.savez(
        args.out,
        method=np.array(args.method),
        labels=labels,
        ao_channels=ao_channels,
        pair_coeff_ab=Cab,
        gamma_spatial_input=gamma_spatial,
        cumulant_ab_input=cumulant_ab,
        dm1_obs=dm1,
        dm2_obs=dm2,
        dm1a_obs=dm1a,
        dm1b_obs=dm1b,
        metrics_json=np.array(json.dumps(metadata, indent=2)),
    )

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    lines = []
    lines.append("=" * 88)
    lines.append("Step 7a | Export ECG-NO RDM data")
    lines.append("=" * 88)
    lines.append(f"input       = {inp}")
    lines.append(f"method      = {args.method} (index {idx})")
    lines.append(f"nobs        = {Cab.shape[0]}")
    lines.append(f"E_input     = {metrics_in.get('energy_Ha', None)} Ha")
    lines.append("")
    lines.append("[Checks]")
    lines.append(f"||Cab||                         = {checks['pair_coeff_norm']:.12f}")
    lines.append(f"Max|Cab-Cab.T|                  = {checks['pair_coeff_symmetry_error']:.3e}")
    lines.append(f"Max|dm1-gamma_spatial_input|    = {checks['dm1_vs_gamma_spatial_maxabs']:.3e}")
    lines.append(f"Tr(dm1)-2                       = {checks['dm1_trace_error']:.3e}")
    lines.append(f"Tr(dm2)-2                       = {checks['dm2_trace_error']:.3e}")
    lines.append(f"Max|dm1-dm1.T|                  = {checks['dm1_asym_error']:.3e}")
    lines.append(f"dm2 bra-ket error               = {checks['dm2_bra_ket_error']:.3e}")
    lines.append("Natural occupations first 8      = " + np.array2string(np.array(checks["natural_occupations"][:8]), precision=10))
    lines.append(f"passed                          = {checks['passed']}")
    if checks["failures"]:
        lines.append(f"failures                        = {checks['failures']}")
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out}")
    lines.append(f"  {args.json}")
    lines.append(f"  {args.summary}")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))

    if not checks["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
