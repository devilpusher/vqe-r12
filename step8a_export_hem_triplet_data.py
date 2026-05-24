#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8a: Export metastable He triplet ECG-NO RDM data.

The local HEM generator is an external asset.  This script only consumes its
`hem_2rdm_compare_matrices.npz` output and converts the stored alpha-alpha pair
vector into the spin-free RDM convention used by the parent-basis R12 pipeline:

    E = einsum(h, dm1) + 0.5 * einsum(eri, dm2) + Enuc

For the HEM file the pair vector is `pair_coeff_upper[p,q]` for p<q in the
Ms=1 alpha-alpha sector.  The ordered alpha-pair tensor from the local script is
transposed as D[p,r,q,s] so that `Tr(dm2)=einsum("pprr", dm2)=2` and the energy
contraction matches the Step 1/4b spin-free convention.
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
    p.add_argument("--inp", default="local_external/he-meta/hem_2rdm_compare_matrices.npz")
    p.add_argument("--method", default="ECG-NO-FCI")
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--trace-tol", type=float, default=1e-10)
    return p.parse_args()


def safe_label(s: str) -> str:
    return s.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def load_metrics(x: Any) -> Dict[str, Any]:
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return json.loads(str(x))


def select_method(data: np.lib.npyio.NpzFile, method: str) -> int:
    methods = [str(x) for x in data["methods"]]
    if method not in methods:
        raise ValueError(f"method {method!r} not found. Available methods: {methods}")
    return methods.index(method)


def upper_to_antisym(B_upper: np.ndarray) -> np.ndarray:
    n = B_upper.shape[0]
    A = np.zeros((n, n), dtype=float)
    for p in range(n):
        for q in range(p + 1, n):
            A[p, q] = B_upper[p, q]
            A[q, p] = -B_upper[p, q]
    return A


def triplet_rdms_from_upper_pair(B_upper: np.ndarray):
    """Return spin-free dm1/dm2 for an Ms=1 two-alpha-electron pair."""
    A = upper_to_antisym(B_upper)
    gamma = A @ A.T
    D_ordered = np.einsum("pq,rs->pqrs", A, A, optimize=True)
    dm2 = D_ordered.transpose(0, 2, 1, 3)
    return gamma, dm2, A, D_ordered


def main():
    args = parse_args()
    inp = Path(args.inp)
    label = safe_label(args.method)
    args.out = args.out or f"step8a_hem_triplet_{label}_rdm_export.npz"
    args.json = args.json or f"step8a_hem_triplet_{label}_rdm_export.json"
    args.summary = args.summary or f"step8a_hem_triplet_{label}_rdm_export_summary.txt"

    data = np.load(inp, allow_pickle=True)
    idx = select_method(data, args.method)
    methods = [str(x) for x in data["methods"]]
    labels = np.array(data["labels"][idx], dtype=object)
    B = np.array(data["pair_coeff_upper"][idx], dtype=float)
    gamma_ref = np.array(data["gamma"][idx], dtype=float)
    metrics_in = load_metrics(data["metrics_json"][idx])
    ao_channels = np.array(data["ao_channels"], dtype=object) if "ao_channels" in data else np.array([], dtype=object)

    dm1, dm2, A, D_ordered = triplet_rdms_from_upper_pair(B)
    diag = rdm_diagnostics(dm1, dm2)
    checks = {
        "pair_coeff_upper_norm": float(np.sqrt(np.sum(B * B))),
        "antisym_pair_tensor_norm": float(np.sqrt(np.sum(A * A))),
        "dm1_vs_gamma_maxabs": maxabs(dm1 - gamma_ref),
        "trace_dm1_error": float(diag["trace_dm1"] - 2.0),
        "trace_dm2_error": float(diag["trace_dm2"] - 2.0),
        "dm1_asym_error": diag["max_dm1_asym"],
        "dm2_bra_ket_error": diag["max_dm2_bra_ket_error"],
        "natural_occupations": diag["natural_occupations"],
    }
    failures = []
    if abs(checks["pair_coeff_upper_norm"] - 1.0) > args.trace_tol:
        failures.append("pair_coeff_upper_norm")
    if checks["dm1_vs_gamma_maxabs"] > 1e-10:
        failures.append("dm1_vs_gamma")
    if abs(checks["trace_dm1_error"]) > args.trace_tol:
        failures.append("dm1_trace")
    if abs(checks["trace_dm2_error"]) > args.trace_tol:
        failures.append("dm2_trace")
    if checks["dm2_bra_ket_error"] > 1e-10:
        failures.append("dm2_bra_ket")
    checks["passed"] = not failures
    checks["failures"] = failures

    metadata = {
        "step": "8a",
        "system": "He metastable triplet",
        "state": "1s2s 3S, Ms=1",
        "source": "hem_2rdm_compare.py output",
        "source_npz": str(inp),
        "method": args.method,
        "available_methods": methods,
        "method_index": idx,
        "nobs": int(B.shape[0]),
        "nelec": 2,
        "spin_sector": "alpha-alpha, nelec=(2,0)",
        "rdm_source_note": "pair_coeff_upper[p,q] for p<q; dm2 = D_ordered.transpose(0,2,1,3)",
        "input_metrics": metrics_in,
        "checks": checks,
    }

    np.savez(
        args.out,
        method=np.array(args.method),
        labels=labels,
        ao_channels=ao_channels,
        pair_coeff_upper=B,
        pair_coeff_antisym=A,
        D_ordered_alpha_alpha=D_ordered,
        gamma_input=gamma_ref,
        dm1_obs=dm1,
        dm2_obs=dm2,
        metrics_json=np.array(json.dumps(metadata, indent=2)),
    )
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    lines = [
        "=" * 88,
        "Step 8a | Export HEM triplet ECG-NO RDM data",
        "=" * 88,
        f"input       = {inp}",
        f"method      = {args.method} (index {idx})",
        f"nobs        = {B.shape[0]} ({2 * B.shape[0]} qubits)",
        f"E_input     = {metrics_in.get('energy_Ha', None)} Ha",
        "",
        "[Checks]",
        f"||B_upper||                     = {checks['pair_coeff_upper_norm']:.12f}",
        f"||A_antisym||                   = {checks['antisym_pair_tensor_norm']:.12f}",
        f"Max|dm1-gamma_input|            = {checks['dm1_vs_gamma_maxabs']:.3e}",
        f"Tr(dm1)-2                       = {checks['trace_dm1_error']:.3e}",
        f"Tr(dm2)-2                       = {checks['trace_dm2_error']:.3e}",
        f"Max|dm1-dm1.T|                  = {checks['dm1_asym_error']:.3e}",
        f"dm2 bra-ket error               = {checks['dm2_bra_ket_error']:.3e}",
        "Natural occupations              = " + np.array2string(np.array(checks["natural_occupations"]), precision=10),
        f"passed                          = {checks['passed']}",
        "",
        "[Saved]",
        f"  {args.out}",
        f"  {args.json}",
        f"  {args.summary}",
    ]
    if failures:
        lines.insert(-4, f"failures                        = {failures}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
