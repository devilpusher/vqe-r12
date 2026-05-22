#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6i: Audit SP prefactor and closed-shell pair normalization for He.

This script starts from the Step-6g audited 3C(FIX)/SP building blocks and asks
how much of the Step-6f overcorrection can be explained by amplitude, spin, and
ordered-pair normalization conventions.

It is not a final [2]R12 correction.  It is a controlled normalization audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from r12_common import reconstruct_energy, rdm_diagnostics, tensor_diagnostics
from step6e_build_vxbc_intermediates import default_prefix
from step6f_he_r12_candidate_energy import as_float, energy_metrics, matching_step4b_path
from step6g_audit_approxc_terms import (
    build_formula_matrices,
    build_tilde_terms,
    orbital_energy_audit,
    pair_index,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="he_ccpvdz_nobs2_fitN7_step5a_r12_intermediates.npz")
    p.add_argument("--step4b", default=None)
    p.add_argument("--nocc", type=int, default=1)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--denom-thresh", type=float, default=1e-10)
    return p.parse_args()


def load_metadata(data) -> Dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    return json.loads(str(data["metadata_json"]))


def solve_T_for_gap(V: float, B: float, gap: float, thresh: float) -> list[float]:
    """Solve B*T^2 + 2*V*T - gap = 0."""
    roots = []
    if abs(B) <= thresh:
        if abs(2.0 * V) > thresh:
            roots.append(gap / (2.0 * V))
        return roots
    disc = 4.0 * V * V + 4.0 * B * gap
    if disc < -1e-14:
        return roots
    disc = max(disc, 0.0)
    sqrt_disc = math.sqrt(disc)
    roots.append((-2.0 * V + sqrt_disc) / (2.0 * B))
    roots.append((-2.0 * V - sqrt_disc) / (2.0 * B))
    return roots


def unique_ab_indices(nri: int, nocc: int) -> Dict[str, np.ndarray]:
    ri_ext = list(range(nocc, nri))
    ordered = []
    unordered_unique = []
    diagonal_only = []
    offdiag_unique = []
    for a in ri_ext:
        for b in ri_ext:
            ordered.append(pair_index(a, b, nri))
            if a == b:
                diagonal_only.append(pair_index(a, b, nri))
            if a <= b:
                unordered_unique.append(pair_index(a, b, nri))
                if a < b:
                    offdiag_unique.append(pair_index(a, b, nri))
    return {
        "ri_external_ordered": np.array(ordered, dtype=int),
        "ri_external_unordered_unique": np.array(unordered_unique, dtype=int),
        "ri_external_diagonal_only": np.array(diagonal_only, dtype=int),
        "ri_external_offdiag_unique": np.array(offdiag_unique, dtype=int),
    }


def make_T_variants() -> Dict[str, float]:
    return {
        "unit_1": 1.0,
        "sp_1_2": 0.5,
        "direct_3_8": 3.0 / 8.0,
        "quarter_1_4": 0.25,
        "exchange_1_8": 1.0 / 8.0,
        "one_over_16": 1.0 / 16.0,
        "one_over_20": 1.0 / 20.0,
        "one_over_24": 1.0 / 24.0,
        "one_over_32": 1.0 / 32.0,
        "one_over_64": 1.0 / 64.0,
    }


def row_for_variant(
    label: str,
    ab_label: str,
    T: float,
    formula_prefactor: float,
    linear_factor: float,
    Vtilde: float,
    Btilde: float,
    E_obs: float,
    E_full: Optional[float],
) -> Dict[str, Any]:
    linear = linear_factor * T * Vtilde
    quadratic = T * T * Btilde
    delta = formula_prefactor * (linear + quadratic)
    return {
        "variant": label,
        "ab_space": ab_label,
        "T": T,
        "formula_prefactor": formula_prefactor,
        "linear_factor": linear_factor,
        "Vtilde": Vtilde,
        "Btilde": Btilde,
        "linear": formula_prefactor * linear,
        "quadratic": formula_prefactor * quadratic,
        "delta_E": delta,
        **energy_metrics(delta, E_obs, E_full),
    }


def closest_rows(rows: Iterable[Dict[str, Any]], limit: int = 12) -> list[Dict[str, Any]]:
    scored = []
    for row in rows:
        residual = row.get("abs_residual_to_full_mEh")
        if residual is None:
            continue
        scored.append((float(residual), row))
    scored.sort(key=lambda x: x[0])
    return [row for _, row in scored[:limit]]


def fmt(x: Any, prec: int = 8) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{prec}e}"
    except Exception:
        return str(x)


def main():
    args = parse_args()
    prefix = default_prefix(args.inp)
    if args.out_json is None:
        args.out_json = f"{prefix}_step6i_sp_normalization_audit.json"
    if args.out_csv is None:
        args.out_csv = f"{prefix}_step6i_sp_normalization_audit.csv"
    if args.summary is None:
        args.summary = f"{prefix}_step6i_sp_normalization_audit_summary.txt"

    data = np.load(args.inp, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta.get("nobs", np.array(data["Cab_obs"]).shape[0]))
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))
    nocc = int(args.nocc)
    if nocc != 1:
        raise RuntimeError("Step 6i currently audits only the He nocc=1 closed-shell pair.")

    step4b_path = args.step4b or matching_step4b_path(args.inp)
    E_full = None
    if step4b_path is not None and Path(step4b_path).exists():
        step4b = np.load(step4b_path, allow_pickle=True)
        E_full = as_float(load_metadata(step4b).get("E_full_parent_fci"))

    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)
    enuc = float(meta.get("enuc", 0.0))
    E_obs = float(meta["E_obs_fci"])
    E_obs_rdm, _, _ = reconstruct_energy(h_ri[:nobs, :nobs], eri_ri[:nobs, :nobs, :nobs, :nobs], dm1_obs, dm2_obs, enuc)
    E_ri_rdm, _, _ = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)

    F_ri = np.array(data["F_ri"], dtype=float)
    eps = np.array(orbital_energy_audit(F_ri, nobs, nocc)["eps_diag"], dtype=float)
    built = build_formula_matrices(data, nri, nobs, nocc)
    ab_spaces = unique_ab_indices(nri, nocc)
    kl_idx = np.array([pair_index(0, 0, nri)], dtype=int)

    rows = []
    target_T = {}
    # Use f12_linear_sign=-1, established by Step 6h for the current negative Slater convention.
    f12_linear_sign = -1.0
    for ab_label, ab_idx in ab_spaces.items():
        if len(ab_idx) == 0:
            continue
        terms = build_tilde_terms(
            built["matrices"],
            eps,
            i=0,
            j=0,
            kl_indices=kl_idx,
            ab_indices=ab_idx,
            n=nri,
            thresh=args.denom_thresh,
            B_source="B_fock_q3",
        )
        Vtilde = f12_linear_sign * float(terms["V_tilde"][0])
        Btilde = float(terms["B_tilde"][0, 0])
        gap = None if E_full is None else E_full - E_obs
        roots = [] if gap is None else solve_T_for_gap(Vtilde, Btilde, gap, args.denom_thresh)
        target_T[ab_label] = {
            "Vtilde_after_f12_linear_sign": Vtilde,
            "Btilde": Btilde,
            "gap": gap,
            "roots_for_T_with_prefactor_1_linear2": roots,
        }

        for t_label, T in make_T_variants().items():
            for pref_label, formula_prefactor, linear_factor in [
                ("E=T(2V+BT)", 1.0, 2.0),
                ("E=1/2*T(2V+BT)", 0.5, 2.0),
                ("E=1/4*T(2V+BT)", 0.25, 2.0),
                ("E=T(V+BT)", 1.0, 1.0),
                ("E=1/2*T(V+BT)", 0.5, 1.0),
            ]:
                label = f"{t_label}__{pref_label}"
                rows.append(row_for_variant(label, ab_label, T, formula_prefactor, linear_factor, Vtilde, Btilde, E_obs, E_full))

        for root in roots:
            if root > 0.0:
                rows.append(row_for_variant("T_solved_for_gap__E=T(2V+BT)", ab_label, root, 1.0, 2.0, Vtilde, Btilde, E_obs, E_full))

    closest = closest_rows(rows)
    diagnostics = {
        "input": args.inp,
        "step4b": step4b_path,
        "nri": nri,
        "nobs": nobs,
        "nocc": nocc,
        "E_obs_fci": E_obs,
        "E_full_parent_fci": E_full,
        "gap": None if E_full is None else E_full - E_obs,
        "energy_checks": {
            "E_obs_rdm": E_obs_rdm,
            "E_ri_embedded_rdm": E_ri_rdm,
            "delta_obs_rdm_minus_fci": E_obs_rdm - E_obs,
            "delta_ri_rdm_minus_fci": E_ri_rdm - E_obs,
        },
        "rdm_diagnostics": {
            "obs": rdm_diagnostics(dm1_obs, dm2_obs),
            "ri": rdm_diagnostics(dm1_ri, dm2_ri),
        },
        "tensor_diagnostics": {
            "f12sq_ri": tensor_diagnostics(np.array(data["f12sq_ri"], dtype=float)),
            "f12g12_ri": tensor_diagnostics(np.array(data["f12g12_ri"], dtype=float)),
            "f12dc_ri": tensor_diagnostics(np.array(data["f12dc_ri"], dtype=float)),
        },
        "f12_linear_sign": f12_linear_sign,
        "target_T": target_T,
        "rows": rows,
        "closest_rows": closest,
        "important_note": (
            "Step 6i audits normalization only. It uses the Step-6h f12_linear_sign=-1 convention "
            "and varies T, formula prefactors, and RI-external pair summations."
        ),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    fieldnames = [
        "variant", "ab_space", "T", "formula_prefactor", "linear_factor",
        "Vtilde", "Btilde", "linear", "quadratic", "delta_E", "E_total",
        "residual_to_full_parent_FCI", "abs_residual_to_full_mEh",
        "recovery_ratio", "overcorrection",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6i | SP normalization and closed-shell pair multiplicity audit")
    lines.append("=" * 100)
    lines.append(f"input       = {args.inp}")
    lines.append(f"nri/nobs/nocc = {nri}/{nobs}/{nocc}")
    lines.append(f"E_OBS-FCI   = {E_obs:.14f} Eh")
    if E_full is not None:
        gap = E_full - E_obs
        lines.append(f"E_full_parent_FCI = {E_full:.14f} Eh")
        lines.append(f"OBS-to-full gap   = {gap:.12e} Eh ({abs(gap) * 1000.0:.6f} mEh)")
    lines.append(f"E checks    = obs-rdm {E_obs_rdm - E_obs:.3e}, ri-rdm {E_ri_rdm - E_obs:.3e}")
    lines.append("")
    lines.append("[Solved T for target gap]")
    lines.append("| ab space | Vtilde | Btilde | positive T roots for E=T(2V+BT) |")
    lines.append("|---|---:|---:|---|")
    for ab_label, info in target_T.items():
        positive_roots = [x for x in info["roots_for_T_with_prefactor_1_linear2"] if x > 0.0]
        roots_txt = ", ".join(f"{x:.8e}" for x in positive_roots) if positive_roots else ""
        lines.append(
            f"| {ab_label} | {info['Vtilde_after_f12_linear_sign']:.8e} "
            f"| {info['Btilde']:.8e} | {roots_txt} |"
        )
    lines.append("")
    lines.append("[Closest rows by absolute residual]")
    lines.append("| variant | ab space | T | pref | linfac | DeltaE | residual / mEh | recovery |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in closest:
        lines.append(
            f"| {row['variant']} | {row['ab_space']} | {fmt(row['T'])} "
            f"| {fmt(row['formula_prefactor'])} | {fmt(row['linear_factor'])} "
            f"| {fmt(row['delta_E'])} | {fmt(row['abs_residual_to_full_mEh'], 6)} "
            f"| {fmt(row['recovery_ratio'], 6)} |"
        )
    lines.append("")
    lines.append("[Reference SP rows]")
    for row in rows:
        if row["variant"].startswith("sp_1_2__E=T(2V+BT)") or row["variant"].startswith("direct_3_8__E=T(2V+BT)") or row["variant"].startswith("exchange_1_8__E=T(2V+BT)"):
            lines.append(
                f"{row['ab_space']} / {row['variant']}: DeltaE={row['delta_E']:.8e}, "
                f"residual_mEh={row['abs_residual_to_full_mEh']:.6f}, recovery={row['recovery_ratio']:.6f}"
            )
    lines.append("")
    lines.append("[Decision]")
    lines.append(diagnostics["important_note"])

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n")
        f.write(json.dumps(diagnostics, indent=2))
        f.write("\n")

    print("\n".join(lines))
    print("\n[Saved]")
    print(f"  {args.out_json}")
    print(f"  {args.out_csv}")
    print(f"  {args.summary}")

    ok = (
        abs(E_obs_rdm - E_obs) < 1e-10
        and abs(E_ri_rdm - E_obs) < 1e-10
        and all(not d["has_nan"] and not d["has_inf"] for d in diagnostics["tensor_diagnostics"].values())
    )
    if not ok:
        print("\nERROR: Step 6i consistency checks failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
