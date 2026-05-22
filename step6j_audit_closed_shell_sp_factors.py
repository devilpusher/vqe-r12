#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6j: Audit He closed-shell spin-free pair and SP normalization factors.

This script narrows the Step-6i overcorrection to explicit factor budgets:

* the validated spin-free RDM convention for a two-electron singlet;
* the SP tensor collapse for the He i=j occupied pair;
* amplitude and energy-prefactor scales needed to recover the full-parent gap;
* common closed-shell/same-pair/spin-free factor products that might explain it.

It is an audit script, not a final [2]R12 correction.
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

from r12_common import build_sp_tensor, maxabs, reconstruct_energy, rdm_diagnostics, tensor_diagnostics
from step6e_build_vxbc_intermediates import default_prefix
from step6f_he_r12_candidate_energy import as_float, energy_metrics, matching_step4b_path
from step6g_audit_approxc_terms import build_formula_matrices, build_tilde_terms, orbital_energy_audit, pair_index


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


def ri_external_pair_indices(nri: int, nocc: int) -> np.ndarray:
    return np.array([pair_index(a, b, nri) for a in range(nocc, nri) for b in range(nocc, nri)], dtype=int)


def solve_T_for_gap(V: float, B: float, gap: float, thresh: float) -> list[float]:
    roots: list[float] = []
    if abs(B) <= thresh:
        if abs(2.0 * V) > thresh:
            roots.append(gap / (2.0 * V))
        return roots
    disc = 4.0 * V * V + 4.0 * B * gap
    if disc < -1e-14:
        return roots
    disc = max(disc, 0.0)
    root = math.sqrt(disc)
    roots.append((-2.0 * V + root) / (2.0 * B))
    roots.append((-2.0 * V - root) / (2.0 * B))
    return roots


def energy_from_T(T: float, V: float, B: float, prefactor: float = 1.0, linear_factor: float = 2.0) -> float:
    return prefactor * (linear_factor * T * V + T * T * B)


def closest_rows(rows: Iterable[Dict[str, Any]], limit: int = 16) -> list[Dict[str, Any]]:
    scored = []
    for row in rows:
        val = row.get("abs_residual_to_full_mEh")
        if val is None:
            continue
        scored.append((float(val), row))
    scored.sort(key=lambda item: item[0])
    return [row for _, row in scored[:limit]]


def validate_singlet_rdm_relation(C: np.ndarray, dm2: np.ndarray) -> Dict[str, Any]:
    """For the current He alpha-beta pair coefficient, dm2[p,q,r,s] = 2*C[p,r]*C[q,s]."""
    model = 2.0 * np.einsum("pr,qs->pqrs", C, C, optimize=True)
    return {
        "relation": "dm2[p,q,r,s] = 2 * Cab[p,r] * Cab[q,s]",
        "max_abs_error": maxabs(dm2 - model),
        "dm2_0000": float(dm2[0, 0, 0, 0]) if dm2.shape[0] else None,
        "Cab_00": float(C[0, 0]) if C.shape[0] else None,
        "2_Cab00_squared": float(2.0 * C[0, 0] * C[0, 0]) if C.shape[0] else None,
        "spin_free_pair_count_trace": float(np.einsum("pprr->", dm2, optimize=True)),
        "two_body_energy_prefactor": 0.5,
        "note": (
            "The spin-free dm2 counts the alpha-beta and beta-alpha ordered spin pairs; "
            "the ordinary electronic energy uses 1/2 * einsum(eri, dm2), giving one physical "
            "electron pair for He."
        ),
    }


def sp_collapse_audit(nobs: int) -> Dict[str, Any]:
    D = build_sp_tensor(nobs)
    return {
        "D_0000_total": float(D[0, 0, 0, 0]),
        "direct_3_8": 3.0 / 8.0,
        "exchange_1_8": 1.0 / 8.0,
        "direct_plus_exchange_for_i_eq_j": 0.5,
        "norm": float(np.linalg.norm(D.reshape(-1))),
        "bra_ket_error": maxabs(D - D.transpose(2, 3, 0, 1)),
        "warning": (
            "For i=j the SP tensor collapses to 1/2 before any closed-shell, "
            "spin-adaptation, or pair-basis normalization factor is applied."
        ),
    }


def factor_catalog() -> Dict[str, float]:
    return {
        "none": 1.0,
        "two_body_energy_1_2": 0.5,
        "ordered_to_unordered_same_pair_1_2": 0.5,
        "closed_shell_opposite_spin_pair_1_2": 0.5,
        "same_spatial_pair_metric_1_2": 0.5,
        "left_right_pair_metric_1_4": 0.25,
        "antisymmetrizer_projector_1_2": 0.5,
        "spin_average_1_4": 0.25,
        "singlet_spin_projector_1_4": 0.25,
        "sqrt_same_pair_norm_1_over_sqrt2": 1.0 / math.sqrt(2.0),
    }


def selected_factor_products() -> list[tuple[str, float]]:
    f = factor_catalog()
    specs = [
        ("none", ["none"]),
        ("energy_1_2", ["two_body_energy_1_2"]),
        ("same_pair_1_2", ["same_spatial_pair_metric_1_2"]),
        ("energy_1_2__same_pair_1_2", ["two_body_energy_1_2", "same_spatial_pair_metric_1_2"]),
        ("energy_1_2__ordered_1_2__same_pair_1_2", ["two_body_energy_1_2", "ordered_to_unordered_same_pair_1_2", "same_spatial_pair_metric_1_2"]),
        ("energy_1_2__spinavg_1_4", ["two_body_energy_1_2", "spin_average_1_4"]),
        ("energy_1_2__singlet_projector_1_4", ["two_body_energy_1_2", "singlet_spin_projector_1_4"]),
        ("energy_1_2__same_pair_1_2__spinavg_1_4", ["two_body_energy_1_2", "same_spatial_pair_metric_1_2", "spin_average_1_4"]),
        ("ordered_1_2__same_pair_1_2__spinavg_1_4", ["ordered_to_unordered_same_pair_1_2", "same_spatial_pair_metric_1_2", "spin_average_1_4"]),
        ("energy_1_2__ordered_1_2__same_pair_1_2__spinavg_1_4", ["two_body_energy_1_2", "ordered_to_unordered_same_pair_1_2", "same_spatial_pair_metric_1_2", "spin_average_1_4"]),
        ("left_right_1_4__spinavg_1_4", ["left_right_pair_metric_1_4", "spin_average_1_4"]),
        ("energy_1_2__left_right_1_4__spinavg_1_4", ["two_body_energy_1_2", "left_right_pair_metric_1_4", "spin_average_1_4"]),
    ]
    out = []
    for label, keys in specs:
        scale = 1.0
        for key in keys:
            scale *= f[key]
        out.append((label, scale))
    return out


def main():
    args = parse_args()
    if args.nocc != 1:
        raise RuntimeError("Step 6j currently audits only He nocc=1.")

    prefix = default_prefix(args.inp)
    if args.out_json is None:
        args.out_json = f"{prefix}_step6j_closed_shell_sp_factor_audit.json"
    if args.out_csv is None:
        args.out_csv = f"{prefix}_step6j_closed_shell_sp_factor_audit.csv"
    if args.summary is None:
        args.summary = f"{prefix}_step6j_closed_shell_sp_factor_audit_summary.txt"

    data = np.load(args.inp, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta.get("nobs", np.array(data["Cab_obs"]).shape[0]))
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))
    nocc = int(args.nocc)

    step4b_path = args.step4b or matching_step4b_path(args.inp)
    E_full: Optional[float] = None
    if step4b_path is not None and Path(step4b_path).exists():
        step4b = np.load(step4b_path, allow_pickle=True)
        E_full = as_float(load_metadata(step4b).get("E_full_parent_fci"))

    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)
    Cab_obs = np.array(data["Cab_obs"], dtype=float)
    E_obs = float(meta["E_obs_fci"])
    enuc = float(meta.get("enuc", 0.0))
    E_obs_rdm, _, _ = reconstruct_energy(h_ri[:nobs, :nobs], eri_ri[:nobs, :nobs, :nobs, :nobs], dm1_obs, dm2_obs, enuc)
    E_ri_rdm, _, _ = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)

    F_ri = np.array(data["F_ri"], dtype=float)
    eps = np.array(orbital_energy_audit(F_ri, nobs, nocc)["eps_diag"], dtype=float)
    matrices = build_formula_matrices(data, nri, nobs, nocc)["matrices"]
    terms = build_tilde_terms(
        matrices,
        eps,
        i=0,
        j=0,
        kl_indices=np.array([pair_index(0, 0, nri)], dtype=int),
        ab_indices=ri_external_pair_indices(nri, nocc),
        n=nri,
        thresh=args.denom_thresh,
        B_source="B_fock_q3",
    )

    # Step 6h established this sign for the current negative Slater convention.
    f12_linear_sign = -1.0
    Vtilde = f12_linear_sign * float(terms["V_tilde"][0])
    Btilde = float(terms["B_tilde"][0, 0])
    T_sp = 0.5
    raw_sp_delta = energy_from_T(T_sp, Vtilde, Btilde)
    gap = None if E_full is None else E_full - E_obs

    target_roots = [] if gap is None else [x for x in solve_T_for_gap(Vtilde, Btilde, gap, args.denom_thresh) if x > 0.0]
    target_T_small = min(target_roots) if target_roots else None
    needed_amplitude_scale = None if target_T_small is None else target_T_small / T_sp
    needed_energy_prefactor = None if gap is None else gap / raw_sp_delta

    rows = []
    for label, scale in selected_factor_products():
        for mode in ["amplitude_scale", "energy_prefactor"]:
            if mode == "amplitude_scale":
                T = T_sp * scale
                pref = 1.0
            else:
                T = T_sp
                pref = scale
            delta = energy_from_T(T, Vtilde, Btilde, prefactor=pref)
            rows.append({
                "label": label,
                "mode": mode,
                "scale": scale,
                "T_effective": T,
                "energy_prefactor": pref,
                "delta_E": delta,
                **energy_metrics(delta, E_obs, E_full),
            })

    if needed_amplitude_scale is not None:
        delta = energy_from_T(T_sp * needed_amplitude_scale, Vtilde, Btilde)
        rows.append({
            "label": "needed_amplitude_scale_from_gap",
            "mode": "amplitude_scale",
            "scale": needed_amplitude_scale,
            "T_effective": T_sp * needed_amplitude_scale,
            "energy_prefactor": 1.0,
            "delta_E": delta,
            **energy_metrics(delta, E_obs, E_full),
        })
    if needed_energy_prefactor is not None:
        delta = energy_from_T(T_sp, Vtilde, Btilde, prefactor=needed_energy_prefactor)
        rows.append({
            "label": "needed_energy_prefactor_from_gap",
            "mode": "energy_prefactor",
            "scale": needed_energy_prefactor,
            "T_effective": T_sp,
            "energy_prefactor": needed_energy_prefactor,
            "delta_E": delta,
            **energy_metrics(delta, E_obs, E_full),
        })

    diagnostics = {
        "input": args.inp,
        "step4b": step4b_path,
        "nri": nri,
        "nobs": nobs,
        "nocc": nocc,
        "E_obs_fci": E_obs,
        "E_full_parent_fci": E_full,
        "gap": gap,
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
        "singlet_rdm_relation": validate_singlet_rdm_relation(Cab_obs, dm2_obs),
        "sp_collapse": sp_collapse_audit(nobs),
        "tilde_terms": {
            "f12_linear_sign": f12_linear_sign,
            "Vtilde": Vtilde,
            "Btilde": Btilde,
            "T_sp_collapsed": T_sp,
            "raw_sp_delta_E": raw_sp_delta,
            "target_T_positive_roots": target_roots,
            "target_T_small": target_T_small,
            "needed_amplitude_scale_vs_T_sp": needed_amplitude_scale,
            "needed_energy_prefactor_vs_raw_sp": needed_energy_prefactor,
        },
        "factor_catalog": factor_catalog(),
        "rows": rows,
        "closest_rows": closest_rows(rows),
        "decision": (
            "If the needed scale is not reproduced by a documented spin/pair convention, "
            "do not bake it into Step 6f.  The next step should map the paper's spin-adapted "
            "geminal operator onto the current spin-free dm2 and ordered-pair tensors."
        ),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    fieldnames = [
        "label", "mode", "scale", "T_effective", "energy_prefactor", "delta_E",
        "E_total", "residual_to_full_parent_FCI", "abs_residual_to_full_mEh",
        "recovery_ratio", "overcorrection",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6j | He closed-shell spin-free pair / SP normalization audit")
    lines.append("=" * 100)
    lines.append(f"input       = {args.inp}")
    lines.append(f"nri/nobs/nocc = {nri}/{nobs}/{nocc}")
    lines.append(f"E_OBS-FCI   = {E_obs:.14f} Eh")
    if E_full is not None:
        lines.append(f"E_full_parent_FCI = {E_full:.14f} Eh")
        lines.append(f"OBS-to-full gap   = {gap:.12e} Eh ({abs(gap) * 1000.0:.6f} mEh)")
    lines.append(f"E checks    = obs-rdm {E_obs_rdm - E_obs:.3e}, ri-rdm {E_ri_rdm - E_obs:.3e}")
    lines.append("")
    rdm_rel = diagnostics["singlet_rdm_relation"]
    lines.append("[Spin-free RDM pair convention]")
    lines.append(f"{rdm_rel['relation']}")
    lines.append(f"max error = {rdm_rel['max_abs_error']:.3e}")
    lines.append(f"dm2[0,0,0,0] = {rdm_rel['dm2_0000']:.12f}; 2*Cab[0,0]^2 = {rdm_rel['2_Cab00_squared']:.12f}")
    lines.append(f"Tr(dm2) = {rdm_rel['spin_free_pair_count_trace']:.12f}; ordinary two-body prefactor = 1/2")
    lines.append("")
    lines.append("[SP collapse and required scales]")
    lines.append(f"D_sp[0,0,0,0] = {T_sp:.12f} = 3/8 + 1/8")
    lines.append(f"Vtilde = {Vtilde:.12e}; Btilde = {Btilde:.12e}")
    lines.append(f"raw SP DeltaE with T=1/2 = {raw_sp_delta:.12e} Eh")
    if needed_amplitude_scale is not None:
        lines.append(f"target T small root = {target_T_small:.12e}")
        lines.append(f"needed amplitude scale relative to T=1/2 = {needed_amplitude_scale:.12e}")
    if needed_energy_prefactor is not None:
        lines.append(f"needed energy prefactor for fixed T=1/2 = {needed_energy_prefactor:.12e}")
    lines.append("")
    lines.append("[Closest factor-product rows]")
    lines.append("| label | mode | scale | T_eff | pref | DeltaE | residual / mEh | recovery |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in diagnostics["closest_rows"]:
        lines.append(
            f"| {row['label']} | {row['mode']} | {row['scale']:.8e} | "
            f"{row['T_effective']:.8e} | {row['energy_prefactor']:.8e} | "
            f"{row['delta_E']:.8e} | {row['abs_residual_to_full_mEh']:.6e} | "
            f"{row['recovery_ratio']:.6e} |"
        )
    lines.append("")
    lines.append("[Decision]")
    lines.append(diagnostics["decision"])

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
        and rdm_rel["max_abs_error"] < 1e-10
        and all(not d["has_nan"] and not d["has_inf"] for d in diagnostics["tensor_diagnostics"].values())
    )
    if not ok:
        print("\nERROR: Step 6j consistency checks failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
