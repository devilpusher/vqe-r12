#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6f: He-only [2]R12 candidate energies from direct F12 tensors.

This script turns the Step-6e V/X/B/C bookkeeping into a compact candidate
energy table.  It deliberately keeps the scope narrow:

* He only;
* parent-basis RI tensors only;
* direct Psi4 f12g12/f12sq/f12dc tensors are the authoritative inputs;
* candidate rows are diagnostics for formula auditing, not final publishable
  [2]R12 corrections.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from r12_common import (
    assert_finite,
    maxabs,
    pair_matrix,
    reconstruct_energy,
    rdm_diagnostics,
    tensor_diagnostics,
)
from step6e_build_vxbc_intermediates import ansatz3_indices, build_intermediate_matrices, default_prefix, source_vectors
from step6g_audit_approxc_terms import (
    ab_space_indices,
    build_formula_matrices,
    convention_variant_rows,
    orbital_energy_audit,
    pair_index,
)
from step6k_audit_paper_tequila_sf2r12 import (
    build_fock_tequila,
    chem_to_phys,
    compute_tequila_style_components,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="he_ccpvdz_nobs2_fitN7_step5a_r12_intermediates.npz")
    p.add_argument("--step4b", default=None, help="Optional matching Step-4b npz for full-parent FCI target.")
    p.add_argument("--nocc", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--denom-thresh", type=float, default=1e-10)
    p.add_argument("--overrecover-tol", type=float, default=0.05)
    return p.parse_args()


def load_metadata(data) -> Dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    try:
        return json.loads(str(data["metadata_json"]))
    except Exception:
        return {}


def matching_step4b_path(inp: str) -> Optional[str]:
    name = Path(inp).name
    suffix = "_step5a_r12_intermediates.npz"
    if not name.endswith(suffix):
        return None
    candidate = Path(name[: -len(suffix)] + "_step4b_obs_fci_rdm.npz")
    return str(candidate) if candidate.exists() else None


def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_div(num: float, den: float, thresh: float) -> Optional[float]:
    if abs(den) <= thresh:
        return None
    return num / den


def energy_metrics(delta: Optional[float], E_obs: float, E_full: Optional[float]) -> Dict[str, Any]:
    if delta is None:
        return {
            "E_total": None,
            "residual_to_full_parent_FCI": None,
            "abs_residual_to_full_mEh": None,
            "recovery_ratio": None,
            "overcorrection": None,
        }
    E_total = E_obs + delta
    if E_full is None:
        residual = None
        abs_residual_mEh = None
        recovery = None
        over = None
    else:
        residual = E_total - E_full
        abs_residual_mEh = abs(residual) * 1000.0
        gap = E_full - E_obs
        recovery = delta / gap if abs(gap) > 0.0 else None
        over = None if recovery is None else recovery > 1.0
    return {
        "E_total": E_total,
        "residual_to_full_parent_FCI": residual,
        "abs_residual_to_full_mEh": abs_residual_mEh,
        "recovery_ratio": recovery,
        "overcorrection": over,
    }


def fmt(x: Any, prec: int = 8) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x)
    try:
        return f"{float(x):.{prec}e}"
    except Exception:
        return str(x)


def candidate_rows(
    sources: Dict[str, np.ndarray],
    matrices: Dict[str, np.ndarray],
    E_obs: float,
    E_full: Optional[float],
    thresh: float,
) -> list[Dict[str, Any]]:
    psi0 = sources["psi0"]
    denominator_models = {
        "X_plus_dc": ("X + f12dc", ["X_direct_f12sq", "B_direct_f12dc"]),
        "X_plus_dc_plus_Cmodel": ("X + f12dc + C_model", ["X_direct_f12sq", "B_direct_f12dc", "C_fock_model"]),
        "dc_only": ("f12dc only", ["B_direct_f12dc"]),
        "X_only": ("X only", ["X_direct_f12sq"]),
        "Cmodel_only": ("C_model only", ["C_fock_model"]),
    }

    rows: list[Dict[str, Any]] = []
    Vmat = matrices["V_direct_f12g12"]
    for source_name, vec in sources.items():
        V = float(psi0 @ (Vmat @ vec))
        for denom_name, (denom_desc, pieces) in denominator_models.items():
            pieces_values = {}
            D = 0.0
            for piece in pieces:
                val = float(vec @ (matrices[piece] @ vec))
                pieces_values[piece] = val
                D += val

            delta_fixed = 2.0 * V + D
            delta_opt = safe_div(-V * V, D, thresh)
            lambda_opt = safe_div(-V, D, thresh)

            for amp_model, delta, lam in [
                ("fixed_SP_like", delta_fixed, 1.0),
                ("one_dim_opt", delta_opt, lambda_opt),
            ]:
                metrics = energy_metrics(delta, E_obs, E_full)
                rows.append({
                    "source": source_name,
                    "denominator_model": denom_name,
                    "denominator_description": denom_desc,
                    "amplitude_model": amp_model,
                    "V_direct": V,
                    "X_direct": pieces_values.get("X_direct_f12sq", 0.0),
                    "B_dc_direct": pieces_values.get("B_direct_f12dc", 0.0),
                    "C_fock_model": pieces_values.get("C_fock_model", 0.0),
                    "denominator": D,
                    "lambda": lam,
                    "delta_E": delta,
                    **metrics,
                })
    return rows


def audited_3c_fix_sp_rows(
    data,
    nri: int,
    nobs: int,
    nocc: int,
    E_obs: float,
    E_full: Optional[float],
    thresh: float,
) -> Dict[str, Any]:
    if nocc != 1:
        raise RuntimeError("audited_3C_FIX_SP candidate currently supports the He nocc=1 case only.")

    F_ri = np.array(data["F_ri"], dtype=float)
    eps_info = orbital_energy_audit(F_ri, nobs, nocc)
    eps = np.array(eps_info["eps_diag"], dtype=float)
    built = build_formula_matrices(data, nri, nobs, nocc)
    spaces = ab_space_indices(nri, nobs, nocc)
    ab_idx = spaces["ri_external"]
    kl_idx = np.array([pair_index(k, l, nri) for k in range(nocc) for l in range(nocc)], dtype=int)
    variants = convention_variant_rows(
        built["matrices"],
        eps,
        0,
        0,
        kl_idx,
        ab_idx,
        nri,
        thresh,
    )

    selected = {
        "formula_baseline": "audited_3C_FIX_SP_formula_baseline",
        "flip_f12_linear_terms": "audited_3C_FIX_SP",
    }
    out = []
    for variant in variants["rows"]:
        if variant["name"] not in selected:
            continue
        metrics = energy_metrics(variant["delta_E_candidate"], E_obs, E_full)
        out.append({
            "source": selected[variant["name"]],
            "denominator_model": f"tilde_B_fock_q3_ri_external__{variant['name']}",
            "denominator_description": variant["description"],
            "amplitude_model": "SP_fixed_3/8_plus_1/8",
            "V_direct": float(variant["V_tilde"][0]) if variant["V_tilde"] else 0.0,
            "X_direct": 0.0,
            "B_dc_direct": 0.0,
            "C_fock_model": variant["C_over_den_B_norm"],
            "denominator": variant["B_tilde_norm"],
            "lambda": 1.0,
            "delta_E": variant["delta_E_candidate"],
            "linear_2T_Vtilde": variant["linear_2T_Vtilde_before_energy_sign"],
            "quadratic_T_Btilde_T": variant["quadratic_T_Btilde_T_before_energy_sign"],
            "V_tilde_norm": float(np.linalg.norm(np.array(variant["V_tilde"], dtype=float))),
            "B_tilde_norm": variant["B_tilde_norm"],
            "C_kl_ab_norm": None,
            "C_over_den_V_norm": float(np.linalg.norm(np.array(variant["C_over_den_V"], dtype=float))),
            "C_over_den_B_norm": variant["C_over_den_B_norm"],
            "ab_space": "ri_external",
            "B_source": "B_fock_q3",
            "occupied_pair_block": [[0, 0]],
            "n_skipped_denominators": int(variant["n_skipped_denominators"]),
            "epsilon_source": eps_info["epsilon_source"],
            "convention_variant": variant["name"],
            "f12_linear_sign": variant["f12_linear_sign"],
            "denom_sign": variant["denom_sign"],
            "energy_sign": variant["energy_sign"],
            **metrics,
        })
    return out


def paper_tequila_sf2r12_rows(
    data,
    nri: int,
    nobs: int,
    E_obs: float,
    E_full: Optional[float],
) -> list[Dict[str, Any]]:
    """Candidate rows following the paper's public Tequila SF-[2]R12 implementation.

    This uses CABS-only passive indices, p = RI minus OBS, and the full
    V+B+X+Delta spin-free contraction instead of an occupied-pair scalar
    prefactor model.
    """
    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    f12_ri = np.array(data["f12_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    step5a_fock = np.array(data["F_ri"], dtype=float)
    g_phys = chem_to_phys(eri_ri)
    r_phys = chem_to_phys(f12_ri)
    tequila_fock = build_fock_tequila(h_ri, g_phys, dm1_obs, list(range(nobs)), list(range(nri)))

    rows = []
    for fock_model, fock in [
        ("tequila_fock_from_paper_formula", tequila_fock),
        ("step5a_generalized_fock_comparison", step5a_fock),
    ]:
        components = compute_tequila_style_components(g_phys, r_phys, fock, dm1_obs, dm2_obs, nobs, nri)
        delta = components["correction"]
        metrics = energy_metrics(delta, E_obs, E_full)
        rows.append({
            "source": "paper_tequila_sf2r12",
            "denominator_model": "CABS_only_passive_full_V+B+X+Delta",
            "denominator_description": (
                "Paper/Tequila SF-[2]R12 contraction with passive p=RI-OBS and "
                "correction=sum(V,B,X,Delta_MBeq)."
            ),
            "amplitude_model": "SP_tensor_3/8_direct_plus_1/8_exchange",
            "V_direct": components["V"],
            "X_direct": components["X"],
            "B_dc_direct": components["B"],
            "C_fock_model": components["Delta"],
            "denominator": components["B"] + components["X"] + components["Delta"],
            "lambda": 1.0,
            "delta_E": delta,
            "V_component": components["V"],
            "B_component": components["B"],
            "X_component": components["X"],
            "Delta_component": components["Delta"],
            "fock_model": fock_model,
            "ab_space": "CABS_only_passive_RI_minus_OBS",
            "passive_indices": list(range(nobs, nri)),
            "n_passive": nri - nobs,
            "formula_source": (
                "D2CP00247G.pdf Eq. (7)-(8); Tequila "
                "quantumchemistry/f12_corrections/_f12_correction_base.py"
            ),
            **metrics,
        })
    return rows


def finite_checks(arrays: Dict[str, np.ndarray]) -> None:
    for name, arr in arrays.items():
        assert_finite(name, arr)


def bool_warn(rows: Iterable[Dict[str, Any]], tol: float) -> list[str]:
    warnings = []
    for row in rows:
        ratio = row.get("recovery_ratio")
        if ratio is None:
            continue
        if ratio > 1.0 + tol:
            warnings.append(
                f"{row['source']} / {row['denominator_model']} / {row['amplitude_model']} "
                f"over-recovers the OBS-full gap: ratio={ratio:.6f}"
            )
        if ratio < -tol:
            warnings.append(
                f"{row['source']} / {row['denominator_model']} / {row['amplitude_model']} "
                f"moves in the wrong direction: ratio={ratio:.6f}"
            )
    return warnings


def closest_rows(rows: Iterable[Dict[str, Any]], limit: int = 8) -> list[Dict[str, Any]]:
    scored = []
    for row in rows:
        residual = row.get("abs_residual_to_full_mEh")
        if residual is None:
            continue
        scored.append((float(residual), row))
    scored.sort(key=lambda x: x[0])
    return [row for _, row in scored[:limit]]


def main():
    args = parse_args()
    prefix = default_prefix(args.inp)
    if args.out is None:
        args.out = f"{prefix}_step6f_he_r12_candidate_energy.json"
    if args.summary is None:
        args.summary = f"{prefix}_step6f_he_r12_candidate_energy_summary.txt"
    if args.csv is None:
        args.csv = f"{prefix}_step6f_he_r12_candidate_energy.csv"

    data = np.load(args.inp, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta.get("nobs", np.array(data["Cab_obs"]).shape[0]))
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))
    nocc = int(args.nocc)
    enuc = float(meta.get("enuc", 0.0))
    E_obs = float(meta["E_obs_fci"])

    step4b_path = args.step4b or matching_step4b_path(args.inp)
    step4b_meta = {}
    E_full = None
    if step4b_path is not None and Path(step4b_path).exists():
        step4b_data = np.load(step4b_path, allow_pickle=True)
        step4b_meta = load_metadata(step4b_data)
        E_full = as_float(step4b_meta.get("E_full_parent_fci"))

    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)
    f12sq_ri = np.array(data["f12sq_ri"], dtype=float)
    f12g12_ri = np.array(data["f12g12_ri"], dtype=float)
    f12dc_ri = np.array(data["f12dc_ri"], dtype=float)

    finite_checks({
        "h_ri": h_ri,
        "eri_ri": eri_ri,
        "dm1_obs": dm1_obs,
        "dm2_obs": dm2_obs,
        "dm1_ri": dm1_ri,
        "dm2_ri": dm2_ri,
        "f12sq_ri": f12sq_ri,
        "f12g12_ri": f12g12_ri,
        "f12dc_ri": f12dc_ri,
    })

    E_obs_rdm, E1_obs, E2_obs = reconstruct_energy(
        h_ri[:nobs, :nobs],
        eri_ri[:nobs, :nobs, :nobs, :nobs],
        dm1_obs,
        dm2_obs,
        enuc,
    )
    E_ri_rdm, E1_ri, E2_ri = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)

    ints = build_intermediate_matrices(data, nri, nobs, nocc)
    sources = source_vectors(data, nri, nobs, ints["indices"]["q_ansatz3"])
    rows = candidate_rows(sources, ints["matrices"], E_obs, E_full, args.denom_thresh)
    audited_rows = audited_3c_fix_sp_rows(data, nri, nobs, nocc, E_obs, E_full, args.denom_thresh)
    paper_rows = paper_tequila_sf2r12_rows(data, nri, nobs, E_obs, E_full)
    rows.extend(audited_rows)
    rows.extend(paper_rows)
    warnings = bool_warn(rows, args.overrecover_tol)
    closest = closest_rows(rows)

    F12 = pair_matrix(np.array(data["f12_ri"], dtype=float))
    G = pair_matrix(eri_ri)
    direct_consistency = {
        "maxabs_direct_f12g12_minus_gf_full_closure": maxabs(pair_matrix(f12g12_ri) - G @ F12),
        "maxabs_direct_f12sq_minus_ff_full_closure": maxabs(pair_matrix(f12sq_ri) - F12 @ F12),
    }
    projector_dims = {key: int(len(value)) for key, value in ansatz3_indices(nri, nobs, nocc).items()}

    diagnostics = {
        "input": args.inp,
        "step4b": step4b_path,
        "nobs": nobs,
        "nri": nri,
        "nocc": nocc,
        "E_obs_fci": E_obs,
        "E_full_parent_fci": E_full,
        "full_parent_gap": None if E_full is None else E_full - E_obs,
        "energy_checks": {
            "E_obs_rdm": E_obs_rdm,
            "E_ri_embedded_rdm": E_ri_rdm,
            "delta_obs_rdm_minus_fci": E_obs_rdm - E_obs,
            "delta_ri_rdm_minus_fci": E_ri_rdm - E_obs,
            "E1_obs": E1_obs,
            "E2_obs": E2_obs,
            "E1_ri": E1_ri,
            "E2_ri": E2_ri,
        },
        "rdm_diagnostics": {
            "obs": rdm_diagnostics(dm1_obs, dm2_obs),
            "ri": rdm_diagnostics(dm1_ri, dm2_ri),
        },
        "tensor_diagnostics": {
            "f12sq_ri": tensor_diagnostics(f12sq_ri),
            "f12g12_ri": tensor_diagnostics(f12g12_ri),
            "f12dc_ri": tensor_diagnostics(f12dc_ri),
        },
        "direct_tensor_consistency": direct_consistency,
        "projector_dimensions": projector_dims,
        "candidate_rows": rows,
        "audited_3C_FIX_SP": audited_rows,
        "paper_tequila_sf2r12": paper_rows,
        "closest_rows_by_abs_residual_mEh": closest,
        "warnings": warnings,
        "important_note": (
            "Step 6f is a He-only candidate-energy ledger using direct f12g12/f12sq/f12dc "
            "tensors as authoritative inputs. The audited_3C_FIX_SP row uses the Step-6g "
            "B_fock_q3 + tilde V/B + occupied-pair-block path. The paper_tequila_sf2r12 "
            "row follows the paper/Tequila SF-[2]R12 contraction with CABS-only passive "
            "indices and full V+B+X+Delta. Other rows involving C_fock_model remain "
            "denominator diagnostics."
        ),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    fieldnames = [
        "source",
        "denominator_model",
        "amplitude_model",
        "V_direct",
        "X_direct",
        "B_dc_direct",
        "C_fock_model",
        "denominator",
        "lambda",
        "delta_E",
        "E_total",
        "residual_to_full_parent_FCI",
        "abs_residual_to_full_mEh",
        "recovery_ratio",
        "overcorrection",
        "linear_2T_Vtilde",
        "quadratic_T_Btilde_T",
        "V_tilde_norm",
        "B_tilde_norm",
        "C_kl_ab_norm",
        "C_over_den_V_norm",
        "C_over_den_B_norm",
        "ab_space",
        "B_source",
        "convention_variant",
        "f12_linear_sign",
        "denom_sign",
        "energy_sign",
        "fock_model",
        "V_component",
        "B_component",
        "X_component",
        "Delta_component",
        "n_passive",
        "formula_source",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6f | He-only [2]R12 candidate energy ledger")
    lines.append("=" * 100)
    lines.append(f"input       = {args.inp}")
    lines.append(f"step4b      = {step4b_path}")
    lines.append(f"nri/nobs/nocc = {nri}/{nobs}/{nocc}")
    lines.append(f"E_OBS-FCI   = {E_obs:.14f} Eh")
    lines.append(f"E_OBS-RDM   = {E_obs_rdm:.14f} Eh")
    lines.append(f"E_RI-RDM    = {E_ri_rdm:.14f} Eh")
    if E_full is not None:
        gap = E_full - E_obs
        lines.append(f"E_full_parent_FCI = {E_full:.14f} Eh")
        lines.append(f"OBS-to-full gap   = {gap:.12e} Eh ({abs(gap) * 1000.0:.6f} mEh)")
    lines.append("")
    lines.append("[Direct tensor policy]")
    lines.append("direct f12g12/f12sq/f12dc tensors are authoritative; finite RI closures are diagnostics only.")
    lines.append(
        "max|direct f12g12 - g*f closure| = "
        f"{direct_consistency['maxabs_direct_f12g12_minus_gf_full_closure']:.3e}"
    )
    lines.append(
        "max|direct f12sq  - f*f closure| = "
        f"{direct_consistency['maxabs_direct_f12sq_minus_ff_full_closure']:.3e}"
    )
    lines.append("")
    lines.append("[Audited 3C(FIX)/SP rows]")
    lines.append("| source | variant | f12 sign | ab space | B source | linear 2T.Vt | quad T.Bt.T | DeltaE / Eh | residual / mEh | recovery | flag |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---:|---:|---|")
    for audited_row in audited_rows:
        flag = ""
        if audited_row["overcorrection"] is True:
            flag = "over"
        elif audited_row["recovery_ratio"] is not None and audited_row["recovery_ratio"] < 0.0:
            flag = "wrong-sign"
        lines.append(
            f"| {audited_row['source']} | {audited_row['convention_variant']} "
            f"| {audited_row['f12_linear_sign']:.0f} | {audited_row['ab_space']} | {audited_row['B_source']} "
            f"| {fmt(audited_row['linear_2T_Vtilde'])} "
            f"| {fmt(audited_row['quadratic_T_Btilde_T'])} "
            f"| {fmt(audited_row['delta_E'])} "
            f"| {fmt(audited_row['abs_residual_to_full_mEh'], 6)} "
            f"| {fmt(audited_row['recovery_ratio'], 6)} | {flag} |"
        )
    lines.append("")
    lines.append("[Paper/Tequila SF-[2]R12 rows]")
    lines.append("| source | fock model | passive | V | B | X | Delta | total / Eh | residual / mEh | recovery |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for paper_row in paper_rows:
        lines.append(
            f"| {paper_row['source']} | {paper_row['fock_model']} | {paper_row['ab_space']} "
            f"| {fmt(paper_row['V_component'])} | {fmt(paper_row['B_component'])} "
            f"| {fmt(paper_row['X_component'])} | {fmt(paper_row['Delta_component'])} "
            f"| {fmt(paper_row['delta_E'])} | {fmt(paper_row['abs_residual_to_full_mEh'], 6)} "
            f"| {fmt(paper_row['recovery_ratio'], 6)} |"
        )
    lines.append("")
    lines.append("[Candidate energy rows]")
    lines.append(
        "| source | denom | amp | DeltaE / Eh | E total / Eh | residual / mEh | recovery | flag |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for row in rows:
        flag = ""
        if row["overcorrection"] is True:
            flag = "over"
        elif row["recovery_ratio"] is not None and row["recovery_ratio"] < 0.0:
            flag = "wrong-sign"
        lines.append(
            f"| {row['source']} | {row['denominator_model']} | {row['amplitude_model']} "
            f"| {fmt(row['delta_E'])} | {fmt(row['E_total'], 12)} "
            f"| {fmt(row['abs_residual_to_full_mEh'], 6)} | {fmt(row['recovery_ratio'], 6)} | {flag} |"
        )
    if closest:
        lines.append("")
        lines.append("[Closest rows by absolute residual]")
        lines.append("| source | denom | amp | residual / mEh | recovery | DeltaE / Eh |")
        lines.append("|---|---|---|---:|---:|---:|")
        for row in closest:
            lines.append(
                f"| {row['source']} | {row['denominator_model']} | {row['amplitude_model']} "
                f"| {fmt(row['abs_residual_to_full_mEh'], 6)} | {fmt(row['recovery_ratio'], 6)} "
                f"| {fmt(row['delta_E'])} |"
            )
    if warnings:
        lines.append("")
        lines.append("[Warnings]")
        for item in warnings:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("[Reminder]")
    lines.append(diagnostics["important_note"])

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n")
        f.write(json.dumps(diagnostics, indent=2))
        f.write("\n")

    print("\n".join(lines))
    print("\n[Saved]")
    print(f"  {args.out}")
    print(f"  {args.csv}")
    print(f"  {args.summary}")

    ok = (
        abs(E_obs_rdm - E_obs) < 1e-10
        and abs(E_ri_rdm - E_obs) < 1e-10
        and abs(diagnostics["rdm_diagnostics"]["obs"]["trace_dm1"] - 2.0) < 1e-10
        and abs(diagnostics["rdm_diagnostics"]["obs"]["trace_dm2"] - 2.0) < 1e-10
        and all(not d["has_nan"] and not d["has_inf"] for d in diagnostics["tensor_diagnostics"].values())
    )
    if not ok:
        print("\nERROR: Step 6f consistency checks failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
