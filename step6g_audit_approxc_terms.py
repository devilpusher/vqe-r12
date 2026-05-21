#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6g: Audit approximation-C/SP formula terms before final He energy.

This is a formula-correspondence script, not a final energy implementation.  It
maps the SR-MP2-F12 3C(FIX)/SP equations onto the current He parent-basis
ordered-pair tensors and prints every intermediate that affects tilde V, tilde
B, C_ab coupling, denominators, and SP prefactors.

Formula anchor: Psi4 MP2-F12 theory documentation, Eqs. (2), (3), (9)-(11).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from r12_common import (
    assert_finite,
    build_pair_fock_operator,
    maxabs,
    pair_matrix,
    pair_projector,
    reconstruct_energy,
    rdm_diagnostics,
    tensor_diagnostics,
)
from step6e_build_vxbc_intermediates import ansatz3_indices, default_prefix


FORMULA_SOURCE = "https://psicode.org/psi4manual/master/mp2f12.html"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="he_ccpvdz_nobs2_fitN7_step5a_r12_intermediates.npz")
    p.add_argument("--nocc", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--denom-thresh", type=float, default=1e-10)
    return p.parse_args()


def load_metadata(data) -> Dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    try:
        return json.loads(str(data["metadata_json"]))
    except Exception:
        return {}


def pair_index(p: int, q: int, n: int) -> int:
    return p * n + q


def pair_labels(indices: Iterable[int], n: int) -> list[list[int]]:
    return [[int(idx // n), int(idx % n)] for idx in indices]


def orbital_energy_audit(F: np.ndarray, nobs: int, nocc: int) -> Dict[str, Any]:
    eps_diag = np.diag(F).copy()
    offdiag = F - np.diag(eps_diag)
    evals = np.linalg.eigvalsh(0.5 * (F + F.T))
    return {
        "epsilon_source": "diag(F_ri) in the current RI orbital basis",
        "eps_diag": eps_diag.tolist(),
        "fock_offdiag_maxabs": maxabs(offdiag),
        "fock_offdiag_norm": float(np.linalg.norm(offdiag)),
        "fock_eigenvalues": evals.tolist(),
        "occupied": list(range(nocc)),
        "obs_virtual": list(range(nocc, nobs)),
        "ri_external": list(range(nocc, F.shape[0])),
        "warning": (
            "diag(F_ri) is a denominator audit convention. If F_ri is not sufficiently "
            "diagonal, a semicanonical rotation should be implemented before final energies."
        ),
    }


def denominator_table(eps: np.ndarray, i: int, j: int, ab_indices: np.ndarray, n: int, thresh: float) -> Dict[str, Any]:
    rows = []
    values = []
    for ab in ab_indices:
        a = int(ab // n)
        b = int(ab % n)
        den = float(eps[a] + eps[b] - eps[i] - eps[j])
        rows.append({"a": a, "b": b, "denominator": den, "near_zero": bool(abs(den) <= thresh)})
        values.append(den)
    arr = np.array(values, dtype=float)
    return {
        "ij": [int(i), int(j)],
        "rows": rows,
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
        "min_abs": float(np.min(np.abs(arr))) if arr.size else None,
        "n_near_zero": int(np.sum(np.abs(arr) <= thresh)) if arr.size else 0,
    }


def build_formula_matrices(data, nri: int, nobs: int, nocc: int) -> Dict[str, Any]:
    idx = ansatz3_indices(nri, nobs, nocc)
    P_rs = pair_projector(nri, idx["rs_obs"])
    P_aj = pair_projector(nri, idx["a_prime_j"])
    P_ib = pair_projector(nri, idx["i_b_prime"])
    P_removed = P_rs + P_aj + P_ib
    Q3 = pair_projector(nri, idx["q_ansatz3"])

    F_ri = np.array(data["F_ri"], dtype=float)
    K_pair = build_pair_fock_operator(F_ri)
    G = pair_matrix(np.array(data["eri_ri"], dtype=float))
    F12 = pair_matrix(np.array(data["f12_ri"], dtype=float))
    GF_direct = pair_matrix(np.array(data["f12g12_ri"], dtype=float))
    F2_direct = pair_matrix(np.array(data["f12sq_ri"], dtype=float))
    DC_direct = pair_matrix(np.array(data["f12dc_ri"], dtype=float))

    # Eq. (3) audit route: direct full integrals are authoritative for <gf>
    # and <f^2>; Ansatz-3 Q is applied by subtracting finite projector closures.
    V_q3 = GF_direct - G @ P_removed @ F12
    X_q3 = F2_direct - F12 @ P_removed @ F12
    C_q3 = F12 @ Q3 @ K_pair
    B_fock_q3 = F12 @ Q3 @ K_pair @ Q3 @ F12

    return {
        "indices": idx,
        "projectors": {
            "P_removed": P_removed,
            "Q3": Q3,
        },
        "matrices": {
            "G": G,
            "F12": F12,
            "GF_direct_full": GF_direct,
            "F2_direct_full": F2_direct,
            "DC_direct_full": DC_direct,
            "V_q3_projector_subtracted": V_q3,
            "X_q3_projector_subtracted": X_q3,
            "C_q3_fock_coupling": C_q3,
            "B_fock_q3": B_fock_q3,
            "B_dc_direct_full": DC_direct,
            "K_pair_fock": K_pair,
        },
    }


def ab_space_indices(nri: int, nobs: int, nocc: int) -> Dict[str, np.ndarray]:
    spaces: Dict[str, list[int]] = {"obs_virtual": [], "ri_external": []}
    obs_virt = set(range(nocc, nobs))
    ri_ext = set(range(nocc, nri))
    for a in range(nri):
        for b in range(nri):
            idx = pair_index(a, b, nri)
            if a in obs_virt and b in obs_virt:
                spaces["obs_virtual"].append(idx)
            if a in ri_ext and b in ri_ext:
                spaces["ri_external"].append(idx)
    return {key: np.array(value, dtype=int) for key, value in spaces.items()}


def make_unit_pair(nri: int, p: int, q: int, scale: float = 1.0) -> np.ndarray:
    v = np.zeros(nri * nri, dtype=float)
    v[pair_index(p, q, nri)] = scale
    return v


def build_tilde_terms(
    matrices: Dict[str, np.ndarray],
    eps: np.ndarray,
    i: int,
    j: int,
    kl_indices: np.ndarray,
    ab_indices: np.ndarray,
    n: int,
    thresh: float,
    B_source: str,
) -> Dict[str, Any]:
    V = matrices["V_q3_projector_subtracted"]
    X = matrices["X_q3_projector_subtracted"]
    C = matrices["C_q3_fock_coupling"]
    G = matrices["G"]
    B = matrices[B_source]
    eps_ij = float(eps[i] + eps[j])
    ij_idx = pair_index(i, j, n)

    V_block = V[np.ix_([ij_idx], kl_indices)].reshape(len(kl_indices))
    X_block = X[np.ix_(kl_indices, kl_indices)]
    B_block = B[np.ix_(kl_indices, kl_indices)]
    C_kl_ab = C[np.ix_(kl_indices, ab_indices)]
    G_ij_ab = G[np.ix_([ij_idx], ab_indices)].reshape(len(ab_indices))

    V_tilde = V_block.copy()
    B_tilde = B_block - eps_ij * X_block
    C_over_den_V = np.zeros_like(V_block)
    C_over_den_B = np.zeros_like(B_block)
    skipped = 0
    for col, ab in enumerate(ab_indices):
        a = int(ab // n)
        b = int(ab % n)
        den = float(eps[a] + eps[b] - eps[i] - eps[j])
        if abs(den) <= thresh:
            skipped += 1
            continue
        v_corr = C_kl_ab[:, col] * G_ij_ab[col] / den
        b_corr = np.outer(C_kl_ab[:, col], C_kl_ab[:, col]) / den
        C_over_den_V += v_corr
        C_over_den_B += b_corr
    V_tilde -= C_over_den_V
    B_tilde -= C_over_den_B

    return {
        "V_block": V_block,
        "X_block": X_block,
        "B_block": B_block,
        "C_kl_ab": C_kl_ab,
        "G_ij_ab": G_ij_ab,
        "C_over_den_V": C_over_den_V,
        "C_over_den_B": C_over_den_B,
        "V_tilde": V_tilde,
        "B_tilde": B_tilde,
        "eps_ij": eps_ij,
        "n_skipped_denominators": skipped,
    }


def energy_components(T: np.ndarray, terms: Dict[str, Any]) -> Dict[str, float]:
    Vt = terms["V_tilde"]
    Bt = terms["B_tilde"]
    linear = float(2.0 * (T @ Vt))
    quadratic = float(T @ (Bt @ T))
    return {
        "linear_2T_Vtilde": linear,
        "quadratic_T_Btilde_T": quadratic,
        "delta_E_candidate": linear + quadratic,
    }


def matrix_diag(name: str, M: np.ndarray) -> Dict[str, Any]:
    return {
        "name": name,
        "shape": list(M.shape),
        "norm": float(np.linalg.norm(M)),
        "max_abs": maxabs(M),
        "asym": maxabs(M - M.T) if M.ndim == 2 and M.shape[0] == M.shape[1] else None,
    }


def main():
    args = parse_args()
    prefix = default_prefix(args.inp)
    if args.out is None:
        args.out = f"{prefix}_step6g_approxc_term_audit.json"
    if args.summary is None:
        args.summary = f"{prefix}_step6g_approxc_term_audit_summary.txt"

    data = np.load(args.inp, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta.get("nobs", np.array(data["Cab_obs"]).shape[0]))
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))
    nocc = int(args.nocc)
    if nocc != 1:
        raise RuntimeError("Step 6g currently audits the He closed-shell pair i=j=0 only.")

    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)
    F_ri = np.array(data["F_ri"], dtype=float)
    enuc = float(meta.get("enuc", 0.0))
    E_obs = float(meta["E_obs_fci"])
    E_obs_rdm, _, _ = reconstruct_energy(h_ri[:nobs, :nobs], eri_ri[:nobs, :nobs, :nobs, :nobs], dm1_obs, dm2_obs, enuc)
    E_ri_rdm, _, _ = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)

    for name in ["F_ri", "h_ri", "eri_ri", "f12_ri", "f12sq_ri", "f12g12_ri", "f12dc_ri"]:
        assert_finite(name, np.array(data[name], dtype=float))

    built = build_formula_matrices(data, nri, nobs, nocc)
    M = built["matrices"]
    eps_info = orbital_energy_audit(F_ri, nobs, nocc)
    eps = np.array(eps_info["eps_diag"], dtype=float)
    spaces = ab_space_indices(nri, nobs, nocc)
    kl_indices = np.array([pair_index(k, l, nri) for k in range(nocc) for l in range(nocc)], dtype=int)
    i = j = 0

    T_full = make_unit_pair(nocc, 0, 0, 0.5)
    T_direct = make_unit_pair(nocc, 0, 0, 3.0 / 8.0)
    T_exchange = make_unit_pair(nocc, 0, 0, 1.0 / 8.0)
    T_unit = make_unit_pair(nocc, 0, 0, 1.0)
    T_variants = {
        "sp_3_8_plus_1_8_for_i_eq_j": T_full,
        "direct_3_8_only": T_direct,
        "exchange_1_8_only": T_exchange,
        "unit_delta_00": T_unit,
    }

    audits: Dict[str, Any] = {}
    for ab_name, ab_idx in spaces.items():
        audits[ab_name] = {
            "ab_pair_indices": ab_idx.tolist(),
            "ab_pair_labels": pair_labels(ab_idx, nri),
            "denominators": denominator_table(eps, i, j, ab_idx, nri, args.denom_thresh),
            "B_source_variants": {},
        }
        for B_source in ["B_fock_q3", "B_dc_direct_full"]:
            terms = build_tilde_terms(M, eps, i, j, kl_indices, ab_idx, nri, args.denom_thresh, B_source)
            energy_rows = {}
            for tname, T in T_variants.items():
                energy_rows[tname] = energy_components(T, terms)
            audits[ab_name]["B_source_variants"][B_source] = {
                "terms": {
                    "V_q3_block": terms["V_block"].tolist(),
                    "C_over_den_V": terms["C_over_den_V"].tolist(),
                    "V_tilde": terms["V_tilde"].tolist(),
                    "X_q3_block_norm": float(np.linalg.norm(terms["X_block"])),
                    "B_block_norm": float(np.linalg.norm(terms["B_block"])),
                    "C_kl_ab_norm": float(np.linalg.norm(terms["C_kl_ab"])),
                    "G_ij_ab_norm": float(np.linalg.norm(terms["G_ij_ab"])),
                    "C_over_den_B_norm": float(np.linalg.norm(terms["C_over_den_B"])),
                    "B_tilde_norm": float(np.linalg.norm(terms["B_tilde"])),
                    "eps_ij": terms["eps_ij"],
                    "n_skipped_denominators": terms["n_skipped_denominators"],
                },
                "energy_rows": energy_rows,
            }

    direct_vs_closure = {
        "maxabs_direct_f12g12_minus_G_F12": maxabs(M["GF_direct_full"] - M["G"] @ M["F12"]),
        "maxabs_direct_f12sq_minus_F12_F12": maxabs(M["F2_direct_full"] - M["F12"] @ M["F12"]),
        "maxabs_V_q3_minus_G_Q3_F12_closure": maxabs(M["V_q3_projector_subtracted"] - M["G"] @ built["projectors"]["Q3"] @ M["F12"]),
        "maxabs_X_q3_minus_F12_Q3_F12_closure": maxabs(M["X_q3_projector_subtracted"] - M["F12"] @ built["projectors"]["Q3"] @ M["F12"]),
    }

    diagnostics = {
        "input": args.inp,
        "formula_source": FORMULA_SOURCE,
        "formula_map": {
            "V": "<ij| r12^-1 Q12 f12 |kl>",
            "X": "<kl| f12 Q12 f12 |mn>",
            "C": "<kl| f12 Q12 (F1+F2) |ab>",
            "B": "<kl| f12 Q12 (F1+F2) Q12 f12 |mn>",
            "V_tilde": "V - sum_ab C_ab^kl G_ab^ij / (eps_a+eps_b-eps_i-eps_j)",
            "B_tilde": "B - (eps_i+eps_j) X - sum_ab C_ab^mn C_ab^kl / denominator",
            "energy": "DeltaE = T_kl^ij (2 Vtilde_kl^ij + Btilde_kl,mn^ij T_mn^ij)",
        },
        "nri": nri,
        "nobs": nobs,
        "nocc": nocc,
        "energy_checks": {
            "E_obs_fci": E_obs,
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
            "f12_ri": tensor_diagnostics(np.array(data["f12_ri"], dtype=float)),
            "f12sq_ri": tensor_diagnostics(np.array(data["f12sq_ri"], dtype=float)),
            "f12g12_ri": tensor_diagnostics(np.array(data["f12g12_ri"], dtype=float)),
            "f12dc_ri": tensor_diagnostics(np.array(data["f12dc_ri"], dtype=float)),
        },
        "projector_dimensions": {key: int(len(value)) for key, value in built["indices"].items()},
        "occupied_pair_indices_kl_mn": kl_indices.tolist(),
        "occupied_pair_labels_kl_mn": pair_labels(kl_indices, nri),
        "orbital_energy_audit": eps_info,
        "direct_vs_closure": direct_vs_closure,
        "matrix_diagnostics": {
            key: matrix_diag(key, value)
            for key, value in M.items()
            if value.ndim == 2
        },
        "audits": audits,
        "important_note": (
            "This audit uses direct <gf>, <f^2>, and f12dc tensors as authoritative integral "
            "inputs, but applies Ansatz-3 Q explicitly through projector subtraction/insertions. "
            "Rows using B_dc_direct_full are included only to compare with the double-commutator "
            "diagnostic; the formula B object is B_fock_q3."
        ),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6g | approximation-C/SP term audit")
    lines.append("=" * 100)
    lines.append(f"input       = {args.inp}")
    lines.append(f"formula ref = {FORMULA_SOURCE}")
    lines.append(f"nri/nobs/nocc = {nri}/{nobs}/{nocc}")
    lines.append(f"E checks    = obs-rdm {E_obs_rdm - E_obs:.3e}, ri-rdm {E_ri_rdm - E_obs:.3e}")
    lines.append("")
    lines.append("[Formula map]")
    for key, value in diagnostics["formula_map"].items():
        lines.append(f"{key:8s}: {value}")
    lines.append(f"occupied kl/mn pair labels audited: {diagnostics['occupied_pair_labels_kl_mn']}")
    lines.append("")
    lines.append("[Orbital denominator audit]")
    lines.append(f"epsilon source       = {eps_info['epsilon_source']}")
    lines.append(f"max |F_offdiag|      = {eps_info['fock_offdiag_maxabs']:.3e}")
    lines.append(f"obs virtual indices  = {eps_info['obs_virtual']}")
    lines.append(f"ri external indices  = {eps_info['ri_external']}")
    for ab_name, audit in audits.items():
        den = audit["denominators"]
        lines.append(
            f"{ab_name:12s}: pairs={audit['ab_pair_labels']}, "
            f"min={den['min']:.8e}, max={den['max']:.8e}, min_abs={den['min_abs']:.8e}, "
            f"near_zero={den['n_near_zero']}"
        )
    lines.append("")
    lines.append("[Direct tensor vs finite closure]")
    for key, value in direct_vs_closure.items():
        lines.append(f"{key}: {value:.3e}")
    lines.append("")
    lines.append("[Tilde-term energy audit]")
    lines.append("| ab space | B source | T variant | linear 2T.Vt | quad T.Bt.T | DeltaE |")
    lines.append("|---|---|---|---:|---:|---:|")
    for ab_name, audit in audits.items():
        for B_source, block in audit["B_source_variants"].items():
            for tname, row in block["energy_rows"].items():
                lines.append(
                    f"| {ab_name} | {B_source} | {tname} "
                    f"| {row['linear_2T_Vtilde']:.8e} "
                    f"| {row['quadratic_T_Btilde_T']:.8e} "
                    f"| {row['delta_E_candidate']:.8e} |"
                )
    lines.append("")
    lines.append("[Term norms]")
    for ab_name, audit in audits.items():
        for B_source, block in audit["B_source_variants"].items():
            t = block["terms"]
            lines.append(
                f"{ab_name}/{B_source}: "
                f"|C_kl_ab|={t['C_kl_ab_norm']:.3e}, |G_ij_ab|={t['G_ij_ab_norm']:.3e}, "
                f"|C/den V|={np.linalg.norm(np.array(t['C_over_den_V'])):.3e}, "
                f"|C/den B|={t['C_over_den_B_norm']:.3e}, |Btilde|={t['B_tilde_norm']:.3e}"
            )
    lines.append("")
    lines.append("[Decision]")
    lines.append(diagnostics["important_note"])
    lines.append("Before changing Step 6f, inspect whether ab space should be OBS virtual only or RI external, and whether F_ri must be semicanonicalized.")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n")
        f.write(json.dumps(diagnostics, indent=2))
        f.write("\n")

    print("\n".join(lines))
    print("\n[Saved]")
    print(f"  {args.out}")
    print(f"  {args.summary}")

    ok = (
        abs(E_obs_rdm - E_obs) < 1e-10
        and abs(E_ri_rdm - E_obs) < 1e-10
        and abs(diagnostics["rdm_diagnostics"]["obs"]["trace_dm1"] - 2.0) < 1e-10
        and abs(diagnostics["rdm_diagnostics"]["obs"]["trace_dm2"] - 2.0) < 1e-10
        and all(not d["has_nan"] and not d["has_inf"] for d in diagnostics["tensor_diagnostics"].values())
    )
    if not ok:
        print("\nERROR: Step 6g consistency checks failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
