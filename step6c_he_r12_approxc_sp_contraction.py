#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6c: He parent-basis approximation-C/SP contraction prototype.

This is the first modular contraction script after the fitted-Slater scans.
It reads a validated Step-5a intermediate file and evaluates SP/Q-projected
F12 contraction diagnostics with several denominator models.

Important
---------
This script is not yet a final article-level [2]R12 implementation.  It is a
controlled He/parent-basis prototype for wiring the approximation-C/SP data
path:

* parent-basis F12 tensors only;
* SP-transformed, Q-projected F12 pair amplitudes from Step 5a;
* energy reconstruction and RDM checks retained;
* comparison of full pair-Hamiltonian and generalized-Fock denominators.

The full Q-pair result is still an external-pair diagnostic reference, not a
published [2]R12 correction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from r12_common import (
    assert_finite,
    build_pair_fock_operator,
    build_pair_hamiltonian,
    maxabs,
    pair_matrix,
    pair_projector,
    pp_pair_indices,
    q_pair_indices,
    reconstruct_energy,
    rdm_diagnostics,
    sym,
    tensor_diagnostics,
    two_body_expectation,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="he_ccpvdz_nobs2_fitN7_step5a_r12_intermediates.npz")
    p.add_argument("--out", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--nocc", type=int, default=1, help="Closed-shell occupied spatial orbitals for SR-F12 formula audit.")
    p.add_argument("--eig-thresh", type=float, default=1e-10)
    p.add_argument("--max-pair-dim", type=int, default=5000)
    return p.parse_args()


def load_metadata(data) -> Dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    try:
        return json.loads(str(data["metadata_json"]))
    except Exception:
        return {}


def default_prefix(inp: str) -> str:
    name = Path(inp).name
    suffix = "_step5a_r12_intermediates.npz"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return Path(inp).stem


def one_vector_contraction(operator: np.ndarray, psi0: np.ndarray, avec: np.ndarray, e0: float, thresh: float) -> Dict[str, Any]:
    shifted = sym(operator - e0 * np.eye(operator.shape[0]))
    q = float(avec @ (shifted @ avec))
    c = float(psi0 @ (operator @ avec))
    fixed = q + 2.0 * c
    if abs(q) > thresh:
        lam = -c / q
        delta = -c * c / q
    else:
        lam = None
        delta = None
    return {
        "norm_A": float(np.linalg.norm(avec)),
        "quadratic": q,
        "coupling": c,
        "fixed_amplitude_J": fixed,
        "lambda_opt": lam,
        "deltaE_1D_opt": delta,
        "q_positive": bool(q > thresh),
    }


def full_q_solve(operator: np.ndarray, psi0: np.ndarray, qidx: np.ndarray, e0: float, thresh: float) -> Dict[str, Any]:
    OQQ = operator[np.ix_(qidx, qidx)]
    A = sym(OQQ - e0 * np.eye(len(qidx)))
    b = operator[np.ix_(qidx, np.arange(operator.shape[0]))] @ psi0
    evals, U = np.linalg.eigh(A)
    n_bad = int(np.sum(evals <= thresh))
    if n_bad == 0:
        x = np.linalg.solve(A, b)
        method = "solve"
    else:
        keep = np.abs(evals) > thresh
        inv = np.zeros_like(evals)
        inv[keep] = 1.0 / evals[keep]
        x = U @ (inv * (U.T @ b))
        method = "eigen_pseudoinverse_signed"
    delta = -float(b @ x)
    residual = float(np.linalg.norm(A @ x - b))
    diag = np.diag(A)
    valid = np.abs(diag) > thresh
    delta_diag = -float(np.sum((b[valid] ** 2) / diag[valid]))
    return {
        "method": method,
        "q_dim": int(len(qidx)),
        "deltaE_full_Q": delta,
        "deltaE_diag": delta_diag,
        "residual_norm": residual,
        "norm_b": float(np.linalg.norm(b)),
        "norm_x": float(np.linalg.norm(x)),
        "min_eig": float(np.min(evals)) if evals.size else None,
        "max_eig": float(np.max(evals)) if evals.size else None,
        "n_eig_le_thresh": n_bad,
        "x_Q": x,
        "b_Q": b,
    }


def vector_alignment(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    cos = None if na == 0.0 or nb == 0.0 else float((a @ b) / (na * nb))
    return {"norm_a": na, "norm_b": nb, "dot": float(a @ b), "cosine": cos}


def ansatz3_projector_masks(nri: int, nobs: int, nocc: int) -> Dict[str, Any]:
    """Build ordered-pair masks matching the Psi4 Ansatz-3 projector labels.

    For closed-shell SR-F12 notation:
        i,j = occupied OBS orbitals
        a,b = virtual OBS orbitals
        a',b' = CABS/external RI complement
        r,s = all OBS orbitals

    Psi4 Eq. (6): Q12 = 1 - |a'j><a'j| - |ib'><ib'| - |rs><rs|.
    """
    pairs = [(p, q) for p in range(nri) for q in range(nri)]
    obs = set(range(nobs))
    occ = set(range(nocc))
    cabs = set(range(nobs, nri))

    rs_obs = []
    cabs_occ_left = []
    occ_cabs_right = []
    q_ansatz3 = []
    for idx, (p, q) in enumerate(pairs):
        is_rs = p in obs and q in obs
        is_a_prime_j = p in cabs and q in occ
        is_i_b_prime = p in occ and q in cabs
        if is_rs:
            rs_obs.append(idx)
        if is_a_prime_j:
            cabs_occ_left.append(idx)
        if is_i_b_prime:
            occ_cabs_right.append(idx)
        if not (is_rs or is_a_prime_j or is_i_b_prime):
            q_ansatz3.append(idx)

    return {
        "rs_obs": np.array(rs_obs, dtype=int),
        "a_prime_j": np.array(cabs_occ_left, dtype=int),
        "i_b_prime": np.array(occ_cabs_right, dtype=int),
        "q_ansatz3": np.array(q_ansatz3, dtype=int),
        "dimensions": {
            "nri_pair_dim": nri * nri,
            "nobs": nobs,
            "nocc": nocc,
            "n_obs_virtual": max(nobs - nocc, 0),
            "n_cabs": nri - nobs,
            "n_rs_obs": len(rs_obs),
            "n_a_prime_j": len(cabs_occ_left),
            "n_i_b_prime": len(occ_cabs_right),
            "n_q_ansatz3": len(q_ansatz3),
        },
    }


def projector_subtraction_diagnostics(
    eri_ri: np.ndarray,
    f12_ri: np.ndarray,
    f12sq_ri: np.ndarray,
    f12g12_ri: np.ndarray,
    nri: int,
    nobs: int,
    nocc: int,
) -> Dict[str, Any]:
    """Compare direct 3C projector subtraction patterns with Psi4 direct integrals."""
    masks = ansatz3_projector_masks(nri, nobs, nocc)
    G = pair_matrix(eri_ri)
    F = pair_matrix(f12_ri)
    F2 = pair_matrix(f12sq_ri)
    GF = pair_matrix(f12g12_ri)

    def P(label: str) -> np.ndarray:
        return pair_projector(nri, masks[label])

    P_a_j = P("a_prime_j")
    P_i_b = P("i_b_prime")
    P_rs = P("rs_obs")
    P_q3 = pair_projector(nri, masks["q_ansatz3"])

    V_3c_matrix = GF - G @ P_a_j @ F - G @ P_i_b @ F - G @ P_rs @ F
    X_3c_matrix = F2 - F @ P_i_b @ F - F @ P_a_j @ F - F @ P_rs @ F
    V_q3_closure = G @ P_q3 @ F
    X_q3_closure = F @ P_q3 @ F

    return {
        "masks": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in masks.items()
        },
        "errors": {
            "maxabs_V_3c_matrix_minus_gQf_closure": maxabs(V_3c_matrix - V_q3_closure),
            "maxabs_X_3c_matrix_minus_fQf_closure": maxabs(X_3c_matrix - X_q3_closure),
            "maxabs_gf_direct_minus_gf_closure_full": maxabs(GF - G @ F),
            "maxabs_f2_direct_minus_ff_closure_full": maxabs(F2 - F @ F),
        },
        "norms": {
            "V_3c_matrix": float(np.linalg.norm(V_3c_matrix)),
            "X_3c_matrix": float(np.linalg.norm(X_3c_matrix)),
            "V_q3_closure": float(np.linalg.norm(V_q3_closure)),
            "X_q3_closure": float(np.linalg.norm(X_q3_closure)),
        },
        "matrices": {
            "V_3c_matrix": V_3c_matrix,
            "X_3c_matrix": X_3c_matrix,
            "V_q3_closure": V_q3_closure,
            "X_q3_closure": X_q3_closure,
        },
    }


def formula_audit_summary() -> Dict[str, Any]:
    return {
        "source": "Psi4 MP2-F12 theory documentation, equations (2)-(11)",
        "projector_ansatz3": "Q12 = 1 - |a'j><a'j| - |ib'><ib'| - |rs><rs|",
        "sp_fixed_amplitude": "T_kl^ij = 3/8 delta_ik delta_jl + 1/8 delta_jk delta_il",
        "residual": "R_kl^ij = V_kl^ij + C_ab^kl T_ab^ij + [B_kl,mn - (eps_i+eps_j) X_kl,mn] T_mn^ij",
        "energy": "Delta E_F12/3C(FIX) = T_kl^ij (2 Vtilde_kl^ij + Btilde_kl,mn^ij T_mn^ij)",
        "tilde_terms": {
            "Vtilde": "V - C_ab^kl G_ab^ij / (eps_a + eps_b - eps_i - eps_j)",
            "Btilde": "B - (eps_i+eps_j)X - C_ab^mn C_ab^kl / (eps_a + eps_b - eps_i - eps_j)",
        },
        "current_limitations": [
            "Current He prototype uses OBS-FCI/RDM pair data, not a strict SR-MP2 occupied-pair residual.",
            "C_ab coupling and orbital-energy denominator terms are not fully implemented yet.",
            "Direct Psi4 f12g12/f12sq tensors are not equivalent to finite RI matrix closure G@F/F@F.",
        ],
    }


def f12_intermediate_contractions(
    eri_ri: np.ndarray,
    f12_ri: np.ndarray,
    f12sq_ri: np.ndarray,
    f12g12_ri: np.ndarray,
    f12dc_ri: np.ndarray,
    psi0_vec: np.ndarray,
    cab_sp_vec: np.ndarray,
    nobs: int,
    nri: int,
    thresh: float,
) -> Dict[str, Any]:
    """Build approximation-C-like F12 intermediates in ordered pair space.

    Direct-integral route:
        V  <- <source| f12g12 |source>
        X  <- <source| f12sq  |source>
        DC <- <source| f12dc  |source>

    Matrix-closure route:
        g Q f and f Q f are also reported as diagnostics.  They are not assumed
        to equal the direct Psi4 integral tensors; the differences are useful
        convention/completeness checks for this finite RI prototype.
    """
    pidx = pp_pair_indices(nri, nobs)
    qidx = q_pair_indices(nri, nobs)
    P = pair_projector(nri, pidx)
    Q = pair_projector(nri, qidx)

    G = pair_matrix(eri_ri)
    F = pair_matrix(f12_ri)
    F2 = pair_matrix(f12sq_ri)
    GF_direct = pair_matrix(f12g12_ri)
    DC = pair_matrix(f12dc_ri)

    V_Q_matrix = G @ Q @ F
    X_Q_matrix = F @ Q @ F
    V_full_matrix = G @ F
    X_full_matrix = F @ F

    def scalar(M: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        return float(left @ (M @ right))

    out: Dict[str, Any] = {
        "matrix_consistency": {
            "maxabs_GF_matrix_minus_f12g12_integral": maxabs(V_full_matrix - GF_direct),
            "maxabs_FF_matrix_minus_f12sq_integral": maxabs(X_full_matrix - F2),
            "maxabs_DC_asym": maxabs(DC - DC.T),
        },
        "reference": {},
        "sp": {},
    }

    for label, source in [("reference", psi0_vec), ("sp", cab_sp_vec)]:
        V = scalar(GF_direct, source, source)
        X = scalar(F2, source, source)
        DCq = scalar(DC, source, source)
        V_matrix_Q = scalar(V_Q_matrix, psi0_vec, source)
        X_matrix_Q = scalar(X_Q_matrix, source, source)
        B_plus = X + DCq
        B_minus = X - DCq
        B_dc_only = DCq

        rows = {
            "V_f12g12_direct": V,
            "X_f12sq_direct": X,
            "B_f12dc_direct": DCq,
            "V_gQf_matrix_closure": V_matrix_Q,
            "X_fQf_matrix_closure": X_matrix_Q,
            "B_X_plus_dc": B_plus,
            "B_X_minus_dc": B_minus,
            "B_dc_only": B_dc_only,
        }
        for denom_name, denom in [
            ("X_plus_dc", B_plus),
            ("X_minus_dc", B_minus),
            ("dc_only", B_dc_only),
            ("X_only", X),
        ]:
            if abs(denom) > thresh:
                rows[f"DeltaE_minus_V2_over_{denom_name}"] = -V * V / denom
            else:
                rows[f"DeltaE_minus_V2_over_{denom_name}"] = None
        out[label] = rows

    return out


def main():
    args = parse_args()
    prefix = default_prefix(args.inp)
    if args.out is None:
        args.out = f"{prefix}_step6c_approxc_sp_contraction.npz"
    if args.summary is None:
        args.summary = f"{prefix}_step6c_approxc_sp_contraction_summary.txt"

    data = np.load(args.inp, allow_pickle=True)
    meta = load_metadata(data)

    required = [
        "F_ri", "Cab_obs", "A_raw_Q", "A_sp_Q",
        "h_ri", "eri_ri", "f12_ri", "f12sq_ri", "f12g12_ri", "f12dc_ri",
        "dm1_obs", "dm2_obs", "dm1_ri", "dm2_ri",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"Missing required arrays in {args.inp}: {missing}")

    F_ri = np.array(data["F_ri"], dtype=float)
    Cab_obs = np.array(data["Cab_obs"], dtype=float)
    A_raw_Q = np.array(data["A_raw_Q"], dtype=float)
    A_sp_Q = np.array(data["A_sp_Q"], dtype=float)
    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    f12_ri = np.array(data["f12_ri"], dtype=float)
    f12sq_ri = np.array(data["f12sq_ri"], dtype=float)
    f12g12_ri = np.array(data["f12g12_ri"], dtype=float)
    f12dc_ri = np.array(data["f12dc_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)

    nobs = int(meta.get("nobs", Cab_obs.shape[0]))
    nri = h_ri.shape[0]
    pair_dim = nri * nri
    if pair_dim > args.max_pair_dim:
        raise RuntimeError(f"pair_dim={pair_dim} exceeds --max-pair-dim={args.max_pair_dim}")

    enuc = float(meta.get("enuc", 0.0))
    E_obs_fci = float(meta["E_obs_fci"])
    E_ref_elec = float(meta["E_ref_elec"])
    E_obs_rdm, E1_obs, E2_obs = reconstruct_energy(h_ri[:nobs, :nobs], eri_ri[:nobs, :nobs, :nobs, :nobs], dm1_obs, dm2_obs, enuc)
    E_ri_rdm, E1_ri, E2_ri = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)

    arrays = {
        "F_ri": F_ri,
        "A_raw_Q": A_raw_Q,
        "A_sp_Q": A_sp_Q,
        "h_ri": h_ri,
        "eri_ri": eri_ri,
        "f12_ri": f12_ri,
        "f12sq_ri": f12sq_ri,
        "f12g12_ri": f12g12_ri,
        "f12dc_ri": f12dc_ri,
        "dm1_ri": dm1_ri,
        "dm2_ri": dm2_ri,
    }
    for name, arr in arrays.items():
        assert_finite(name, arr)

    psi0 = np.zeros((nri, nri), dtype=float)
    psi0[:nobs, :nobs] = Cab_obs
    psi0_vec = psi0.reshape(-1)
    cab_sp = np.zeros((nri, nri), dtype=float)
    if "Cab_sp" in data:
        cab_sp[:nobs, :nobs] = np.array(data["Cab_sp"], dtype=float)
    else:
        cab_sp[:nobs, :nobs] = Cab_obs
    cab_sp_vec = cab_sp.reshape(-1)
    A_raw_vec = A_raw_Q.reshape(-1)
    A_sp_vec = A_sp_Q.reshape(-1)

    H_pair = build_pair_hamiltonian(h_ri, eri_ri)
    K_pair = build_pair_fock_operator(F_ri)
    qidx = q_pair_indices(nri, nobs)
    sr_formula_audit = formula_audit_summary()
    projector_audit = projector_subtraction_diagnostics(
        eri_ri=eri_ri,
        f12_ri=f12_ri,
        f12sq_ri=f12sq_ri,
        f12g12_ri=f12g12_ri,
        nri=nri,
        nobs=nobs,
        nocc=args.nocc,
    )

    # Full-H is the Step-5b reference.  Fock-denominator rows are the new 6c
    # approximation-C/SP contraction prototype.
    contractions = {
        "raw_full_pair_H": one_vector_contraction(H_pair, psi0_vec, A_raw_vec, E_ref_elec, args.eig_thresh),
        "sp_full_pair_H": one_vector_contraction(H_pair, psi0_vec, A_sp_vec, E_ref_elec, args.eig_thresh),
        "raw_fock_pair": one_vector_contraction(K_pair, psi0_vec, A_raw_vec, E_ref_elec, args.eig_thresh),
        "sp_fock_pair": one_vector_contraction(K_pair, psi0_vec, A_sp_vec, E_ref_elec, args.eig_thresh),
    }

    qsolve_H = full_q_solve(H_pair, psi0_vec, qidx, E_ref_elec, args.eig_thresh)
    qsolve_F = full_q_solve(K_pair, psi0_vec, qidx, E_ref_elec, args.eig_thresh)

    xH = np.zeros(pair_dim, dtype=float)
    xH[qidx] = qsolve_H["x_Q"]
    xF = np.zeros(pair_dim, dtype=float)
    xF[qidx] = qsolve_F["x_Q"]

    f12_expectations = {
        "eri": two_body_expectation(eri_ri, dm2_ri),
        "f12": two_body_expectation(f12_ri, dm2_ri),
        "f12sq": two_body_expectation(f12sq_ri, dm2_ri),
        "f12g12": two_body_expectation(f12g12_ri, dm2_ri),
        "f12dc": two_body_expectation(f12dc_ri, dm2_ri),
    }

    f12_approxc = f12_intermediate_contractions(
        eri_ri=eri_ri,
        f12_ri=f12_ri,
        f12sq_ri=f12sq_ri,
        f12g12_ri=f12g12_ri,
        f12dc_ri=f12dc_ri,
        psi0_vec=psi0_vec,
        cab_sp_vec=cab_sp_vec,
        nobs=nobs,
        nri=nri,
        thresh=args.eig_thresh,
    )

    corrected_total_energies = {}
    for key, value in contractions.items():
        delta = value["deltaE_1D_opt"]
        corrected_total_energies[key] = None if delta is None else E_obs_fci + delta
    corrected_total_energies["full_Q_pair_H_reference"] = E_obs_fci + qsolve_H["deltaE_full_Q"]
    corrected_total_energies["diag_Q_pair_H_reference"] = E_obs_fci + qsolve_H["deltaE_diag"]
    corrected_total_energies["full_Q_pair_Fock_model"] = E_obs_fci + qsolve_F["deltaE_full_Q"]
    corrected_total_energies["diag_Q_pair_Fock_model"] = E_obs_fci + qsolve_F["deltaE_diag"]

    diagnostics = {
        "input": args.inp,
        "nobs": nobs,
        "nri": nri,
        "pair_dim": pair_dim,
        "enuc": enuc,
        "E_obs_fci": E_obs_fci,
        "E_ref_elec": E_ref_elec,
        "energy_checks": {
            "E_obs_rdm": E_obs_rdm,
            "E_ri_embedded_rdm": E_ri_rdm,
            "delta_obs_rdm_minus_fci": E_obs_rdm - E_obs_fci,
            "delta_ri_rdm_minus_fci": E_ri_rdm - E_obs_fci,
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
            "eri_ri": tensor_diagnostics(eri_ri),
            "f12_ri": tensor_diagnostics(f12_ri),
            "f12sq_ri": tensor_diagnostics(f12sq_ri),
            "f12g12_ri": tensor_diagnostics(f12g12_ri),
            "f12dc_ri": tensor_diagnostics(f12dc_ri),
        },
        "f12_expectations": f12_expectations,
        "sr_formula_audit": sr_formula_audit,
        "projector_subtraction_audit": {
            key: value for key, value in projector_audit.items()
            if key != "matrices"
        },
        "f12_approxc_intermediates": f12_approxc,
        "contractions": contractions,
        "qsolve_reference": {k: v for k, v in qsolve_H.items() if k not in ["x_Q", "b_Q"]},
        "qsolve_fock_model": {k: v for k, v in qsolve_F.items() if k not in ["x_Q", "b_Q"]},
        "alignment": {
            "A_sp_Q_vs_full_H_solution": vector_alignment(A_sp_vec, xH),
            "A_sp_Q_vs_fock_solution": vector_alignment(A_sp_vec, xF),
            "A_raw_Q_vs_full_H_solution": vector_alignment(A_raw_vec, xH),
            "A_raw_Q_vs_fock_solution": vector_alignment(A_raw_vec, xF),
        },
        "corrected_total_energies": corrected_total_energies,
        "important_note": (
            "Step 6c is an approximation-C/SP contraction prototype. "
            "Rows using the full pair Hamiltonian are retained as diagnostics. "
            "Rows using F(1)+F(2) are the current generalized-Fock denominator model. "
            "The f12_approxc_intermediates block now includes gQf, fQf, and fdc contractions "
            "built from f12g12, f12sq, and f12dc tensors; final prefactors/exchange terms "
            "still need literature-level auditing. "
            "This is not yet the final article-level [2]R12 tensor formula."
        ),
    }

    np.savez(
        args.out,
        H_pair=H_pair,
        K_pair_fock=K_pair,
        V_3c_projector_matrix=projector_audit["matrices"]["V_3c_matrix"],
        X_3c_projector_matrix=projector_audit["matrices"]["X_3c_matrix"],
        V_q3_closure_matrix=projector_audit["matrices"]["V_q3_closure"],
        X_q3_closure_matrix=projector_audit["matrices"]["X_q3_closure"],
        qidx=qidx,
        psi0_pair=psi0,
        Cab_sp_pair=cab_sp,
        A_raw_Q=A_raw_Q,
        A_sp_Q=A_sp_Q,
        x_full_H=xH.reshape(nri, nri),
        x_fock=xF.reshape(nri, nri),
        metadata_json=np.array(json.dumps(diagnostics, indent=2)),
    )

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6c | approximation-C/SP contraction prototype")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"input       = {args.inp}")
    lines.append(f"nobs/nri    = {nobs}/{nri}")
    lines.append(f"E_OBS-FCI   = {E_obs_fci:.14f} Eh")
    lines.append(f"E_OBS-RDM   = {E_obs_rdm:.14f} Eh")
    lines.append(f"E_RI-RDM    = {E_ri_rdm:.14f} Eh")
    lines.append("")
    lines.append("[One-vector contractions]")
    lines.append("| model | q | c | DeltaE opt | E total |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, d in contractions.items():
        e_tot = corrected_total_energies[key]
        lines.append(
            f"| {key} | {d['quadratic']:.8e} | {d['coupling']:.8e} | "
            f"{d['deltaE_1D_opt']:.8e} | {e_tot:.12f} |"
        )
    lines.append("")
    lines.append("[Q-space reference/model solves]")
    lines.append(f"full H Q solve DeltaE = {qsolve_H['deltaE_full_Q']:.8e}, residual={qsolve_H['residual_norm']:.3e}")
    lines.append(f"Fock Q solve DeltaE   = {qsolve_F['deltaE_full_Q']:.8e}, residual={qsolve_F['residual_norm']:.3e}")
    lines.append(f"H diag EN-like DeltaE = {qsolve_H['deltaE_diag']:.8e}")
    lines.append(f"F diag EN-like DeltaE = {qsolve_F['deltaE_diag']:.8e}")
    lines.append("")
    lines.append("[F12 expectations]")
    for key, value in f12_expectations.items():
        lines.append(f"<{key}> = {value:.12e}")
    lines.append("")
    lines.append("[Approximation-C-like F12 intermediate contractions]")
    chk = f12_approxc["matrix_consistency"]
    lines.append(f"max|g*f matrix closure - f12g12 integral| = {chk['maxabs_GF_matrix_minus_f12g12_integral']:.3e}")
    lines.append(f"max|f*f matrix closure - f12sq integral|  = {chk['maxabs_FF_matrix_minus_f12sq_integral']:.3e}")
    lines.append("| source | V=<f12g12> | X=<f12sq> | B=<f12dc> | -V^2/(X+dc) | -V^2/(X-dc) | -V^2/dc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in ["reference", "sp"]:
        block = f12_approxc[label]
        def fmt(x):
            return "" if x is None else f"{x:.8e}"
        lines.append(
            f"| {label} | {block['V_f12g12_direct']:.8e} | {block['X_f12sq_direct']:.8e} | {block['B_f12dc_direct']:.8e} "
            f"| {fmt(block['DeltaE_minus_V2_over_X_plus_dc'])} "
            f"| {fmt(block['DeltaE_minus_V2_over_X_minus_dc'])} "
            f"| {fmt(block['DeltaE_minus_V2_over_dc_only'])} |"
        )
    lines.append("")
    lines.append("[Formula audit: SR-MP2-F12 3C(FIX)/SP]")
    lines.append(f"source      : {sr_formula_audit['source']}")
    lines.append(f"projector   : {sr_formula_audit['projector_ansatz3']}")
    lines.append(f"SP amplitude: {sr_formula_audit['sp_fixed_amplitude']}")
    dims = projector_audit["masks"]["dimensions"]
    lines.append(
        "projector dimensions: "
        f"pair={dims['nri_pair_dim']}, OBS={dims['nobs']}, occ={dims['nocc']}, "
        f"CABS={dims['n_cabs']}, Q3={dims['n_q_ansatz3']}"
    )
    pe = projector_audit["errors"]
    lines.append(f"max|V_3C matrix - gQ3f closure| = {pe['maxabs_V_3c_matrix_minus_gQf_closure']:.3e}")
    lines.append(f"max|X_3C matrix - fQ3f closure| = {pe['maxabs_X_3c_matrix_minus_fQf_closure']:.3e}")
    lines.append(f"max|direct f12g12 - full g*f closure| = {pe['maxabs_gf_direct_minus_gf_closure_full']:.3e}")
    lines.append(f"max|direct f12sq  - full f*f closure| = {pe['maxabs_f2_direct_minus_ff_closure_full']:.3e}")
    lines.append("missing terms before final 3C(FIX): C_ab coupling/orbital-energy denominator and exact tilde prefactors.")
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
    print(f"  {args.summary}")

    ok = (
        abs(E_obs_rdm - E_obs_fci) < 1e-10
        and abs(E_ri_rdm - E_obs_fci) < 1e-10
        and abs(diagnostics["rdm_diagnostics"]["obs"]["trace_dm1"] - 2.0) < 1e-10
        and abs(diagnostics["rdm_diagnostics"]["obs"]["trace_dm2"] - 2.0) < 1e-10
        and all(not d["has_nan"] and not d["has_inf"] for d in diagnostics["tensor_diagnostics"].values())
        and np.isfinite(qsolve_H["deltaE_full_Q"])
        and np.isfinite(qsolve_F["deltaE_full_Q"])
    )
    if not ok:
        print("\nERROR: Step 6c consistency checks failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
