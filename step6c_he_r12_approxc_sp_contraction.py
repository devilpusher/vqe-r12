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
    A_raw_vec = A_raw_Q.reshape(-1)
    A_sp_vec = A_sp_Q.reshape(-1)

    H_pair = build_pair_hamiltonian(h_ri, eri_ri)
    K_pair = build_pair_fock_operator(F_ri)
    qidx = q_pair_indices(nri, nobs)

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
            "This is not yet the final article-level [2]R12 tensor formula."
        ),
    }

    np.savez(
        args.out,
        H_pair=H_pair,
        K_pair_fock=K_pair,
        qidx=qidx,
        psi0_pair=psi0,
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

