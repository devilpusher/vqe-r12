#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for the parent-basis He R12 prototype workflow.

This module is intentionally conservative: it mirrors the conventions already
validated in Step 4b/5a/5b, while giving Step 6+ code a single import target.
Existing scripts can keep their local copies until each migration is tested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def maxabs(A: np.ndarray) -> float:
    return float(np.max(np.abs(A))) if A.size else 0.0


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def load_metadata(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    try:
        return json.loads(str(data["metadata_json"]))
    except Exception:
        return {}


def load_step4b_data(path: str | Path) -> Dict[str, Any]:
    """Load a Step-4b npz file and expose arrays plus decoded metadata."""
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {key: data[key] for key in data.files}
    out["metadata"] = load_metadata(data)
    return out


def assert_finite(name: str, A: np.ndarray) -> None:
    if np.isnan(A).any() or np.isinf(A).any():
        raise ValueError(f"{name} contains NaN or Inf")


def reconstruct_energy(
    h: np.ndarray,
    eri: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    enuc: float = 0.0,
) -> Tuple[float, float, float]:
    """Spin-free RDM energy convention validated in Step 1 and Step 4b."""
    e1 = float(np.einsum("pq,pq", h, dm1, optimize=True))
    e2 = float(0.5 * np.einsum("pqrs,pqrs", eri, dm2, optimize=True))
    return e1 + e2 + enuc, e1, e2


def build_generalized_fock_spinfree(
    h: np.ndarray,
    eri: np.ndarray,
    dm1: np.ndarray,
) -> np.ndarray:
    """Build the spin-free generalized Fock used by Step 5a diagnostics."""
    J = np.einsum("rs,pqrs->pq", dm1, eri, optimize=True)
    K = np.einsum("rs,prqs->pq", dm1, eri, optimize=True)
    return sym(h + J - 0.5 * K)


def build_sp_tensor(nobs: int) -> np.ndarray:
    """SP ansatz tensor d[p,q,r,s] = 3/8 delta_pr delta_qs + 1/8 delta_ps delta_qr."""
    D = np.zeros((nobs, nobs, nobs, nobs), dtype=float)
    for p in range(nobs):
        for q in range(nobs):
            D[p, q, p, q] += 3.0 / 8.0
            D[p, q, q, p] += 1.0 / 8.0
    return D


def transform_4index(I_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Transform a four-index AO tensor with the same coefficient matrix on all legs."""
    return np.einsum("up,vq,wr,xs,uvwx->pqrs", C, C, C, C, I_ao, optimize=True)


def embed_rdm_to_ri(
    dm1_obs: np.ndarray,
    dm2_obs: np.ndarray,
    nri: int,
    nobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed OBS RDMs into the leading OBS block of an RI-sized tensor."""
    dm1_ri = np.zeros((nri, nri), dtype=float)
    dm2_ri = np.zeros((nri, nri, nri, nri), dtype=float)
    dm1_ri[:nobs, :nobs] = dm1_obs
    dm2_ri[:nobs, :nobs, :nobs, :nobs] = dm2_obs
    return dm1_ri, dm2_ri


def q_project_pair_amplitude(A: np.ndarray, nobs: int) -> np.ndarray:
    """Remove the pure OBS-OBS pair block from an ordered pair-amplitude matrix."""
    out = np.array(A, dtype=float, copy=True)
    out[:nobs, :nobs] = 0.0
    return out


def build_pair_hamiltonian(h: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Ordered alpha-beta pair Hamiltonian used by Step 5b diagnostics."""
    n = h.shape[0]
    H = np.zeros((n * n, n * n), dtype=float)
    for p in range(n):
        for q in range(n):
            I = p * n + q
            for r in range(n):
                for s in range(n):
                    J = r * n + s
                    val = 0.0
                    if q == s:
                        val += h[p, r]
                    if p == r:
                        val += h[q, s]
                    val += eri[p, r, q, s]
                    H[I, J] = val
    return sym(H)


def build_pair_fock_operator(F: np.ndarray) -> np.ndarray:
    """Ordered pair-space operator F(1)+F(2)."""
    n = F.shape[0]
    return np.kron(F, np.eye(n)) + np.kron(np.eye(n), F)


def q_pair_indices(nri: int, nobs: int) -> np.ndarray:
    idx = []
    for p in range(nri):
        for q in range(nri):
            if not (p < nobs and q < nobs):
                idx.append(p * nri + q)
    return np.array(idx, dtype=int)


def pp_pair_indices(nri: int, nobs: int) -> np.ndarray:
    idx = []
    for p in range(nobs):
        for q in range(nobs):
            idx.append(p * nri + q)
    return np.array(idx, dtype=int)


def two_body_expectation(T: np.ndarray, dm2: np.ndarray) -> float:
    """Return 1/2 sum_pqrs T[p,q,r,s] dm2[p,q,r,s]."""
    return float(0.5 * np.einsum("pqrs,pqrs", T, dm2, optimize=True))


def pair_matrix(T: np.ndarray) -> np.ndarray:
    """Flatten T[p,q,r,s] to ordered-pair matrix T[p*n+q, r*n+s]."""
    n = T.shape[0]
    return np.asarray(T, dtype=float).reshape(n * n, n * n)


def pair_projector(nri: int, indices: np.ndarray) -> np.ndarray:
    """Diagonal projector in ordered pair space."""
    P = np.zeros((nri * nri, nri * nri), dtype=float)
    P[indices, indices] = 1.0
    return P


def rdm_diagnostics(dm1: np.ndarray, dm2: np.ndarray) -> Dict[str, Any]:
    occ = np.linalg.eigvalsh(sym(dm1))[::-1]
    return {
        "trace_dm1": float(np.trace(dm1)),
        "trace_dm2": float(np.einsum("pprr->", dm2, optimize=True)),
        "max_dm1_asym": maxabs(dm1 - dm1.T),
        "max_dm2_bra_ket_error": maxabs(dm2 - dm2.transpose(2, 3, 0, 1)),
        "natural_occupations": occ.tolist(),
    }


def tensor_diagnostics(T: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(T.shape),
        "has_nan": bool(np.isnan(T).any()),
        "has_inf": bool(np.isinf(T).any()),
        "norm": float(np.linalg.norm(T.reshape(-1))) if T.size else 0.0,
        "max_abs": float(np.max(np.abs(T))) if T.size else 0.0,
        "bra_ket_error": maxabs(T - T.transpose(2, 3, 0, 1)),
        "pair_bra_error": maxabs(T - T.transpose(1, 0, 2, 3)),
        "pair_ket_error": maxabs(T - T.transpose(0, 1, 3, 2)),
    }
