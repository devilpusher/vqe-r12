#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production-facing SF-[2]R12 correction helpers for the He parent-basis path.

The implemented path is the audited paper/Tequila spin-free [2]R12 contraction:

* parent-basis RI tensors only;
* passive indices are CABS-only, i.e. RI minus OBS;
* SP ansatz t[p,q,r,s] = 3/8 delta[p,r]delta[q,s] + 1/8 delta[p,s]delta[q,r];
* correction = V + B + X + Delta_MBeq.

This module deliberately does not expose the older Step-6 candidate rows.  Those
remain in the audit scripts for debugging formula variants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from r12_common import build_sp_tensor, maxabs, reconstruct_energy, rdm_diagnostics, tensor_diagnostics


def load_metadata(data) -> Dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    try:
        return json.loads(str(data["metadata_json"]))
    except Exception:
        return {}


def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def metadata_energy(meta: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    """Read energy keys from Step5a-style flat metadata or Step7c nested metadata."""
    val = as_float(meta.get(key))
    if val is not None:
        return val
    energies = meta.get("energies", {})
    if isinstance(energies, dict):
        val = as_float(energies.get(key))
        if val is not None:
            return val
    return default


def matching_step4b_path(inp: str) -> Optional[str]:
    name = Path(inp).name
    suffix = "_step5a_r12_intermediates.npz"
    if not name.endswith(suffix):
        return None
    candidate = Path(name[: -len(suffix)] + "_step4b_obs_fci_rdm.npz")
    return str(candidate) if candidate.exists() else None


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


def chem_to_phys(T: np.ndarray) -> np.ndarray:
    """Convert chem/Mulliken (pq|rs) to phys/Dirac <p q|r s> = chem[p,r,q,s]."""
    return np.asarray(T, dtype=float).transpose(0, 2, 1, 3)


def block2(A: np.ndarray, i, j) -> np.ndarray:
    return A[np.ix_(i, j)]


def block4(T: np.ndarray, i, j, k, l) -> np.ndarray:
    return T[np.ix_(i, j, k, l)]


def build_fock_tequila(h: np.ndarray, g_phys: np.ndarray, rdm1: np.ndarray, active: list[int], full: list[int]) -> np.ndarray:
    """Build the paper/Tequila spin-free Fock operator for the active OBS RDM."""
    g_fafa = block4(g_phys, full, active, full, active)
    g_1 = np.einsum("sr,krls->kl", rdm1, g_fafa, optimize=True)
    g_2 = np.einsum("sr,krsl->kl", rdm1, g_fafa.transpose(0, 1, 3, 2), optimize=True)
    return h[np.ix_(full, full)] + g_1 - 0.5 * g_2


def compute_sf2r12_components(
    g_phys: np.ndarray,
    r_phys: np.ndarray,
    fock: np.ndarray,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    nobs: int,
    nri: int,
) -> Dict[str, float]:
    """Compute the audited CABS-only passive V/B/X/Delta SF-[2]R12 components."""
    a = list(range(nobs))
    p = list(range(nobs, nri))
    f = list(range(nri))
    t = build_sp_tensor(nobs)

    gKLxy_rRSkl = np.einsum(
        "klxy,rskl->rsxy",
        block4(g_phys, f, f, a, a),
        block4(r_phys, a, a, f, f),
        optimize=True,
    )
    gTUxy_rRStu = np.einsum(
        "tuxy,rstu->rsxy",
        block4(g_phys, a, a, a, a),
        block4(r_phys, a, a, a, a),
        optimize=True,
    )
    gATxy_rdm1Ut_rRSau = np.einsum(
        "atxy,ut,rsau->rsxy",
        block4(g_phys, p, a, a, a),
        rdm1,
        block4(r_phys, a, a, p, a),
        optimize=True,
    )
    V_mid = gKLxy_rRSkl - gTUxy_rRStu - gATxy_rdm1Ut_rRSau
    V = float(np.einsum("pqrs,xypq,rsxy", t, rdm2, V_mid, optimize=True))

    rZYpq_fockXy_rTUzx = np.einsum(
        "zypq,xy,tuzx->tupq",
        block4(r_phys, a, a, a, a),
        block2(fock, a, a),
        block4(r_phys, a, a, a, a),
        optimize=True,
    )
    rAYpq_fockXa_rTUxy = np.einsum(
        "aypq,xa,tuxy->tupq",
        block4(r_phys, p, a, a, a),
        block2(fock, a, p),
        block4(r_phys, a, a, a, a),
        optimize=True,
    )
    rYXpq_fockAx_rTUya = np.einsum(
        "yxpq,ax,tuya->tupq",
        block4(r_phys, a, a, a, a),
        block2(fock, p, a),
        block4(r_phys, a, a, a, p),
        optimize=True,
    )
    rMLpq_fockKl_rTUmk = np.einsum(
        "mlpq,kl,tumk->tupq",
        block4(r_phys, f, f, a, a),
        block2(fock, f, f),
        block4(r_phys, a, a, f, f),
        optimize=True,
    )
    rBYpq_rdm1Xy_fockAb_rTUax = np.einsum(
        "bypq,xy,ab,tuax->tupq",
        block4(r_phys, p, a, a, a),
        rdm1,
        block2(fock, p, p),
        block4(r_phys, a, a, p, a),
        optimize=True,
    )
    rAYpq_rdm1Xy_fockKx_rTUak = np.einsum(
        "aypq,xy,kx,tuak->tupq",
        block4(r_phys, p, a, a, a),
        rdm1,
        block2(fock, f, a),
        block4(r_phys, a, a, p, f),
        optimize=True,
    )
    B_mid = (
        rMLpq_fockKl_rTUmk
        - rZYpq_fockXy_rTUzx
        - rAYpq_fockXa_rTUxy
        - rYXpq_fockAx_rTUya
        - 0.5 * rBYpq_rdm1Xy_fockAb_rTUax
        - 0.5 * rAYpq_rdm1Xy_fockKx_rTUak
    )
    B = float(np.einsum("pqrs,vwtu,rsvw,tupq", t, t, rdm2, B_mid, optimize=True))

    rTUkl_rKLpq = np.einsum(
        "tukl,klpq->tupq",
        block4(r_phys, a, a, f, f),
        block4(r_phys, f, f, a, a),
        optimize=True,
    )
    rTUyz_rYZpq = np.einsum(
        "tuyz,yzpq->tupq",
        block4(r_phys, a, a, a, a),
        block4(r_phys, a, a, a, a),
        optimize=True,
    )
    rUTya_rdm1Yz_rAZpq = np.einsum(
        "utya,yz,azpq->tupq",
        block4(r_phys, a, a, a, p),
        rdm1,
        block4(r_phys, p, a, a, a),
        optimize=True,
    )
    rTUay_rdm1Yz_rAZqp = np.einsum(
        "tuay,yz,azqp->tupq",
        block4(r_phys, a, a, p, a),
        rdm1,
        block4(r_phys, p, a, a, a),
        optimize=True,
    )
    X_mid = rTUkl_rKLpq - rTUyz_rYZpq - 0.5 * rUTya_rdm1Yz_rAZpq - 0.5 * rTUay_rdm1Yz_rAZqp
    X = float(-np.einsum("pqrs,vwtu,rsvx,xw,tupq", t, t, rdm2, block2(fock, a, a), X_mid, optimize=True))

    Delta1 = (
        -0.5 * np.einsum("pqrs,aypq,vwtu,xrvy,kx,sw,utak", t, block4(r_phys, p, a, a, a), t, rdm2, block2(fock, f, a), rdm1, block4(r_phys, a, a, p, f), optimize=True)
        -0.5 * np.einsum("pqrs,aypq,vwtu,xryv,kx,sw,tuak", t, block4(r_phys, p, a, a, a), t, rdm2, block2(fock, f, a), rdm1, block4(r_phys, a, a, p, f), optimize=True)
        -0.5 * np.einsum("pqrs,aypq,vwtu,kx,rv,sw,xy,utak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
        + np.einsum("pqrs,aypq,vwtu,kx,rv,sw,xy,tuak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
        +0.5 * np.einsum("pqrs,aypq,vwtu,kx,ry,sv,xw,tuak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
        -0.25 * np.einsum("pqrs,aypq,vwtu,kx,ry,sv,xw,utak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
    )
    Delta2 = (
        np.einsum("pqrs,ayqp,vwtu,xrvy,kx,sw,utak", t, block4(r_phys, p, a, a, a), t, rdm2, block2(fock, f, a), rdm1, block4(r_phys, a, a, p, f), optimize=True)
        -0.5 * np.einsum("pqrs,ayqp,vwtu,xrvy,kx,sw,tuak", t, block4(r_phys, p, a, a, a), t, rdm2, block2(fock, f, a), rdm1, block4(r_phys, a, a, p, f), optimize=True)
        - np.einsum("pqrs,ayqp,vwtu,kx,ry,sv,xw,tuak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
        +0.5 * np.einsum("pqrs,ayqp,vwtu,kx,ry,sv,xw,utak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
    )
    Delta = float(Delta1 + Delta2)
    return {
        "V": V,
        "B": B,
        "X": X,
        "Delta": Delta,
        "correction": V + B + X + Delta,
    }


def compute_he_sf2r12_correction(
    step5a_path: str | Path,
    step4b_path: str | Path | None = None,
    scale_f12: float = 1.0,
) -> Dict[str, Any]:
    """Load Step-5a/4b files and return the formal He SF-[2]R12 correction."""
    step5a_path = str(step5a_path)
    data = np.load(step5a_path, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta.get("nobs", np.array(data["dm1_obs"]).shape[0]))
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))

    resolved_step4b = str(step4b_path) if step4b_path is not None else matching_step4b_path(step5a_path)
    E_full: Optional[float] = None
    if resolved_step4b is not None and Path(resolved_step4b).exists():
        step4b = np.load(resolved_step4b, allow_pickle=True)
        step4b_meta = load_metadata(step4b)
        E_full = metadata_energy(step4b_meta, "E_full_parent_fci")

    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    f12_ri = scale_f12 * np.array(data["f12_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)
    E_obs_val = metadata_energy(meta, "E_obs_fci")
    if E_obs_val is None:
        raise KeyError("Cannot find E_obs_fci in metadata or metadata['energies']")
    E_obs = float(E_obs_val)
    enuc = float(metadata_energy(meta, "enuc", 0.0) or 0.0)

    E_obs_rdm, _, _ = reconstruct_energy(h_ri[:nobs, :nobs], eri_ri[:nobs, :nobs, :nobs, :nobs], dm1_obs, dm2_obs, enuc)
    E_ri_rdm, _, _ = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)

    g_phys = chem_to_phys(eri_ri)
    r_phys = chem_to_phys(f12_ri)
    fock = build_fock_tequila(h_ri, g_phys, dm1_obs, list(range(nobs)), list(range(nri)))
    components = compute_sf2r12_components(g_phys, r_phys, fock, dm1_obs, dm2_obs, nobs, nri)
    metrics = energy_metrics(components["correction"], E_obs, E_full)

    fock_step5a = np.array(data["F_ri"], dtype=float) if "F_ri" in data.files else None
    fock_gap = None if fock_step5a is None else maxabs(fock - fock_step5a)
    diagnostics = {
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
            "eri_phys": tensor_diagnostics(g_phys),
            "f12_phys": tensor_diagnostics(r_phys),
        },
        "fock_diagnostics": {
            "maxabs_fock_tequila_minus_step5a": fock_gap,
            "fock_step5a_available": fock_step5a is not None,
            "norm_fock_tequila": float(np.linalg.norm(fock)),
        },
        "sp_ansatz": {
            "direct": 3.0 / 8.0,
            "exchange": 1.0 / 8.0,
            "t_0000": float(build_sp_tensor(nobs)[0, 0, 0, 0]),
        },
    }

    return {
        "method": "paper_tequila_sf2r12",
        "fock_model": "tequila_fock_from_paper_formula",
        "passive_space": "CABS_only_RI_minus_OBS",
        "step5a_path": step5a_path,
        "step4b_path": resolved_step4b,
        "nobs": nobs,
        "nri": nri,
        "ncabs": nri - nobs,
        "scale_f12": scale_f12,
        "E_obs_fci": E_obs,
        "E_full_parent_fci": E_full,
        "full_parent_gap": None if E_full is None else E_full - E_obs,
        "components": components,
        "delta_E_r12": components["correction"],
        **metrics,
        "diagnostics": diagnostics,
    }


def validate_correction_result(result: Dict[str, Any], tol: float = 1e-10) -> None:
    checks = result["diagnostics"]["energy_checks"]
    if abs(checks["delta_obs_rdm_minus_fci"]) > tol:
        raise ValueError("OBS RDM energy reconstruction check failed")
    if abs(checks["delta_ri_rdm_minus_fci"]) > tol:
        raise ValueError("RI-embedded RDM energy reconstruction check failed")
    for name, diag in result["diagnostics"]["tensor_diagnostics"].items():
        if diag["has_nan"] or diag["has_inf"]:
            raise ValueError(f"{name} contains NaN or Inf")
    for name, diag in result["diagnostics"]["rdm_diagnostics"].items():
        if abs(diag["trace_dm1"] - 2.0) > 1e-8:
            raise ValueError(f"{name} dm1 trace check failed")
        if abs(diag["trace_dm2"] - 2.0) > 1e-8:
            raise ValueError(f"{name} dm2 trace check failed")
