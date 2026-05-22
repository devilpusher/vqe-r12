#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6k: Paper/Tequila SF-[2]R12 factor-map audit for He.

The VQE+[2]R12 paper gives the high-level SF-[2]R12 setup and points to the
original SF-[2]R12 literature for the full approximation-C contractions.  Its
public Tequila implementation is therefore a useful executable formula map.

This script implements the Tequila-style spin-free contractions directly on
our parent-basis tensors:

* Eq. (7): correlation operator R has a 1/2 sum, but Tequila's executable
  tensor formulas absorb the spin-free convention into the contracted
  intermediates rather than exposing a final scalar prefactor.
* Eq. (8): SP ansatz t[p,q,r,s] = 3/8 delta[p,r]delta[q,s] + 1/8 delta[p,s]delta[q,r].
* Passive/CABS space is RI minus OBS, not RI minus occupied orbitals.

It is an audit and factor-map script, not a replacement for the earlier
formula-building scripts yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from r12_common import build_sp_tensor, maxabs, reconstruct_energy, rdm_diagnostics, tensor_diagnostics
from step6e_build_vxbc_intermediates import default_prefix


PAPER_PDF = r"D:\vqe\D2CP00247G.pdf"
TEQUILA_SOURCE = "https://github.com/tequilahub/tequila"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="he_ccpvdz_nobs2_fitN7_step5a_r12_intermediates.npz")
    p.add_argument("--step4b", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--scale-f12", type=float, default=1.0)
    return p.parse_args()


def load_metadata(data) -> Dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    return json.loads(str(data["metadata_json"]))


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
    # f^k_l = h^k_l + Gamma^s_r * (g^{kr}_{ls} - 1/2 g^{kr}_{sl})
    g_fafa = block4(g_phys, full, active, full, active)
    g_1 = np.einsum("sr,krls->kl", rdm1, g_fafa, optimize=True)
    g_2 = np.einsum("sr,krsl->kl", rdm1, g_fafa.transpose(0, 1, 3, 2), optimize=True)
    return h[np.ix_(full, full)] + g_1 - 0.5 * g_2


def sp_ansatz(nobs: int) -> np.ndarray:
    return build_sp_tensor(nobs)


def compute_tequila_style_components(
    g_phys: np.ndarray,
    r_phys: np.ndarray,
    fock: np.ndarray,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    nobs: int,
    nri: int,
) -> Dict[str, float]:
    a = list(range(nobs))
    p = list(range(nobs, nri))
    f = list(range(nri))
    t = sp_ansatz(nobs)

    # V intermediate from tequila._compute_intermediate_V
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

    # B intermediate from tequila._compute_intermediate_B
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

    # X intermediate from tequila._compute_intermediate_X
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

    # Delta intermediate from Tequila MBeq implementation.
    Delta1 = (
        -0.5
        * np.einsum(
            "pqrs,aypq,vwtu,xrvy,kx,sw,utak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            rdm2,
            block2(fock, f, a),
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
        -0.5
        * np.einsum(
            "pqrs,aypq,vwtu,xryv,kx,sw,tuak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            rdm2,
            block2(fock, f, a),
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
        -0.5
        * np.einsum(
            "pqrs,aypq,vwtu,kx,rv,sw,xy,utak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            block2(fock, f, a),
            rdm1,
            rdm1,
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
        + np.einsum(
            "pqrs,aypq,vwtu,kx,rv,sw,xy,tuak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            block2(fock, f, a),
            rdm1,
            rdm1,
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
        +0.5
        * np.einsum(
            "pqrs,aypq,vwtu,kx,ry,sv,xw,tuak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            block2(fock, f, a),
            rdm1,
            rdm1,
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
        -0.25
        * np.einsum(
            "pqrs,aypq,vwtu,kx,ry,sv,xw,utak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            block2(fock, f, a),
            rdm1,
            rdm1,
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
    )
    Delta2 = (
        np.einsum(
            "pqrs,ayqp,vwtu,xrvy,kx,sw,utak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            rdm2,
            block2(fock, f, a),
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
        -0.5
        * np.einsum(
            "pqrs,ayqp,vwtu,xrvy,kx,sw,tuak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            rdm2,
            block2(fock, f, a),
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
        - np.einsum(
            "pqrs,ayqp,vwtu,kx,ry,sv,xw,tuak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            block2(fock, f, a),
            rdm1,
            rdm1,
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
        +0.5
        * np.einsum(
            "pqrs,ayqp,vwtu,kx,ry,sv,xw,utak",
            t,
            block4(r_phys, p, a, a, a),
            t,
            block2(fock, f, a),
            rdm1,
            rdm1,
            rdm1,
            block4(r_phys, a, a, p, f),
            optimize=True,
        )
    )
    Delta = float(Delta1 + Delta2)
    return {
        "V": V,
        "B": B,
        "X": X,
        "Delta": Delta,
        "correction": V + B + X + Delta,
    }


def main():
    args = parse_args()
    prefix = default_prefix(args.inp)
    if args.out_json is None:
        args.out_json = f"{prefix}_step6k_paper_tequila_sf2r12_audit.json"
    if args.out_csv is None:
        args.out_csv = f"{prefix}_step6k_paper_tequila_sf2r12_audit.csv"
    if args.summary is None:
        args.summary = f"{prefix}_step6k_paper_tequila_sf2r12_audit_summary.txt"

    data = np.load(args.inp, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta.get("nobs", np.array(data["Cab_obs"]).shape[0]))
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))
    step4b_path = args.step4b or matching_step4b_path(args.inp)
    E_full: Optional[float] = None
    if step4b_path is not None and Path(step4b_path).exists():
        step4b = np.load(step4b_path, allow_pickle=True)
        E_full = as_float(load_metadata(step4b).get("E_full_parent_fci"))

    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    r_ri = args.scale_f12 * np.array(data["f12_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)
    E_obs = float(meta["E_obs_fci"])
    enuc = float(meta.get("enuc", 0.0))
    E_obs_rdm, _, _ = reconstruct_energy(h_ri[:nobs, :nobs], eri_ri[:nobs, :nobs, :nobs, :nobs], dm1_obs, dm2_obs, enuc)
    E_ri_rdm, _, _ = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)

    g_phys = chem_to_phys(eri_ri)
    r_phys = chem_to_phys(r_ri)
    fock_tequila = build_fock_tequila(h_ri, g_phys, dm1_obs, list(range(nobs)), list(range(nri)))
    fock_step5a = np.array(data["F_ri"], dtype=float)
    components_tequila_fock = compute_tequila_style_components(g_phys, r_phys, fock_tequila, dm1_obs, dm2_obs, nobs, nri)
    components_step5a_fock = compute_tequila_style_components(g_phys, r_phys, fock_step5a, dm1_obs, dm2_obs, nobs, nri)

    rows = []
    for fock_label, components in [
        ("tequila_fock_from_paper_formula", components_tequila_fock),
        ("step5a_generalized_fock", components_step5a_fock),
    ]:
        for label, delta in components.items():
            if label == "correction":
                continue
            rows.append({
                "fock_model": fock_label,
                "component": label,
                "delta_E": delta,
                **energy_metrics(delta, E_obs, E_full),
            })
        rows.append({
            "fock_model": fock_label,
            "component": "total_V+B+X+Delta",
            "delta_E": components["correction"],
            **energy_metrics(components["correction"], E_obs, E_full),
        })

    diagnostics = {
        "input": args.inp,
        "paper_pdf": PAPER_PDF,
        "tequila_source": TEQUILA_SOURCE,
        "formula_anchors": {
            "paper_eq7": "R = 1/2 sum d * excitation operator; Tequila absorbs this in spin-free tensor contractions.",
            "paper_eq8": "SP ansatz t[p,q,r,s] = 3/8 delta[p,r]delta[q,s] + 1/8 delta[p,s]delta[q,r].",
            "tequila_passive_space": "p = full - active = RI minus OBS; not RI minus occupied.",
            "tequila_correction": "correction = V + B + X + Delta_MBeq.",
        },
        "nri": nri,
        "nobs": nobs,
        "npassive_cabs": nri - nobs,
        "passive_indices": list(range(nobs, nri)),
        "E_obs_fci": E_obs,
        "E_full_parent_fci": E_full,
        "gap": None if E_full is None else E_full - E_obs,
        "scale_f12": args.scale_f12,
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
        "fock_audit": {
            "maxabs_fock_tequila_minus_step5a": maxabs(fock_tequila - fock_step5a),
            "norm_fock_tequila": float(np.linalg.norm(fock_tequila)),
            "norm_fock_step5a": float(np.linalg.norm(fock_step5a)),
        },
        "sp_audit": {
            "t_0000": float(sp_ansatz(nobs)[0, 0, 0, 0]),
            "direct": 3.0 / 8.0,
            "exchange": 1.0 / 8.0,
        },
        "components": {
            "tequila_fock_from_paper_formula": components_tequila_fock,
            "step5a_generalized_fock": components_step5a_fock,
        },
        "rows": rows,
        "decision": (
            "The paper/Tequila route does not reduce to a single He occupied-pair prefactor. "
            "It uses CABS-only passive indices and full spin-free RDM contractions V+B+X+Delta."
        ),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    fieldnames = ["fock_model", "component", "delta_E", "E_total", "residual_to_full_parent_FCI", "abs_residual_to_full_mEh", "recovery_ratio", "overcorrection"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6k | Paper/Tequila SF-[2]R12 factor-map audit")
    lines.append("=" * 100)
    lines.append(f"input       = {args.inp}")
    lines.append(f"nri/nobs/passive = {nri}/{nobs}/{nri - nobs}; passive indices = {list(range(nobs, nri))}")
    lines.append(f"E_OBS-FCI   = {E_obs:.14f} Eh")
    if E_full is not None:
        gap = E_full - E_obs
        lines.append(f"E_full_parent_FCI = {E_full:.14f} Eh")
        lines.append(f"OBS-to-full gap   = {gap:.12e} Eh ({abs(gap) * 1000.0:.6f} mEh)")
    lines.append(f"E checks    = obs-rdm {E_obs_rdm - E_obs:.3e}, ri-rdm {E_ri_rdm - E_obs:.3e}")
    lines.append(f"max|F_tequila - F_step5a| = {diagnostics['fock_audit']['maxabs_fock_tequila_minus_step5a']:.3e}")
    lines.append("")
    lines.append("[Formula anchors]")
    for key, val in diagnostics["formula_anchors"].items():
        lines.append(f"{key}: {val}")
    lines.append("")
    lines.append("[Components]")
    lines.append("| fock model | component | DeltaE / Eh | residual / mEh | recovery |")
    lines.append("|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['fock_model']} | {row['component']} | {row['delta_E']:.12e} | "
            f"{row['abs_residual_to_full_mEh']:.6e} | {row['recovery_ratio']:.6e} |"
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
        and not diagnostics["tensor_diagnostics"]["eri_phys"]["has_nan"]
        and not diagnostics["tensor_diagnostics"]["f12_phys"]["has_nan"]
    )
    if not ok:
        print("\nERROR: Step 6k consistency checks failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
