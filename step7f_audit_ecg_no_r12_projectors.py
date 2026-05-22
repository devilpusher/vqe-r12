#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7f: Audit ECG-NO R12 projector/passive-space behavior.

This is not a new production correction.  It diagnoses why the Step7d
paper/Tequila SF-[2]R12 row changes from a stable negative correction in
s[0,1,2]+p[0] to near-zero or small-positive corrections in larger ECG-NO
spaces.

Audits performed for each input:

* baseline CABS-only passive space, matching Step7d;
* occupied-external passive space, where passive indices are all RI orbitals
  except the first occupied natural orbital;
* scale/sign response inferred from V*s + (B+X+Delta)*s^2;
* OBS/CABS block norms for f12 and ERI in phys/Dirac ordering;
* natural occupations and energy reconstruction checks from the Step7c file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from r12_common import build_sp_tensor, maxabs, reconstruct_energy, rdm_diagnostics, tensor_diagnostics
from r12_correction import block2, block4, build_fock_tequila, chem_to_phys, load_metadata, metadata_energy


DEFAULT_INPUTS = [
    "step7c_ecg_no_sp_s012_p0_fitN7_r12only_step4b_like.npz",
    "step7c_ecg_no_sp_s012_p01_fitN7_r12only_step4b_like.npz",
    "step7c_ecg_no_spd_s012_p01_d0_fitN7_r12only_step4b_like.npz",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--nocc", type=int, default=1)
    p.add_argument("--scales", default="-1,-0.5,0.5,1")
    p.add_argument("--out-json", default="step7f_ecg_no_r12_projector_audit.json")
    p.add_argument("--out-csv", default="step7f_ecg_no_r12_projector_audit.csv")
    p.add_argument("--summary", default="step7f_ecg_no_r12_projector_audit_summary.txt")
    return p.parse_args()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def case_label(path: str | Path) -> str:
    name = Path(path).name
    if name.startswith("step7c_ecg_no_"):
        name = name[len("step7c_ecg_no_") :]
    for suffix in ["_r12only_step4b_like.npz", "_step4b_like.npz"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def norm_block(T: np.ndarray, i, j, k, l) -> float:
    return float(np.linalg.norm(block4(T, i, j, k, l).reshape(-1)))


def compute_components_passive(
    g_phys: np.ndarray,
    r_phys: np.ndarray,
    fock: np.ndarray,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    nobs: int,
    nri: int,
    passive: List[int],
) -> Dict[str, float]:
    """Same audited contraction as Step7d, but with explicit passive indices."""
    a = list(range(nobs))
    p = passive
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
        + 0.5 * np.einsum("pqrs,aypq,vwtu,kx,ry,sv,xw,tuak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
        - 0.25 * np.einsum("pqrs,aypq,vwtu,kx,ry,sv,xw,utak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
    )
    Delta2 = (
        np.einsum("pqrs,ayqp,vwtu,xrvy,kx,sw,utak", t, block4(r_phys, p, a, a, a), t, rdm2, block2(fock, f, a), rdm1, block4(r_phys, a, a, p, f), optimize=True)
        - 0.5 * np.einsum("pqrs,ayqp,vwtu,xrvy,kx,sw,tuak", t, block4(r_phys, p, a, a, a), t, rdm2, block2(fock, f, a), rdm1, block4(r_phys, a, a, p, f), optimize=True)
        - np.einsum("pqrs,ayqp,vwtu,kx,ry,sv,xw,tuak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
        + 0.5 * np.einsum("pqrs,ayqp,vwtu,kx,ry,sv,xw,utak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), rdm1, rdm1, rdm1, block4(r_phys, a, a, p, f), optimize=True)
    )
    Delta = float(Delta1 + Delta2)
    return {"V": V, "B": B, "X": X, "Delta": Delta, "correction": V + B + X + Delta}


def load_case(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta["nobs"])
    nri = int(meta["nri"])
    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    f12_ri = np.array(data["f12_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)
    enuc = float(metadata_energy(meta, "enuc", 0.0) or 0.0)
    E_obs = float(metadata_energy(meta, "E_obs_fci"))
    E_obs_rdm, _, _ = reconstruct_energy(h_ri[:nobs, :nobs], eri_ri[:nobs, :nobs, :nobs, :nobs], dm1_obs, dm2_obs, enuc)
    E_ri_rdm, _, _ = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)
    g_phys = chem_to_phys(eri_ri)
    r_phys = chem_to_phys(f12_ri)
    fock = build_fock_tequila(h_ri, g_phys, dm1_obs, list(range(nobs)), list(range(nri)))
    return {
        "path": path,
        "label": case_label(path),
        "meta": meta,
        "nobs": nobs,
        "nri": nri,
        "h_ri": h_ri,
        "g_phys": g_phys,
        "r_phys": r_phys,
        "fock": fock,
        "dm1_obs": dm1_obs,
        "dm2_obs": dm2_obs,
        "E_obs": E_obs,
        "energy_checks": {
            "E_obs_rdm": E_obs_rdm,
            "E_ri_rdm": E_ri_rdm,
            "delta_obs_rdm_minus_fci": E_obs_rdm - E_obs,
            "delta_ri_rdm_minus_fci": E_ri_rdm - E_obs,
        },
        "rdm_diag": rdm_diagnostics(dm1_obs, dm2_obs),
        "tensor_diag": {
            "eri_phys": tensor_diagnostics(g_phys),
            "f12_phys": tensor_diagnostics(r_phys),
        },
    }


def audit_case(case: Dict[str, Any], nocc: int, scales: List[float]) -> Dict[str, Any]:
    nobs = case["nobs"]
    nri = case["nri"]
    a = list(range(nobs))
    cabs = list(range(nobs, nri))
    occ_external = list(range(nocc, nri))

    rows = []
    for mode, passive in [
        ("cabs_only", cabs),
        ("occupied_external", occ_external),
    ]:
        comp = compute_components_passive(
            case["g_phys"],
            case["r_phys"],
            case["fock"],
            case["dm1_obs"],
            case["dm2_obs"],
            nobs,
            nri,
            passive,
        )
        quad = comp["B"] + comp["X"] + comp["Delta"]
        scale_rows = []
        for s in scales:
            scale_rows.append(
                {
                    "scale": s,
                    "linear_V": s * comp["V"],
                    "quadratic_BXD": s * s * quad,
                    "correction": s * comp["V"] + s * s * quad,
                }
            )
        rows.append(
            {
                "passive_mode": mode,
                "passive_size": len(passive),
                "passive_first": passive[0] if passive else None,
                "V": comp["V"],
                "B": comp["B"],
                "X": comp["X"],
                "Delta": comp["Delta"],
                "B_plus_X_plus_Delta": quad,
                "correction": comp["correction"],
                "scale_scan": scale_rows,
            }
        )

    block_norms = {
        "f12_aa_aa": norm_block(case["r_phys"], a, a, a, a),
        "f12_aa_cabs_a": norm_block(case["r_phys"], a, a, cabs, a) if cabs else 0.0,
        "f12_cabs_a_a_a": norm_block(case["r_phys"], cabs, a, a, a) if cabs else 0.0,
        "f12_cabs_cabs_a_a": norm_block(case["r_phys"], cabs, cabs, a, a) if cabs else 0.0,
        "eri_aa_aa": norm_block(case["g_phys"], a, a, a, a),
        "eri_cabs_a_a_a": norm_block(case["g_phys"], cabs, a, a, a) if cabs else 0.0,
    }
    return {
        "label": case["label"],
        "path": case["path"],
        "nobs": nobs,
        "nri": nri,
        "ncabs": nri - nobs,
        "nqubits": 2 * nobs,
        "E_obs_fci": case["E_obs"],
        "energy_checks": case["energy_checks"],
        "natural_occupations": case["rdm_diag"]["natural_occupations"],
        "tensor_diagnostics": case["tensor_diag"],
        "block_norms": block_norms,
        "passive_audits": rows,
    }


def write_outputs(args, audits: List[Dict[str, Any]]) -> None:
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"audits": audits}, f, indent=2)

    csv_rows = []
    for audit in audits:
        for row in audit["passive_audits"]:
            csv_rows.append(
                {
                    "case": audit["label"],
                    "nqubits": audit["nqubits"],
                    "nobs": audit["nobs"],
                    "nri": audit["nri"],
                    "passive_mode": row["passive_mode"],
                    "passive_size": row["passive_size"],
                    "E_obs_fci": audit["E_obs_fci"],
                    "V": row["V"],
                    "B": row["B"],
                    "X": row["X"],
                    "Delta": row["Delta"],
                    "B_plus_X_plus_Delta": row["B_plus_X_plus_Delta"],
                    "correction": row["correction"],
                    "f12_aa_aa_norm": audit["block_norms"]["f12_aa_aa"],
                    "f12_cabs_a_a_a_norm": audit["block_norms"]["f12_cabs_a_a_a"],
                    "eri_cabs_a_a_a_norm": audit["block_norms"]["eri_cabs_a_a_a"],
                }
            )
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = []
    lines.append("=" * 112)
    lines.append("Step 7f | ECG-NO R12 projector/passive-space audit")
    lines.append("=" * 112)
    for audit in audits:
        occ = audit["natural_occupations"][:8]
        lines.append("")
        lines.append(f"[{audit['label']}] nobs={audit['nobs']} nri={audit['nri']} qubits={audit['nqubits']}")
        lines.append(f"E_obs_fci = {audit['E_obs_fci']:.14f} Eh")
        lines.append(
            "energy checks: "
            f"OBS-RDM {audit['energy_checks']['delta_obs_rdm_minus_fci']:.3e}, "
            f"RI-RDM {audit['energy_checks']['delta_ri_rdm_minus_fci']:.3e} Eh"
        )
        lines.append("natural occupations first 8 = " + np.array2string(np.array(occ), precision=10))
        bn = audit["block_norms"]
        lines.append(
            "block norms: "
            f"||f12_aa_aa||={bn['f12_aa_aa']:.6e}, "
            f"||f12_cabs_a_a_a||={bn['f12_cabs_a_a_a']:.6e}, "
            f"||eri_cabs_a_a_a||={bn['eri_cabs_a_a_a']:.6e}"
        )
        for row in audit["passive_audits"]:
            lines.append(
                f"  {row['passive_mode']:<18s} psize={row['passive_size']:<4d} "
                f"V={1000*row['V']: .8f} B={1000*row['B']: .8f} "
                f"X={1000*row['X']: .8f} D={1000*row['Delta']: .8f} "
                f"total={1000*row['correction']: .8f} mEh"
            )
            scale_text = ", ".join(f"s={s['scale']:+.1f}:{1000*s['correction']:+.6f}" for s in row["scale_scan"])
            lines.append(f"    scale response mEh: {scale_text}")
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out_json}")
    lines.append(f"  {args.out_csv}")
    lines.append(f"  {args.summary}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    scales = parse_float_list(args.scales)
    audits = [audit_case(load_case(path), args.nocc, scales) for path in args.inputs]
    write_outputs(args, audits)


if __name__ == "__main__":
    main()
