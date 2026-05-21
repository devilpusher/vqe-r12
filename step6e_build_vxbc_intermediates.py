#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6e: Build explicit V/X/B/C intermediates for the He [2]R12 prototype.

The goal is to produce named, auditable intermediate scalars and matrices, not
to claim a final [2]R12 energy.  The script uses direct Psi4 F12 tensors as the
authoritative integral inputs and keeps C/orbital-denominator terms separated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from r12_common import (
    build_pair_fock_operator,
    maxabs,
    pair_matrix,
    pair_projector,
    pp_pair_indices,
    q_pair_indices,
    reconstruct_energy,
    rdm_diagnostics,
    tensor_diagnostics,
)


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


def default_prefix(inp: str) -> str:
    name = Path(inp).name
    suffix = "_step5a_r12_intermediates.npz"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return Path(inp).stem


def ansatz3_indices(nri: int, nobs: int, nocc: int) -> Dict[str, np.ndarray]:
    obs = set(range(nobs))
    occ = set(range(nocc))
    cabs = set(range(nobs, nri))
    rs_obs = []
    a_prime_j = []
    i_b_prime = []
    q3 = []
    for p in range(nri):
        for q in range(nri):
            idx = p * nri + q
            removed = (p in obs and q in obs) or (p in cabs and q in occ) or (p in occ and q in cabs)
            if p in obs and q in obs:
                rs_obs.append(idx)
            if p in cabs and q in occ:
                a_prime_j.append(idx)
            if p in occ and q in cabs:
                i_b_prime.append(idx)
            if not removed:
                q3.append(idx)
    return {
        "rs_obs": np.array(rs_obs, dtype=int),
        "a_prime_j": np.array(a_prime_j, dtype=int),
        "i_b_prime": np.array(i_b_prime, dtype=int),
        "q_ansatz3": np.array(q3, dtype=int),
    }


def safe_div(num: float, den: float, thresh: float):
    if abs(den) <= thresh:
        return None
    return num / den


def source_vectors(data, nri: int, nobs: int, idx_q3: np.ndarray) -> Dict[str, np.ndarray]:
    Cab_obs = np.array(data["Cab_obs"], dtype=float)
    psi0 = np.zeros((nri, nri), dtype=float)
    psi0[:nobs, :nobs] = Cab_obs

    cab_sp = np.zeros((nri, nri), dtype=float)
    cab_sp[:nobs, :nobs] = np.array(data["Cab_sp"], dtype=float)

    A_raw = np.array(data["A_raw_Q"], dtype=float).reshape(-1)
    A_sp = np.array(data["A_sp_Q"], dtype=float).reshape(-1)
    Q3 = pair_projector(nri, idx_q3)

    return {
        "psi0": psi0.reshape(-1),
        "cab_sp": cab_sp.reshape(-1),
        "A_raw_Q3": Q3 @ A_raw,
        "A_sp_Q3": Q3 @ A_sp,
    }


def build_intermediate_matrices(data, nri: int, nobs: int, nocc: int) -> Dict[str, Any]:
    idx = ansatz3_indices(nri, nobs, nocc)
    P_rs = pair_projector(nri, idx["rs_obs"])
    P_aj = pair_projector(nri, idx["a_prime_j"])
    P_ib = pair_projector(nri, idx["i_b_prime"])
    P_removed = P_rs + P_aj + P_ib
    Q3 = pair_projector(nri, idx["q_ansatz3"])

    F_ri = np.array(data["F_ri"], dtype=float)
    K_pair = build_pair_fock_operator(F_ri)
    F12 = pair_matrix(np.array(data["f12_ri"], dtype=float))
    G = pair_matrix(np.array(data["eri_ri"], dtype=float))
    X_direct = pair_matrix(np.array(data["f12sq_ri"], dtype=float))
    V_direct = pair_matrix(np.array(data["f12g12_ri"], dtype=float))
    B_dc = pair_matrix(np.array(data["f12dc_ri"], dtype=float))

    # Candidate C block: old MP2-F12-style C_ab coupling analogue using Q3 f amplitudes
    # and the generalized-Fock pair operator.  Kept separate from V/B tilde terms.
    C_fock = F12.T @ Q3 @ K_pair @ Q3 @ F12

    return {
        "indices": idx,
        "projectors": {
            "P_rs": P_rs,
            "P_a_prime_j": P_aj,
            "P_i_b_prime": P_ib,
            "P_removed": P_removed,
            "Q3": Q3,
        },
        "matrices": {
            "V_direct_f12g12": V_direct,
            "X_direct_f12sq": X_direct,
            "B_direct_f12dc": B_dc,
            "K_pair_fock": K_pair,
            "C_fock_model": C_fock,
            "V_projector_sub_diagnostic": V_direct - G @ P_removed @ F12,
            "X_projector_sub_diagnostic": X_direct - F12 @ P_removed @ F12,
        },
    }


def scalar_block(name: str, M: np.ndarray, sources: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {}
    for label, vec in sources.items():
        out[label] = float(vec @ (M @ vec))
    return {"name": name, "values": out}


def cross_block(name: str, M: np.ndarray, left: np.ndarray, sources: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {}
    for label, vec in sources.items():
        out[label] = float(left @ (M @ vec))
    return {"name": name, "values": out}


def main():
    args = parse_args()
    prefix = default_prefix(args.inp)
    if args.out is None:
        args.out = f"{prefix}_step6e_vxbc_intermediates.npz"
    if args.summary is None:
        args.summary = f"{prefix}_step6e_vxbc_intermediates_summary.txt"

    data = np.load(args.inp, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta.get("nobs", np.array(data["Cab_obs"]).shape[0]))
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))
    nocc = int(args.nocc)
    enuc = float(meta.get("enuc", 0.0))
    E_obs_fci = float(meta["E_obs_fci"])

    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)
    E_obs_rdm, _, _ = reconstruct_energy(h_ri[:nobs, :nobs], eri_ri[:nobs, :nobs, :nobs, :nobs], dm1_obs, dm2_obs, enuc)
    E_ri_rdm, _, _ = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)

    ints = build_intermediate_matrices(data, nri, nobs, nocc)
    idx_q3 = ints["indices"]["q_ansatz3"]
    sources = source_vectors(data, nri, nobs, idx_q3)
    psi0 = sources["psi0"]

    M = ints["matrices"]
    scalar_intermediates = {
        "V_direct_f12g12": cross_block("V_direct_f12g12", M["V_direct_f12g12"], psi0, sources),
        "X_direct_f12sq": scalar_block("X_direct_f12sq", M["X_direct_f12sq"], sources),
        "B_direct_f12dc": scalar_block("B_direct_f12dc", M["B_direct_f12dc"], sources),
        "C_fock_model": scalar_block("C_fock_model", M["C_fock_model"], sources),
        "V_projector_sub_diagnostic": cross_block("V_projector_sub_diagnostic", M["V_projector_sub_diagnostic"], psi0, sources),
        "X_projector_sub_diagnostic": scalar_block("X_projector_sub_diagnostic", M["X_projector_sub_diagnostic"], sources),
    }

    candidate_rows: Dict[str, Any] = {}
    for label in ["psi0", "cab_sp", "A_raw_Q3", "A_sp_Q3"]:
        V = scalar_intermediates["V_direct_f12g12"]["values"][label]
        X = scalar_intermediates["X_direct_f12sq"]["values"][label]
        Bdc = scalar_intermediates["B_direct_f12dc"]["values"][label]
        C = scalar_intermediates["C_fock_model"]["values"][label]
        B_plus = X + Bdc
        B_plus_C = X + Bdc + C
        candidate_rows[label] = {
            "V": V,
            "X": X,
            "B_dc": Bdc,
            "C_fock_model": C,
            "B_X_plus_dc": B_plus,
            "B_X_plus_dc_plus_C": B_plus_C,
            "minus_V2_over_X_plus_dc": safe_div(-V * V, B_plus, args.denom_thresh),
            "minus_V2_over_X_plus_dc_plus_C": safe_div(-V * V, B_plus_C, args.denom_thresh),
        }

    diagnostics = {
        "input": args.inp,
        "nri": nri,
        "nobs": nobs,
        "nocc": nocc,
        "energy_checks": {
            "E_obs_fci": E_obs_fci,
            "E_obs_rdm": E_obs_rdm,
            "E_ri_embedded_rdm": E_ri_rdm,
            "delta_obs_rdm_minus_fci": E_obs_rdm - E_obs_fci,
            "delta_ri_rdm_minus_fci": E_ri_rdm - E_obs_fci,
        },
        "rdm_diagnostics": {
            "obs": rdm_diagnostics(dm1_obs, dm2_obs),
            "ri": rdm_diagnostics(dm1_ri, dm2_ri),
        },
        "tensor_diagnostics": {
            "V_direct_f12g12": tensor_diagnostics(np.array(data["f12g12_ri"], dtype=float)),
            "X_direct_f12sq": tensor_diagnostics(np.array(data["f12sq_ri"], dtype=float)),
            "B_direct_f12dc": tensor_diagnostics(np.array(data["f12dc_ri"], dtype=float)),
        },
        "projector_dimensions": {
            key: int(len(value)) for key, value in ints["indices"].items()
        },
        "matrix_diagnostics": {
            "C_fock_model_norm": float(np.linalg.norm(M["C_fock_model"])),
            "C_fock_model_asym": maxabs(M["C_fock_model"] - M["C_fock_model"].T),
            "V_projector_sub_norm": float(np.linalg.norm(M["V_projector_sub_diagnostic"])),
            "X_projector_sub_norm": float(np.linalg.norm(M["X_projector_sub_diagnostic"])),
        },
        "scalar_intermediates": scalar_intermediates,
        "candidate_denominator_rows": candidate_rows,
        "important_note": (
            "Step 6e builds explicit V/X/B/C intermediates. Candidate ratios are diagnostics only; "
            "final [2]R12 requires exact approximation-C prefactors, antisymmetrization, and tilde terms."
        ),
    }

    np.savez(
        args.out,
        V_direct_f12g12=M["V_direct_f12g12"],
        X_direct_f12sq=M["X_direct_f12sq"],
        B_direct_f12dc=M["B_direct_f12dc"],
        C_fock_model=M["C_fock_model"],
        V_projector_sub_diagnostic=M["V_projector_sub_diagnostic"],
        X_projector_sub_diagnostic=M["X_projector_sub_diagnostic"],
        Q3=ints["projectors"]["Q3"],
        P_removed=ints["projectors"]["P_removed"],
        metadata_json=np.array(json.dumps(diagnostics, indent=2)),
    )

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6e | explicit V/X/B/C intermediates")
    lines.append("=" * 100)
    lines.append(f"input      = {args.inp}")
    lines.append(f"nri/nobs/nocc = {nri}/{nobs}/{nocc}")
    lines.append(f"E checks   = obs-rdm {E_obs_rdm - E_obs_fci:.3e}, ri-rdm {E_ri_rdm - E_obs_fci:.3e}")
    lines.append("")
    lines.append("[Projector dimensions]")
    for key, value in diagnostics["projector_dimensions"].items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("[V/X/B/C scalar rows]")
    lines.append("| source | V | X | B_dc | C_model | -V^2/(X+B) | -V^2/(X+B+C) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, row in candidate_rows.items():
        def fmt(x):
            return "" if x is None else f"{x:.8e}"
        lines.append(
            f"| {label} | {row['V']:.8e} | {row['X']:.8e} | {row['B_dc']:.8e} "
            f"| {row['C_fock_model']:.8e} | {fmt(row['minus_V2_over_X_plus_dc'])} "
            f"| {fmt(row['minus_V2_over_X_plus_dc_plus_C'])} |"
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
    print(f"  {args.out}")
    print(f"  {args.summary}")

    ok = (
        abs(E_obs_rdm - E_obs_fci) < 1e-10
        and abs(E_ri_rdm - E_obs_fci) < 1e-10
        and all(not d["has_nan"] and not d["has_inf"] for d in diagnostics["tensor_diagnostics"].values())
        and diagnostics["matrix_diagnostics"]["C_fock_model_asym"] < 1e-10
    )
    if not ok:
        print("\nERROR: Step 6e intermediate checks failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()

