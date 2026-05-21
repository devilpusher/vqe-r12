#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6d: formula map and projector audit for the He [2]R12 prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from r12_common import maxabs, pair_matrix, pair_projector, pp_pair_indices, q_pair_indices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="he_ccpvdz_nobs2_fitN7_step5a_r12_intermediates.npz")
    p.add_argument("--nocc", type=int, default=1)
    p.add_argument("--out-json", default=None)
    p.add_argument("--summary", default=None)
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
            is_rs = p in obs and q in obs
            is_a_j = p in cabs and q in occ
            is_i_b = p in occ and q in cabs
            if is_rs:
                rs_obs.append(idx)
            if is_a_j:
                a_prime_j.append(idx)
            if is_i_b:
                i_b_prime.append(idx)
            if not (is_rs or is_a_j or is_i_b):
                q3.append(idx)
    return {
        "rs_obs": np.array(rs_obs, dtype=int),
        "a_prime_j": np.array(a_prime_j, dtype=int),
        "i_b_prime": np.array(i_b_prime, dtype=int),
        "q_ansatz3": np.array(q3, dtype=int),
    }


def projector_report(nri: int, nobs: int, nocc: int) -> Dict[str, Any]:
    idx = ansatz3_indices(nri, nobs, nocc)
    pair_dim = nri * nri

    projectors = {key: pair_projector(nri, value) for key, value in idx.items()}
    P_removed = projectors["rs_obs"] + projectors["a_prime_j"] + projectors["i_b_prime"]
    Q3 = projectors["q_ansatz3"]
    I = np.eye(pair_dim)

    overlaps = {}
    keys = ["rs_obs", "a_prime_j", "i_b_prime", "q_ansatz3"]
    for a in keys:
        for b in keys:
            if a >= b:
                continue
            overlaps[f"{a}__{b}"] = float(np.trace(projectors[a] @ projectors[b]))

    return {
        "indices": {key: value.tolist() for key, value in idx.items()},
        "dimensions": {
            "nri": nri,
            "nobs": nobs,
            "nocc": nocc,
            "ncabs": nri - nobs,
            "pair_dim": pair_dim,
            "dim_rs_obs": int(len(idx["rs_obs"])),
            "dim_a_prime_j": int(len(idx["a_prime_j"])),
            "dim_i_b_prime": int(len(idx["i_b_prime"])),
            "dim_q_ansatz3": int(len(idx["q_ansatz3"])),
        },
        "idempotency": {
            "P_removed": maxabs(P_removed @ P_removed - P_removed),
            "Q_ansatz3": maxabs(Q3 @ Q3 - Q3),
            "P_plus_Q_minus_I": maxabs(P_removed + Q3 - I),
            "P_Q_overlap": maxabs(P_removed @ Q3),
        },
        "block_overlaps_trace": overlaps,
    }


def tensor_closure_report(data, projector: Dict[str, Any]) -> Dict[str, Any]:
    nri = projector["dimensions"]["nri"]
    idx = {key: np.array(vals, dtype=int) for key, vals in projector["indices"].items()}
    P_rs = pair_projector(nri, idx["rs_obs"])
    P_aj = pair_projector(nri, idx["a_prime_j"])
    P_ib = pair_projector(nri, idx["i_b_prime"])
    Q3 = pair_projector(nri, idx["q_ansatz3"])

    G = pair_matrix(np.array(data["eri_ri"], dtype=float))
    F = pair_matrix(np.array(data["f12_ri"], dtype=float))
    F2_direct = pair_matrix(np.array(data["f12sq_ri"], dtype=float))
    GF_direct = pair_matrix(np.array(data["f12g12_ri"], dtype=float))
    DC_direct = pair_matrix(np.array(data["f12dc_ri"], dtype=float))

    P_removed = P_rs + P_aj + P_ib
    GF_closure_full = G @ F
    F2_closure_full = F @ F
    GF_closure_q3 = G @ Q3 @ F
    F2_closure_q3 = F @ Q3 @ F
    GF_projector_sub = GF_direct - G @ P_removed @ F
    F2_projector_sub = F2_direct - F @ P_removed @ F

    return {
        "direct_tensor_symmetry": {
            "f12g12_asym": maxabs(GF_direct - GF_direct.T),
            "f12sq_asym": maxabs(F2_direct - F2_direct.T),
            "f12dc_asym": maxabs(DC_direct - DC_direct.T),
        },
        "closure_errors": {
            "maxabs_f12g12_direct_minus_gf_closure_full": maxabs(GF_direct - GF_closure_full),
            "maxabs_f12sq_direct_minus_ff_closure_full": maxabs(F2_direct - F2_closure_full),
            "maxabs_projector_sub_gf_minus_gq3f": maxabs(GF_projector_sub - GF_closure_q3),
            "maxabs_projector_sub_ff_minus_fq3f": maxabs(F2_projector_sub - F2_closure_q3),
        },
        "norms": {
            "f12g12_direct": float(np.linalg.norm(GF_direct)),
            "f12sq_direct": float(np.linalg.norm(F2_direct)),
            "f12dc_direct": float(np.linalg.norm(DC_direct)),
            "gf_closure_full": float(np.linalg.norm(GF_closure_full)),
            "ff_closure_full": float(np.linalg.norm(F2_closure_full)),
            "gq3f_closure": float(np.linalg.norm(GF_closure_q3)),
            "fq3f_closure": float(np.linalg.norm(F2_closure_q3)),
        },
    }


def amplitude_report(data, projector: Dict[str, Any]) -> Dict[str, Any]:
    nri = projector["dimensions"]["nri"]
    nobs = projector["dimensions"]["nobs"]
    idx_q3 = np.array(projector["indices"]["q_ansatz3"], dtype=int)
    q_old = q_pair_indices(nri, nobs)
    Q3 = pair_projector(nri, idx_q3)
    Q_old = pair_projector(nri, q_old)

    A_raw = np.array(data["A_raw_Q"], dtype=float).reshape(-1)
    A_sp = np.array(data["A_sp_Q"], dtype=float).reshape(-1)

    return {
        "old_Q_vs_ansatz3": {
            "dim_old_Q": int(len(q_old)),
            "dim_ansatz3_Q": int(len(idx_q3)),
            "old_Q_minus_ansatz3_Q_projector_norm": float(np.linalg.norm(Q_old - Q3)),
        },
        "amplitude_leakage": {
            "A_raw_norm": float(np.linalg.norm(A_raw)),
            "A_raw_outside_ansatz3_Q_norm": float(np.linalg.norm((np.eye(nri * nri) - Q3) @ A_raw)),
            "A_sp_norm": float(np.linalg.norm(A_sp)),
            "A_sp_outside_ansatz3_Q_norm": float(np.linalg.norm((np.eye(nri * nri) - Q3) @ A_sp)),
        },
    }


def main():
    args = parse_args()
    prefix = default_prefix(args.inp)
    if args.out_json is None:
        args.out_json = f"{prefix}_step6d_formula_projector_audit.json"
    if args.summary is None:
        args.summary = f"{prefix}_step6d_formula_projector_audit_summary.txt"

    data = np.load(args.inp, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta.get("nobs", np.array(data["Cab_obs"]).shape[0]))
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))
    nocc = int(args.nocc)
    if nocc < 1 or nocc > nobs:
        raise ValueError(f"nocc must satisfy 1 <= nocc <= nobs; got nocc={nocc}, nobs={nobs}")

    projector = projector_report(nri, nobs, nocc)
    closure = tensor_closure_report(data, projector)
    amps = amplitude_report(data, projector)

    report = {
        "input": args.inp,
        "formula_source_summary": {
            "paper": "Schleich/Kottmann/Aspuru-Guzik VQE + perturbative [2]R12 correction.",
            "psi4_anchor": "Psi4 MP2-F12 theory documentation for 3C(FIX)/SP notation.",
            "projector": "Q12 = 1 - |a'j><a'j| - |ib'><ib'| - |rs><rs|",
            "sp_ansatz": "d[p,q,r,s] = 3/8 delta[p,r]delta[q,s] + 1/8 delta[p,s]delta[q,r]",
        },
        "projector": projector,
        "tensor_closure": closure,
        "amplitudes": amps,
        "implementation_decision": [
            "Use direct Psi4 f12g12/f12sq/f12dc tensors as authoritative integral inputs.",
            "Keep finite RI matrix closures only as diagnostics.",
            "Switch future Step 6e/6f amplitudes from old Q to Ansatz-3 Q before final energy candidates.",
            "C_ab/orbital-denominator terms still need explicit construction before final 3C(FIX) energy.",
        ],
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6d | formula map and projector audit")
    lines.append("=" * 100)
    dims = projector["dimensions"]
    lines.append(f"input      = {args.inp}")
    lines.append(f"nri/nobs/nocc = {nri}/{nobs}/{nocc}")
    lines.append(f"pair dim   = {dims['pair_dim']}")
    lines.append(f"dim Q(old) = {amps['old_Q_vs_ansatz3']['dim_old_Q']}")
    lines.append(f"dim Q3     = {dims['dim_q_ansatz3']}")
    lines.append("")
    lines.append("[Projector checks]")
    for key, value in projector["idempotency"].items():
        lines.append(f"{key}: {value:.3e}")
    lines.append("")
    lines.append("[Amplitude leakage relative to Ansatz-3 Q]")
    leak = amps["amplitude_leakage"]
    lines.append(f"A_raw outside Q3 norm = {leak['A_raw_outside_ansatz3_Q_norm']:.12e} / {leak['A_raw_norm']:.12e}")
    lines.append(f"A_sp  outside Q3 norm = {leak['A_sp_outside_ansatz3_Q_norm']:.12e} / {leak['A_sp_norm']:.12e}")
    lines.append("")
    lines.append("[Direct-integral vs finite-closure diagnostics]")
    for key, value in closure["closure_errors"].items():
        lines.append(f"{key}: {value:.3e}")
    lines.append("")
    lines.append("[Decision]")
    for item in report["implementation_decision"]:
        lines.append(f"- {item}")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print("\n".join(lines))
    print("\n[Saved]")
    print(f"  {args.out_json}")
    print(f"  {args.summary}")

    ok = (
        projector["idempotency"]["P_removed"] < 1e-12
        and projector["idempotency"]["Q_ansatz3"] < 1e-12
        and projector["idempotency"]["P_Q_overlap"] < 1e-12
    )
    if not ok:
        print("\nERROR: projector audit failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()

