#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8h: Generate HEM triplet ECG-NO RDMs for selected even s/p spaces.

This script rebuilds the selected ECG-NO orbital space from local HEM NO files
and solves the two-alpha-electron FCI problem in that selected OBS space.  It is
the local, reproducible counterpart to the fixed `hem_2rdm_compare.py` output,
but limited to the arrays needed by the Step8 parent-basis R12 bridge.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
from pyscf import ao2mo, fci
from pyscf.fci import cistring

from r12_common import maxabs, rdm_diagnostics, reconstruct_energy
from step7b_export_ecg_no_orbitals import (
    L_INFO,
    build_ao_index,
    build_ecg_no_cobs,
    build_pyscf_mol,
    even_tempered_exponents,
    load_no_mats,
    parse_int_list,
)
from step8a_export_hem_triplet_data import triplet_rdms_from_upper_pair
from step8b_build_hem_triplet_step4b_like import HEM_GTO_PARAMS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-dir", default="local_external/he-meta")
    p.add_argument("--channels", default="s,p")
    p.add_argument("--s-pick", default="0,1")
    p.add_argument("--p-pick", default="0,1")
    p.add_argument("--d-pick", default="")
    p.add_argument("--f-pick", default="")
    p.add_argument("--prefix", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--require-even-sp", action="store_true", default=True)
    return p.parse_args()


def tag_for(channels: List[str], picks: Dict[str, List[int]]) -> str:
    parts = ["".join(channels)]
    for ch in channels:
        parts.append(f"{ch}{''.join(str(i) for i in picks[ch])}")
    return "_".join(parts)


def transform_integrals(mol, h_ao: np.ndarray, C: np.ndarray):
    n = C.shape[1]
    h = C.T @ h_ao @ C
    eri = ao2mo.kernel(mol, (C, C, C, C), aosym="s1", compact=False)
    eri = eri.reshape(n, n, n, n)
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return h, eri


def ci_to_pair_coeff_upper(ci: np.ndarray, nmo: int) -> np.ndarray:
    strs_a = cistring.gen_strings4orblist(range(nmo), 2)
    ci2 = np.asarray(ci).reshape(len(strs_a), 1)
    B = np.zeros((nmo, nmo), dtype=float)
    for I, st in enumerate(strs_a):
        occ = [i for i in range(nmo) if (int(st) >> i) & 1]
        p, q = sorted(occ)
        B[p, q] = float(ci2[I, 0])
    norm = float(np.sqrt(np.sum(B * B)))
    if norm <= 0.0:
        raise RuntimeError("zero pair coefficient norm")
    return B / norm


def check_even_space(channels: List[str], picks: Dict[str, List[int]]) -> List[str]:
    failures = []
    if "s" in channels and len(picks["s"]) % 2 != 0:
        failures.append("s radial pick count is not even")
    if "p" in channels and len(picks["p"]) % 2 != 0:
        failures.append("p radial pick count is not even")
    return failures


def main():
    args = parse_args()
    channels = [x.strip() for x in args.channels.split(",") if x.strip()]
    picks = {
        "s": parse_int_list(args.s_pick),
        "p": parse_int_list(args.p_pick),
        "d": parse_int_list(args.d_pick),
        "f": parse_int_list(args.f_pick),
    }
    for ch in channels:
        if ch not in L_INFO:
            raise ValueError(f"Unknown channel {ch}")
    failures = check_even_space(channels, picks) if args.require_even_sp else []
    if failures:
        raise SystemExit(f"Space parity check failed: {failures}")

    tag = tag_for(channels, picks)
    prefix = args.prefix or f"step8h_hem_triplet_{tag}"
    args.out = args.out or f"{prefix}_rdm_export.npz"
    args.json = args.json or f"{prefix}_rdm_export.json"
    args.summary = args.summary or f"{prefix}_rdm_export_summary.txt"

    ns, mats = load_no_mats(args.no_dir)
    exponents = {ch: even_tempered_exponents(*HEM_GTO_PARAMS[ch], ns) for ch in L_INFO}
    mol = build_pyscf_mol(ns, channels, exponents)
    mol.spin = 2
    mol.nelectron = 2
    S = mol.intor("int1e_ovlp")
    h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    ao_index = build_ao_index(ns, channels)
    C_obs, labels = build_ecg_no_cobs(mats, ns, mol, S, ao_index, channels, picks)
    nobs = C_obs.shape[1]
    h, eri = transform_integrals(mol, h_ao, C_obs)
    E_fci, ci = fci.direct_spin1.kernel(h, eri, nobs, nelec=(2, 0), conv_tol=1e-12)
    B = ci_to_pair_coeff_upper(ci, nobs)
    dm1, dm2, A, D_ordered = triplet_rdms_from_upper_pair(B)
    E_rdm, E1, E2 = reconstruct_energy(h, eri, dm1, dm2, enuc=0.0)

    diag = rdm_diagnostics(dm1, dm2)
    checks = {
        "cobs_orth_error": maxabs(C_obs.T @ S @ C_obs - np.eye(nobs)),
        "pair_coeff_upper_norm": float(np.sqrt(np.sum(B * B))),
        "trace_dm1_error": float(diag["trace_dm1"] - 2.0),
        "trace_dm2_error": float(diag["trace_dm2"] - 2.0),
        "dm2_bra_ket_error": float(diag["max_dm2_bra_ket_error"]),
        "delta_rdm_minus_fci": float(E_rdm - E_fci),
        "natural_occupations": diag["natural_occupations"],
    }
    fail = []
    if checks["cobs_orth_error"] > 1e-8:
        fail.append("cobs_orth")
    if abs(checks["trace_dm1_error"]) > 1e-10:
        fail.append("dm1_trace")
    if abs(checks["trace_dm2_error"]) > 1e-10:
        fail.append("dm2_trace")
    if abs(checks["delta_rdm_minus_fci"]) > 1e-9:
        fail.append("energy_reconstruction")
    checks["passed"] = not fail
    checks["failures"] = fail

    metadata: Dict[str, Any] = {
        "step": "8h",
        "system": "He metastable triplet",
        "state": "1s2s 3S, Ms=1",
        "method": "ECG-NO selected-space alpha-alpha FCI",
        "no_dir": str(args.no_dir),
        "channels": channels,
        "picks": picks,
        "nobs": int(nobs),
        "nelec": 2,
        "spin_sector": "alpha-alpha, nelec=(2,0)",
        "input_metrics": {"energy_Ha": float(E_fci), "E_rdm": float(E_rdm), "E1": float(E1), "E2": float(E2)},
        "checks": checks,
    }
    np.savez(
        args.out,
        method=np.array("ECG-NO-FCI"),
        labels=np.array(labels, dtype=object),
        ao_channels=np.array(channels, dtype=object),
        pair_coeff_upper=B,
        pair_coeff_antisym=A,
        D_ordered_alpha_alpha=D_ordered,
        gamma_input=dm1,
        dm1_obs=dm1,
        dm2_obs=dm2,
        metrics_json=np.array(json.dumps(metadata, indent=2)),
    )
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    lines = [
        "=" * 92,
        "Step 8h | Generate HEM triplet selected-space RDM",
        "=" * 92,
        f"no_dir      = {args.no_dir}",
        f"channels    = {channels}",
        f"picks       = {picks}",
        f"nobs        = {nobs} ({2*nobs} qubits)",
        f"E_FCI       = {E_fci:.14f} Eh",
        f"E_RDM       = {E_rdm:.14f} Eh",
        f"Delta       = {E_rdm-E_fci:.3e} Eh",
        f"Tr(dm1/2)   = {diag['trace_dm1']:.12f} / {diag['trace_dm2']:.12f}",
        "Natural occ = " + np.array2string(np.array(diag["natural_occupations"]), precision=10),
        f"passed      = {checks['passed']}",
        "",
        "[Saved]",
        f"  {args.out}",
        f"  {args.json}",
        f"  {args.summary}",
    ]
    if fail:
        lines.insert(-4, f"failures    = {fail}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
