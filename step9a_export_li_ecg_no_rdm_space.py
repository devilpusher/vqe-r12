#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 9a: export and verify Li s01+p0 ECG-NO FCI RDMs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from r12_common import maxabs, rdm_diagnostics, reconstruct_energy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--li-dir", default="/mnt/d/vqecodex/lino")
    p.add_argument("--channels", default="s,p")
    p.add_argument("--s-pick", default="0,1")
    p.add_argument("--p-pick", default="0")
    p.add_argument("--d-pick", default="")
    p.add_argument("--f-pick", default="")
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--energy-tol", type=float, default=1e-9)
    p.add_argument("--orth-tol", type=float, default=1e-8)
    p.add_argument("--save-compressed", action="store_true")
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def safe_tag(channels: List[str], picks: Dict[str, List[int]]) -> str:
    parts = ["".join(channels)]
    for ch in channels:
        parts.append(f"{ch}{''.join(str(i) for i in picks[ch])}")
    return "_".join(parts)


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_li_module(li_dir: Path):
    script = li_dir / "li_2rdm_compare.py"
    if not script.exists():
        raise FileNotFoundError(f"Cannot find {script}")
    spec = importlib.util.spec_from_file_location("li_2rdm_compare_external", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_picks(li, picks: Dict[str, List[int]]) -> None:
    for ch in ["s", "p", "d", "f"]:
        li.L_INFO_BASE[ch]["pick"] = list(picks[ch])


def main():
    args = parse_args()
    li_dir = Path(args.li_dir).expanduser().resolve()
    channels = [x.strip() for x in args.channels.split(",") if x.strip()]
    picks = {
        "s": parse_int_list(args.s_pick),
        "p": parse_int_list(args.p_pick),
        "d": parse_int_list(args.d_pick),
        "f": parse_int_list(args.f_pick),
    }
    for ch, vals in picks.items():
        if vals and ch not in channels:
            raise ValueError(f"{ch}_pick={vals} but channel {ch} is not included")

    tag = safe_tag(channels, picks)
    args.out = args.out or f"step9a_li_{tag}_rdm_export.npz"
    args.json = args.json or f"step9a_li_{tag}_rdm_export.json"
    args.summary = args.summary or f"step9a_li_{tag}_rdm_export_summary.txt"

    li = load_li_module(li_dir)
    configure_picks(li, picks)
    with pushd(li_dir):
        ns, mats = li.load_ecg_no_matrices(li.NO_FILES)
        mol, exponents, ao_index = li.build_mol_and_layout(ns, channels)
        S_ao = mol.intor("int1e_ovlp")
        h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        C_obs, labels = li.build_ecg_no_subspace(mats, ns, mol, S_ao, ao_index, channels)
        energy, ci, h_obs, eri_obs = li.run_li_fci(mol, h_ao, C_obs)

    nobs = int(C_obs.shape[1])
    dm1_obs, dm2_obs = li.fci.direct_spin1.make_rdm12(ci, nobs, li.NELEC)
    gamma_so, D_pair, cumulant_pair, pair_list = li.build_spinorbital_rdms_from_ci(ci, nobs, nelec=li.NELEC)
    gamma_spatial = li.spatial_gamma_from_spinorbital(gamma_so, nobs)
    channel_probs = li.channel_probabilities_for_orbitals(C_obs, S_ao, ao_index, channels)
    dom_channels, _ = li.dominant_channel_labels(C_obs, S_ao, ao_index, channels)

    enuc = float(mol.energy_nuc())
    E_rdm, E1, E2 = reconstruct_energy(h_obs, eri_obs, dm1_obs, dm2_obs, enuc=enuc)
    diag = rdm_diagnostics(dm1_obs, dm2_obs)
    nelec = sum(li.NELEC)
    pair_trace = float(np.trace(D_pair))
    checks = {
        "cobs_orth_error": float(maxabs(C_obs.T @ S_ao @ C_obs - np.eye(nobs))),
        "delta_rdm_minus_fci": float(E_rdm - energy),
        "trace_dm1_error": float(diag["trace_dm1"] - nelec),
        "trace_dm2_error": float(diag["trace_dm2"] - nelec * (nelec - 1)),
        "pair_trace_error": float(pair_trace - nelec * (nelec - 1) / 2.0),
        "max_dm1_minus_spinorbital_gamma": float(maxabs(dm1_obs - gamma_spatial)),
        "dm2_bra_ket_error": float(diag["max_dm2_bra_ket_error"]),
    }

    failures = []
    if checks["cobs_orth_error"] > args.orth_tol:
        failures.append("cobs_orth")
    if abs(checks["delta_rdm_minus_fci"]) > args.energy_tol:
        failures.append("energy_reconstruction")
    for key in ["trace_dm1_error", "trace_dm2_error", "pair_trace_error", "max_dm1_minus_spinorbital_gamma"]:
        if abs(checks[key]) > 1e-10:
            failures.append(key)
    if checks["dm2_bra_ket_error"] > 1e-10:
        failures.append("dm2_bra_ket")

    metadata: Dict[str, Any] = {
        "step": "9a",
        "system": "Li",
        "state": "1s^2 2s 2S doublet",
        "li_dir": str(li_dir),
        "channels": channels,
        "picks": picks,
        "selection_note": "Default s[0,1]+p[0] gives 5 spatial orbitals = 10 qubits.",
        "ns": int(ns),
        "nao": int(mol.nao),
        "nobs": nobs,
        "nelec": list(li.NELEC),
        "labels": labels,
        "dominant_channels": dom_channels,
        "enuc": enuc,
        "energies": {"E_fci": float(energy), "E_rdm": float(E_rdm), "E1": float(E1), "E2": float(E2)},
        "checks": checks,
        "passed": not failures,
        "failures": failures,
        "rdm_obs_diag": diag,
        "pair_trace": pair_trace,
    }

    save_dict = {
        "S_ao": S_ao,
        "h_ao": h_ao,
        "C_obs": C_obs,
        "h_obs": h_obs,
        "eri_obs": eri_obs,
        "dm1_obs": dm1_obs,
        "dm2_obs": dm2_obs,
        "gamma_spinorbital": gamma_so,
        "gamma_spatial_from_spinorbital": gamma_spatial,
        "D_pair": D_pair,
        "cumulant_pair": cumulant_pair,
        "pair_list": np.array(pair_list, dtype=int),
        "channel_probs": channel_probs,
        "labels": np.array(labels, dtype=object),
        "channels": np.array(channels, dtype=object),
        "E_obs_fci": np.array(energy),
        "E_obs_rdm": np.array(E_rdm),
        "Enuc": np.array(enuc),
        "metadata_json": np.array(json.dumps(metadata, indent=2)),
    }
    save_func = np.savez_compressed if args.save_compressed else np.savez
    save_func(args.out, **save_dict)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    lines = [
        "=" * 96,
        "Step 9a | Li selected ECG-NO RDM export and energy reconstruction",
        "=" * 96,
        f"li_dir       = {li_dir}",
        f"channels     = {channels}",
        f"picks        = {picks}",
        f"ns / nao     = {ns} / {mol.nao}",
        f"nobs         = {nobs}  ({2 * nobs} qubits)",
        f"labels       = {labels}",
        "",
        "[Energy reconstruction]",
        f"E_FCI                            = {energy: .14f} Eh",
        f"E_RDM                            = {E_rdm: .14f} Eh",
        f"Delta(RDM-FCI)                   = {E_rdm - energy: .3e} Eh",
        f"E1 / E2                          = {E1: .14f} / {E2: .14f} Eh",
        "",
        "[RDM]",
        f"Tr(dm1_obs)                      = {diag['trace_dm1']:.12f}",
        f"Tr(dm2_obs)                      = {diag['trace_dm2']:.12f}",
        f"Tr(D_pair)                       = {pair_trace:.12f}",
        f"Max|dm1-gamma_spatial|           = {checks['max_dm1_minus_spinorbital_gamma']:.3e}",
        f"dm2 bra-ket error                = {diag['max_dm2_bra_ket_error']:.3e}",
        "Natural occupations              = " + np.array2string(np.array(diag["natural_occupations"]), precision=10),
        "",
        f"passed                           = {metadata['passed']}",
    ]
    if failures:
        lines.append(f"failures                         = {failures}")
    lines.extend(["", "[Saved]", f"  {args.out}", f"  {args.json}", f"  {args.summary}"])
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
