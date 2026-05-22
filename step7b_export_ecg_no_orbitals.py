#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7b: Rebuild ECG-NO C_obs and audit PySCF/Psi4 AO ordering.

Step 7a exports the ECG-NO RDMs.  This step reconstructs the matching ECG-NO
orbital coefficient matrix in the PySCF even-tempered AO basis and compares the
same AO basis as built in Psi4.

The output is still an audit/bridge artifact, not the final R12 input.  The
next step should consume this file only after the AO overlap checks show that
PySCF and Psi4 use a compatible AO ordering for the selected basis.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from r12_common import maxabs


GTO_PARAMS = {
    "s": (0.11138001, 2.40586399),
    "p": (0.13439018, 2.43934942),
    "d": (0.20214187, 2.44141981),
    "f": (0.30207461, 2.33570137),
}

L_INFO = {
    "s": {"l": 0, "L": "S", "deg": 1},
    "p": {"l": 1, "L": "P", "deg": 3},
    "d": {"l": 2, "L": "D", "deg": 5},
    "f": {"l": 3, "L": "F", "deg": 7},
}

DEFAULT_PICKS = {
    "s": [0, 1, 2],
    "p": [0, 1],
    "d": [0],
    "f": [],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-dir", default="local_external/he-sin", help="Local directory containing no.dat/nop.dat/nod.dat/nof.dat.")
    p.add_argument("--channels", default="s,p,d", help="Comma-separated AO channels to include.")
    p.add_argument("--s-pick", default="0,1,2")
    p.add_argument("--p-pick", default="0,1")
    p.add_argument("--d-pick", default="0")
    p.add_argument("--f-pick", default="")
    p.add_argument("--basis-name", default="heecgnoet")
    p.add_argument("--out", default="step7b_ecg_no_orbitals.npz")
    p.add_argument("--summary", default="step7b_ecg_no_orbitals_summary.txt")
    p.add_argument("--json", default="step7b_ecg_no_orbitals.json")
    p.add_argument("--memory", default="2 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--psi4-output", default="psi4_step7b_ecg_no_basis_audit.out")
    p.add_argument("--direct-overlap-tol", type=float, default=1e-8)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def read_no_fortran_cols(path: str | Path) -> Tuple[int, np.ndarray]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tokens = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?", text)
    if not tokens:
        raise ValueError(f"{path} is empty or unreadable")
    tokens = [t.replace("D", "E").replace("d", "E") for t in tokens]
    ns = int(float(tokens[0]))
    vals = np.array([float(x) for x in tokens[1:]], dtype=float)
    if vals.size % ns != 0:
        raise ValueError(f"{path}: data length {vals.size} not divisible by NS={ns}")
    return ns, vals.reshape(ns, -1, order="F")


def dedup_columns(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    keep = []
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        vn = v / (np.linalg.norm(v) + 1e-300)
        is_new = True
        for w in keep:
            wn = w / (np.linalg.norm(w) + 1e-300)
            if min(np.linalg.norm(vn - wn), np.linalg.norm(vn + wn)) < tol:
                is_new = False
                break
        if is_new:
            keep.append(v)
    return np.stack(keep, axis=1) if keep else A[:, :0]


def even_tempered_exponents(a: float, b: float, ns: int) -> np.ndarray:
    return np.array([a * (b ** n) for n in range(ns)], dtype=float)


def load_no_mats(no_dir: str | Path) -> Tuple[int, Dict[str, np.ndarray]]:
    no_dir = Path(no_dir)
    files = {"s": "no.dat", "p": "nop.dat", "d": "nod.dat", "f": "nof.dat"}
    mats = {}
    ns_values = []
    for ch, fn in files.items():
        ns, mat = read_no_fortran_cols(no_dir / fn)
        if ch in ("p", "d", "f"):
            mat = dedup_columns(mat)
        mats[ch] = mat
        ns_values.append(ns)
    if len(set(ns_values)) != 1:
        raise ValueError(f"NS mismatch: {dict(zip(files, ns_values))}")
    return ns_values[0], mats


def build_pyscf_mol(ns: int, channels: List[str], exponents: Dict[str, np.ndarray]):
    from pyscf import gto

    basis = []
    for ch in channels:
        l = L_INFO[ch]["l"]
        for alpha in exponents[ch]:
            basis.append([l, (float(alpha), 1.0)])
    mol = gto.Mole()
    mol.atom = "He 0 0 0"
    mol.unit = "Bohr"
    mol.spin = 0
    mol.charge = 0
    mol.cart = False
    mol.basis = {"He": basis}
    mol.build()
    return mol


def build_ao_index(ns: int, channels: List[str]) -> Dict[str, Dict[str, np.ndarray | List[List[int]]]]:
    ao_index = {}
    offset = 0
    for ch in channels:
        deg = L_INFO[ch]["deg"]
        comp_lists = [[] for _ in range(deg)]
        all_idx = []
        for k in range(ns):
            base = offset + deg * k
            for m in range(deg):
                idx = base + m
                comp_lists[m].append(idx)
                all_idx.append(idx)
        ao_index[ch] = {"components": comp_lists, "all": np.array(all_idx, dtype=int)}
        offset += deg * ns
    return ao_index


def s_orthonormalize(C: np.ndarray, S: np.ndarray, labels: List[str], eps: float = 1e-12) -> np.ndarray:
    Q = np.zeros_like(C, dtype=float)
    for j in range(C.shape[1]):
        v = C[:, j].copy()
        for k in range(j):
            v -= (Q[:, k].T @ S @ v) * Q[:, k]
        n2 = float(v.T @ S @ v)
        if n2 < eps:
            raise RuntimeError(f"S-metric linear dependence at column {j} ({labels[j]}), norm2={n2:.3e}")
        Q[:, j] = v / np.sqrt(n2)
    return Q


def build_ecg_no_cobs(mats, ns, mol, S, ao_index, channels, picks):
    cols = []
    labels = []
    for ch in channels:
        mat = mats[ch]
        deg = L_INFO[ch]["deg"]
        for j in picks[ch]:
            if j < 0 or j >= mat.shape[1]:
                raise ValueError(f"{ch}_pick index {j} outside available columns {mat.shape[1]}")
            for m in range(deg):
                v = np.zeros(mol.nao)
                idx_m = ao_index[ch]["components"][m]
                for i, coeff in enumerate(mat[:, j]):
                    v[idx_m[i]] = coeff
                cols.append(v)
                labels.append(f"ECG-{ch}{j}_m{m}")
    if not cols:
        raise RuntimeError("No ECG-NO columns selected")
    C = np.stack(cols, axis=1)
    return s_orthonormalize(C, S, labels), labels


def psi4_basis_string(name: str, channels: List[str], exponents: Dict[str, np.ndarray]) -> str:
    lines = [
        f"assign {name}",
        f"[{name}]",
        "spherical",
        "****",
        "He 0",
    ]
    for ch in channels:
        shell = L_INFO[ch]["L"]
        for alpha in exponents[ch]:
            lines.append(f"{shell} 1 1.0")
            lines.append(f"  {float(alpha):.16g}  1.0")
    lines.append("****")
    return "\n".join(lines) + "\n"


def build_psi4_overlap(name: str, basis_text: str, memory: str, nthreads: int, output_file: str) -> np.ndarray:
    import psi4

    psi4.core.clean()
    psi4.set_memory(memory)
    psi4.set_num_threads(nthreads)
    psi4.core.set_output_file(output_file, False)
    mol = psi4.geometry(
        """
        0 1
        He 0.0 0.0 0.0
        symmetry c1
        units bohr
        """
    )
    psi4.basis_helper(basis_text, name=name, set_option=True)
    psi4.set_options({"basis": name, "reference": "rhf", "scf_type": "pk"})
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    mints = psi4.core.MintsHelper(wfn.basisset())
    return np.array(np.asarray(mints.ao_overlap()), dtype=float, copy=True)


def main():
    args = parse_args()
    channels = [x.strip() for x in args.channels.split(",") if x.strip()]
    for ch in channels:
        if ch not in L_INFO:
            raise ValueError(f"Unknown channel {ch}")
    picks = {
        "s": parse_int_list(args.s_pick),
        "p": parse_int_list(args.p_pick),
        "d": parse_int_list(args.d_pick),
        "f": parse_int_list(args.f_pick),
    }
    for ch, vals in picks.items():
        if vals and ch not in channels:
            raise ValueError(f"{ch}_pick={vals} but channel {ch} is not included")

    ns, mats = load_no_mats(args.no_dir)
    exponents = {ch: even_tempered_exponents(*GTO_PARAMS[ch], ns) for ch in L_INFO}
    mol = build_pyscf_mol(ns, channels, exponents)
    S_pyscf = mol.intor("int1e_ovlp")
    ao_index = build_ao_index(ns, channels)
    C_obs, labels = build_ecg_no_cobs(mats, ns, mol, S_pyscf, ao_index, channels, picks)

    basis_text = psi4_basis_string(args.basis_name, channels, exponents)
    S_psi4 = build_psi4_overlap(args.basis_name, basis_text, args.memory, args.nthreads, args.psi4_output)

    direct_overlap_error = maxabs(S_pyscf - S_psi4) if S_pyscf.shape == S_psi4.shape else None
    eig_pyscf = np.linalg.eigvalsh(0.5 * (S_pyscf + S_pyscf.T))
    eig_psi4 = np.linalg.eigvalsh(0.5 * (S_psi4 + S_psi4.T))
    eig_error = maxabs(np.sort(eig_pyscf) - np.sort(eig_psi4)) if eig_pyscf.shape == eig_psi4.shape else None
    cobs_orth_pyscf = maxabs(C_obs.T @ S_pyscf @ C_obs - np.eye(C_obs.shape[1]))
    cobs_orth_psi4_same_order = (
        maxabs(C_obs.T @ S_psi4 @ C_obs - np.eye(C_obs.shape[1]))
        if S_pyscf.shape == S_psi4.shape
        else None
    )
    same_order = (
        direct_overlap_error is not None
        and direct_overlap_error < args.direct_overlap_tol
        and cobs_orth_psi4_same_order is not None
        and cobs_orth_psi4_same_order < 1e-8
    )

    audit = {
        "no_dir": str(args.no_dir),
        "channels": channels,
        "picks": picks,
        "ns": ns,
        "nao_pyscf": int(S_pyscf.shape[0]),
        "nao_psi4": int(S_psi4.shape[0]),
        "nobs": int(C_obs.shape[1]),
        "basis_name": args.basis_name,
        "gto_params": GTO_PARAMS,
        "direct_overlap_error": direct_overlap_error,
        "overlap_eigenvalue_error": eig_error,
        "cobs_orth_pyscf": cobs_orth_pyscf,
        "cobs_orth_psi4_same_order": cobs_orth_psi4_same_order,
        "same_ao_order_supported": same_order,
        "labels": labels,
        "basis_text": basis_text,
        "warning": None if same_order else "PySCF/Psi4 AO overlap is not directly identical; Step 7c must resolve AO permutation/sign/rotation before using C_obs with Psi4 tensors.",
    }

    np.savez(
        args.out,
        C_obs_pyscf=C_obs,
        C_obs_psi4_same_order=C_obs if same_order else np.array([]),
        S_pyscf=S_pyscf,
        S_psi4=S_psi4,
        labels=np.array(labels, dtype=object),
        channels=np.array(channels, dtype=object),
        basis_text=np.array(basis_text),
        metadata_json=np.array(json.dumps(audit, indent=2)),
    )
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    lines = []
    lines.append("=" * 96)
    lines.append("Step 7b | ECG-NO C_obs rebuild and PySCF/Psi4 AO overlap audit")
    lines.append("=" * 96)
    lines.append(f"no_dir       = {args.no_dir}")
    lines.append(f"channels     = {channels}")
    lines.append(f"picks        = {picks}")
    lines.append(f"ns / nao     = {ns} / {S_pyscf.shape[0]}")
    lines.append(f"nobs         = {C_obs.shape[1]}")
    lines.append("")
    lines.append("[PySCF C_obs]")
    lines.append(f"Max|C_obs^T S_pyscf C_obs - I| = {cobs_orth_pyscf:.3e}")
    lines.append(f"labels first 8 = {labels[:8]}")
    lines.append("")
    lines.append("[PySCF/Psi4 AO overlap audit]")
    lines.append(f"S_pyscf shape              = {S_pyscf.shape}")
    lines.append(f"S_psi4 shape               = {S_psi4.shape}")
    lines.append(f"Max|S_pyscf - S_psi4|      = {direct_overlap_error}")
    lines.append(f"sorted eig max error       = {eig_error}")
    lines.append(f"C_obs orth in Psi4 order   = {cobs_orth_psi4_same_order}")
    lines.append(f"same AO order supported    = {same_order}")
    if audit["warning"]:
        lines.append(f"warning                    = {audit['warning']}")
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out}")
    lines.append(f"  {args.json}")
    lines.append(f"  {args.summary}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
