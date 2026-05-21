#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Psi4 DETCI -> RDM -> energy reconstruction, using He as the first test.

Important
---------
Do NOT use He/STO-3G for this DETCI test.  He/STO-3G has only one spatial
orbital, hence only one closed-shell determinant.  Psi4 DETCI requires at
least two determinants and will stop with:

    DETCI requires at least two determinants

The default basis is therefore 6-31G.

Purpose
-------
This script prepares a stable RDM reference for later ECG-NO + [2]R12 work.

It runs Psi4 DETCI for He, obtains MO integrals from Psi4, then builds an
independent two-electron alpha-beta FCI/RDM in the same MO basis.  This avoids
version-dependent Psi4 DETCI RDM access while preserving the intended interface:
Psi4 supplies the CI reference/integrals, and the saved output supplies tested
spin-free 1- and 2-RDMs.

RDM convention
--------------
dm1[p,q] = sum_sigma < a^+_{p sigma} a_{q sigma} >

dm2[p,q,r,s] = sum_{sigma,tau}
               < a^+_{p sigma} a^+_{r tau} a_{s tau} a_{q sigma} >

Energy convention:
E = sum_pq h[p,q] dm1[p,q] + 1/2 sum_pqrs eri[p,q,r,s] dm2[p,q,r,s] + Enuc

eri[p,q,r,s] = (p q | r s), chemist notation.

Usage
-----
    python step1_psi4_he_detci_rdm_check.py
    python step1_psi4_he_detci_rdm_check.py --basis cc-pvdz
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Tuple, Dict, Any

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--basis",
        default="6-31g",
        help="Orbital basis for Psi4 DETCI. Default is 6-31g; He/STO-3G is too small for DETCI.",
    )
    p.add_argument("--memory", default="2 GB", help="Psi4 memory, e.g. '2 GB'.")
    p.add_argument("--nthreads", type=int, default=1, help="Number of Psi4 threads.")
    p.add_argument("--out", default=None, help="Output npz file. Default depends on basis.")
    p.add_argument("--summary", default=None, help="Text summary file. Default depends on basis.")
    p.add_argument("--psi4-output", default=None, help="Psi4 output file. Default depends on basis.")
    p.add_argument(
        "--allow-single-determinant",
        action="store_true",
        help="Allow single-determinant bases by falling back to SCF integrals + pair-FCI. "
             "This is only a trivial RDM convention check, not a DETCI test.",
    )
    return p.parse_args()


def safe_basis_label(basis: str) -> str:
    return basis.lower().replace("*", "s").replace("+", "p").replace("-", "").replace("_", "")


def asarray_psi4_matrix(x):
    return np.array(np.asarray(x), dtype=float, copy=True)


def run_psi4_scf_he(psi4, basis: str, memory: str, nthreads: int, psi4_output: str):
    psi4.core.clean()
    psi4.set_memory(memory)
    psi4.set_num_threads(nthreads)
    psi4.core.set_output_file(psi4_output, False)

    mol = psi4.geometry(
        """
        0 1
        He 0.0 0.0 0.0
        symmetry c1
        units bohr
        """
    )

    psi4.set_options(
        {
            "basis": basis,
            "reference": "rhf",
            "scf_type": "pk",
            "e_convergence": 1e-12,
            "d_convergence": 1e-12,
        }
    )
    e_scf, scf_wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    return float(e_scf), scf_wfn, mol


def run_psi4_detci_he(basis: str, memory: str, nthreads: int, psi4_output: str, allow_single: bool):
    try:
        import psi4
    except Exception as exc:
        raise RuntimeError("Cannot import psi4. Activate the Psi4 environment first.") from exc

    # First run SCF to inspect basis size and provide useful diagnostics.
    e_scf, scf_wfn, mol = run_psi4_scf_he(psi4, basis, memory, nthreads, psi4_output)
    nmo_scf = int(scf_wfn.nmo())
    if nmo_scf < 2 and not allow_single:
        raise RuntimeError(
            f"He/{basis} has only {nmo_scf} spatial orbital. DETCI has only one determinant and Psi4 will fail. "
            f"Use --basis 6-31g or larger. If you only want a trivial convention test, rerun with "
            f"--allow-single-determinant."
        )

    # DETCI with ex_level=2 is FCI for a two-electron system.
    psi4.set_options(
        {
            "detci__ex_level": 2,
            "detci__opdm": True,
            "detci__tpdm": True,
            "detci__num_roots": 1,
            "detci__ci_maxiter": 100,
            "detci__r_convergence": 1e-12,
        }
    )

    if nmo_scf < 2 and allow_single:
        return e_scf, scf_wfn, psi4, mol, "SCF fallback for single-determinant basis"

    try:
        e_detci, wfn = psi4.energy("detci", molecule=mol, return_wfn=True)
        return float(e_detci), wfn, psi4, mol, "Psi4 DETCI"
    except RuntimeError as exc:
        msg = str(exc)
        if "DETCI requires at least two determinants" in msg:
            raise RuntimeError(
                f"Psi4 DETCI failed because He/{basis} gives a one-determinant CI space. "
                f"Use --basis 6-31g or larger. Original Psi4 message:\n{msg}"
            ) from exc
        raise


def get_psi4_mo_integrals(psi4, wfn) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mints = psi4.core.MintsHelper(wfn.basisset())

    S_ao = asarray_psi4_matrix(mints.ao_overlap())
    h_ao = asarray_psi4_matrix(mints.ao_kinetic()) + asarray_psi4_matrix(mints.ao_potential())

    C = asarray_psi4_matrix(wfn.Ca())
    h_mo = C.T @ h_ao @ C

    eri_obj = mints.mo_eri(wfn.Ca(), wfn.Ca(), wfn.Ca(), wfn.Ca())
    eri_mo = asarray_psi4_matrix(eri_obj)
    nmo = h_mo.shape[0]
    if eri_mo.size != nmo**4:
        raise RuntimeError(f"Unexpected MO ERI size {eri_mo.size}, expected {nmo**4}.")
    eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)

    eri_mo = 0.25 * (
        eri_mo
        + eri_mo.transpose(1, 0, 3, 2)
        + eri_mo.transpose(2, 3, 0, 1)
        + eri_mo.transpose(3, 2, 1, 0)
    )
    return S_ao, C, h_mo, eri_mo


def build_ab_pair_hamiltonian(h: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """
    Ordered pair basis |p alpha, q beta>.

    <p_a q_b | H | r_a s_b>
      = h[p,r] delta[q,s] + h[q,s] delta[p,r] + (p r | q s)
    """
    nmo = h.shape[0]
    H = np.zeros((nmo * nmo, nmo * nmo), dtype=float)

    for p in range(nmo):
        for q in range(nmo):
            I = p * nmo + q
            for r in range(nmo):
                for s in range(nmo):
                    J = r * nmo + s
                    val = 0.0
                    if q == s:
                        val += h[p, r]
                    if p == r:
                        val += h[q, s]
                    val += eri[p, r, q, s]
                    H[I, J] = val

    return 0.5 * (H + H.T)


def solve_two_electron_ab_fci(h: np.ndarray, eri: np.ndarray, enuc: float = 0.0):
    H = build_ab_pair_hamiltonian(h, eri)
    evals, evecs = np.linalg.eigh(H)
    idx = int(np.argmin(evals))
    e_elec = float(evals[idx])
    nmo = h.shape[0]
    Cab = evecs[:, idx].reshape(nmo, nmo)

    maxpos = np.unravel_index(np.argmax(np.abs(Cab)), Cab.shape)
    if Cab[maxpos] < 0:
        Cab = -Cab
    Cab /= np.linalg.norm(Cab)
    return e_elec + enuc, e_elec, Cab, evals


def rdms_from_ab_pair(Cab: np.ndarray):
    dm1a = Cab @ Cab.T
    dm1b = Cab.T @ Cab
    dm1 = dm1a + dm1b

    # dm2[p,q,r,s] = sum_st <a+_{p s} a+_{r t} a_{s t} a_{q s}>
    # alpha-beta: Cab[p,r] Cab[q,s]
    # beta-alpha: Cab[r,p] Cab[s,q]
    dm2 = np.einsum("pr,qs->pqrs", Cab, Cab, optimize=True)
    dm2 += np.einsum("rp,sq->pqrs", Cab, Cab, optimize=True)
    return dm1, dm2, dm1a, dm1b


def reconstruct_energy(h: np.ndarray, eri: np.ndarray, dm1: np.ndarray, dm2: np.ndarray, enuc: float):
    e1 = float(np.einsum("pq,pq", h, dm1, optimize=True))
    e2 = float(0.5 * np.einsum("pqrs,pqrs", eri, dm2, optimize=True))
    return e1 + e2 + enuc, e1, e2


def rdm_diagnostics(dm1: np.ndarray, dm2: np.ndarray, nelec: int = 2) -> Dict[str, float]:
    occ = np.linalg.eigvalsh(0.5 * (dm1 + dm1.T))[::-1]
    return {
        "trace_dm1": float(np.trace(dm1)),
        "trace_dm2": float(np.einsum("pprr->", dm2, optimize=True)),
        "target_trace_dm1": float(nelec),
        "target_trace_dm2": float(nelec * (nelec - 1)),
        "max_dm1_asym": float(np.max(np.abs(dm1 - dm1.T))),
        "max_dm2_bra_ket_error": float(np.max(np.abs(dm2 - dm2.transpose(2, 3, 0, 1)))),
        "occ": occ.tolist(),
    }


def try_extract_psi4_detci_rdms(wfn) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False, "notes": []}
    for name in ["get_opdm", "opdm", "get_tpdm", "tpdm"]:
        if hasattr(wfn, name):
            try:
                arr = np.array(np.asarray(getattr(wfn, name)()), dtype=float, copy=True)
                out[name] = arr
                out["available"] = True
                out["notes"].append(f"extracted {name}, shape={arr.shape}")
            except Exception as exc:
                out["notes"].append(f"{name} exists but failed: {repr(exc)}")
    if not out["available"]:
        out["notes"].append(
            "No stable Psi4 DETCI RDM method found on this build; "
            "saved RDMs are generated by the transparent two-electron FCI builder."
        )
    return out


def main():
    args = parse_args()
    label = safe_basis_label(args.basis)
    if args.out is None:
        args.out = f"he_{label}_detci_rdm_ref.npz"
    if args.summary is None:
        args.summary = f"he_{label}_detci_rdm_summary.txt"
    if args.psi4_output is None:
        args.psi4_output = f"psi4_he_{label}_detci.out"

    print("=" * 80)
    print("Step 1 | Psi4 DETCI -> RDM -> energy reconstruction | He singlet")
    print("=" * 80)
    print(f"basis       = {args.basis}")
    print(f"output npz  = {args.out}")

    e_ref, wfn, psi4, mol, ref_label = run_psi4_detci_he(
        basis=args.basis,
        memory=args.memory,
        nthreads=args.nthreads,
        psi4_output=args.psi4_output,
        allow_single=args.allow_single_determinant,
    )
    enuc = float(mol.nuclear_repulsion_energy())

    S_ao, C_mo, h_mo, eri_mo = get_psi4_mo_integrals(psi4, wfn)
    nmo = h_mo.shape[0]

    e_fci_total, e_fci_elec, Cab, evals = solve_two_electron_ab_fci(h_mo, eri_mo, enuc=enuc)
    dm1, dm2, dm1a, dm1b = rdms_from_ab_pair(Cab)
    e_rdm_total, e1, e2 = reconstruct_energy(h_mo, eri_mo, dm1, dm2, enuc)
    diag = rdm_diagnostics(dm1, dm2, nelec=2)
    psi4_rdm_attempt = try_extract_psi4_detci_rdms(wfn)

    diff_pair_ref = e_fci_total - e_ref
    diff_rdm_pair = e_rdm_total - e_fci_total
    diff_rdm_ref = e_rdm_total - e_ref

    print("\n[Energies]")
    print(f"Reference label                 = {ref_label}")
    print(f"Psi4 reference total energy     = {e_ref: .14f} Eh")
    print(f"Python pair-FCI total energy    = {e_fci_total: .14f} Eh")
    print(f"RDM reconstructed total energy  = {e_rdm_total: .14f} Eh")
    print(f"  E1                             = {e1: .14f} Eh")
    print(f"  E2                             = {e2: .14f} Eh")
    print(f"  Enuc                           = {enuc: .14f} Eh")
    print(f"Delta(pair-FCI - reference)     = {diff_pair_ref: .3e} Eh")
    print(f"Delta(RDM - pair-FCI)           = {diff_rdm_pair: .3e} Eh")
    print(f"Delta(RDM - reference)          = {diff_rdm_ref: .3e} Eh")

    print("\n[RDM diagnostics]")
    print(f"nmo                              = {nmo}")
    print(f"Tr(dm1)                          = {diag['trace_dm1']:.12f}  target=2")
    print(f"Tr(dm2)                          = {diag['trace_dm2']:.12f}  target=2")
    print(f"Max|dm1-dm1.T|                   = {diag['max_dm1_asym']:.3e}")
    print(f"Max dm2 bra-ket error            = {diag['max_dm2_bra_ket_error']:.3e}")
    occ4 = diag["occ"][:4] + [float("nan")] * max(0, 4-len(diag["occ"]))
    print("Natural occupations first 4       = " + ", ".join(f"{x:.10f}" for x in occ4[:4]))
    print(f"Pair matrix symmetry error        = {np.linalg.norm(Cab-Cab.T)/(np.linalg.norm(Cab)+1e-300):.3e}")

    print("\n[Psi4 RDM API attempt]")
    for note in psi4_rdm_attempt["notes"]:
        print("  -", note)

    metadata = {
        "system": "He",
        "charge": 0,
        "multiplicity": 1,
        "basis": args.basis,
        "nmo": int(nmo),
        "nelec": 2,
        "reference_label": ref_label,
        "method": "Psi4 DETCI ex_level=2 + independent two-electron alpha-beta FCI RDM",
        "rdm1_convention": "dm1[p,q] = sum_sigma <a^+_{p sigma} a_{q sigma}>",
        "rdm2_convention": "dm2[p,q,r,s] = sum_{sigma,tau}<a^+_{p sigma} a^+_{r tau} a_{s tau} a_{q sigma}>",
        "energy_formula": "E = einsum('pq,pq',h,dm1)+0.5*einsum('pqrs,pqrs',eri,dm2)+Enuc",
        "eri_convention": "eri[p,q,r,s] = (p q | r s), chemist notation",
        "E_reference": float(e_ref),
        "E_pair_fci": float(e_fci_total),
        "E_rdm": float(e_rdm_total),
        "diff_pair_fci_minus_reference": float(diff_pair_ref),
        "diff_rdm_minus_pair_fci": float(diff_rdm_pair),
        "diff_rdm_minus_reference": float(diff_rdm_ref),
    }
    metadata.update({k: v for k, v in diag.items() if k != "occ"})
    metadata["occ_first_10"] = diag["occ"][:10]

    np.savez(
        args.out,
        S_ao=S_ao,
        C_mo=C_mo,
        h_mo=h_mo,
        eri_mo=eri_mo,
        dm1=dm1,
        dm2=dm2,
        dm1_alpha=dm1a,
        dm1_beta=dm1b,
        pair_coeff_ab=Cab,
        E_reference=np.array(e_ref),
        E_pair_fci=np.array(e_fci_total),
        E_pair_fci_elec=np.array(e_fci_elec),
        E_rdm=np.array(e_rdm_total),
        E1=np.array(e1),
        E2=np.array(e2),
        Enuc=np.array(enuc),
        fci_eigenvalues=evals,
        metadata_json=np.array(json.dumps(metadata, indent=2)),
    )

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, indent=2))
        f.write("\n\nPsi4 RDM API attempt:\n")
        for note in psi4_rdm_attempt["notes"]:
            f.write(f"- {note}\n")

    print("\n[Saved]")
    print(" ", args.out)
    print(" ", args.summary)

    ok = (
        abs(diff_rdm_pair) < 1e-10
        and abs(diag["trace_dm1"] - 2.0) < 1e-10
        and abs(diag["trace_dm2"] - 2.0) < 1e-10
    )
    if nmo > 1 and abs(diff_pair_ref) > 1e-6:
        print("\nWARNING: pair-FCI energy differs from Psi4 reference by more than 1e-6 Eh.")
        print("Check frozen-core/active-space/orbital conventions if this happens.")
    if not ok:
        print("\nERROR: RDM self-consistency check failed.")
        sys.exit(2)

    print("\nStatus: RDM self-consistency check passed.")


if __name__ == "__main__":
    main()
