#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4b: Parent-basis OBS-FCI/RDM energy reconstruction check for He.

Purpose
-------
This script connects the parent-basis RI/F12 infrastructure from Step 4 with
a reference wavefunction and its RDMs.

It does the following in a single parent AO basis:

1. Run RHF for He in the parent basis.
2. Choose the first nobs RHF MOs as the orbital basis set (OBS).
3. Construct CABS+ as the S-orthogonal complement of OBS inside the same
   parent AO basis.
4. Build RI = [OBS, CABS].
5. Transform h, ERI, and F12-like tensors to OBS and RI bases.
6. Solve the two-electron alpha-beta FCI problem in the OBS.
7. Build spin-free 1-RDM and 2-RDM in the OBS.
8. Reconstruct the OBS-FCI energy from RDMs.
9. Embed the OBS RDMs into RI dimension and reconstruct the same energy from
   RI tensors as a final convention check.

This is not the final [2]R12 correction.  It prepares the validated input data:

    E_ref
    dm1_obs, dm2_obs
    h_obs, eri_obs
    h_ri, eri_ri
    f12_ri, f12sq_ri, f12g12_ri, f12dc_ri
    C_obs, C_cabs, C_ri

Default
-------
    He / parent cc-pVDZ / nobs=2 / corr=[(1.4, -1/1.4)]

The corr convention follows the Step 3c result:
    corr = [(Gaussian exponent, coefficient), ...]

Usage
-----
    python step4b_he_parent_obs_fci_rdm_check.py

Optional:
    python step4b_he_parent_obs_fci_rdm_check.py --parent-basis cc-pvtz --nobs 2
    python step4b_he_parent_obs_fci_rdm_check.py --corr "1.4,-0.7142857142857143"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Any, List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parent-basis", default="cc-pvdz", help="Single parent AO basis.")
    p.add_argument("--nobs", type=int, default=2, help="Number of lowest RHF MOs used as OBS.")
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument(
        "--corr",
        default=None,
        help="Manual Gaussian corr list: 'exponent,coefficient;exponent,coefficient;...'. "
             "If omitted, uses 'gamma,-1/gamma'.",
    )
    p.add_argument("--thresh", type=float, default=1e-10, help="CABS+ eigenvalue threshold.")
    p.add_argument("--memory", default="2 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--psi4-output", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--save-ao-integrals", action="store_true", help="Also save AO four-index tensors.")
    return p.parse_args()


def safe_label(s: str) -> str:
    return s.lower().replace("*", "s").replace("+", "p").replace("-", "").replace("_", "")


def parse_corr(s: str) -> List[Tuple[float, float]]:
    pairs = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        xs = [float(x.strip()) for x in part.split(",")]
        if len(xs) != 2:
            raise ValueError(f"Bad corr pair: {part!r}")
        pairs.append((xs[0], xs[1]))
    if not pairs:
        raise ValueError("Empty corr list.")
    return pairs


def asarray_psi4(x) -> np.ndarray:
    return np.array(np.asarray(x), dtype=float, copy=True)


def ensure_4d_tensor(x, nbf: int, label: str) -> np.ndarray:
    arr = asarray_psi4(x)
    if arr.shape == (nbf, nbf, nbf, nbf):
        return arr
    if arr.size == nbf**4:
        return arr.reshape(nbf, nbf, nbf, nbf)
    raise RuntimeError(f"{label}: unexpected shape {arr.shape}, size={arr.size}, nbf={nbf}")


def maxabs(A: np.ndarray) -> float:
    return float(np.max(np.abs(A))) if A.size else 0.0


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def build_psi4_he(parent_basis: str, memory: str, nthreads: int, output_file: str):
    try:
        import psi4
    except Exception as exc:
        raise RuntimeError("Cannot import psi4. Activate the Psi4 environment first.") from exc

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

    psi4.set_options(
        {
            "basis": parent_basis,
            "reference": "rhf",
            "scf_type": "pk",
            "e_convergence": 1e-12,
            "d_convergence": 1e-12,
        }
    )

    e_scf, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    mints = psi4.core.MintsHelper(wfn.basisset())

    return psi4, mol, wfn, mints, float(e_scf)


def compute_parent_integrals(wfn, mints, corr: List[Tuple[float, float]]) -> Dict[str, np.ndarray]:
    nao = int(wfn.basisset().nbf())

    S = asarray_psi4(mints.ao_overlap())
    T = asarray_psi4(mints.ao_kinetic())
    V = asarray_psi4(mints.ao_potential())
    h = T + V
    eri = ensure_4d_tensor(mints.ao_eri(), nao, "ao_eri")

    f12 = ensure_4d_tensor(mints.ao_f12(corr), nao, "ao_f12")
    f12sq = ensure_4d_tensor(mints.ao_f12_squared(corr), nao, "ao_f12_squared")
    f12g12 = ensure_4d_tensor(mints.ao_f12g12(corr), nao, "ao_f12g12")
    f12dc = ensure_4d_tensor(mints.ao_f12_double_commutator(corr), nao, "ao_f12_double_commutator")

    return {
        "S": S,
        "h": h,
        "eri": eri,
        "f12": f12,
        "f12sq": f12sq,
        "f12g12": f12g12,
        "f12dc": f12dc,
    }


def construct_parent_cabs_plus(C_obs: np.ndarray, S: np.ndarray, thresh: float) -> Dict[str, Any]:
    nao, nobs = C_obs.shape
    Iao = np.eye(nao)

    obs_metric = C_obs.T @ S @ C_obs
    obs_orth_error = maxabs(obs_metric - np.eye(nobs))

    # Project all parent AO functions against OBS.
    # OBS is S-orthonormal, so P = C_obs C_obs^T S.
    O = C_obs.T @ S @ Iao
    Q = Iao - C_obs @ O

    SQ = sym(Q.T @ S @ Q)
    evals, U = np.linalg.eigh(SQ)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    U = U[:, idx]
    keep = evals > thresh

    if np.count_nonzero(keep) == 0:
        C_cabs = np.zeros((nao, 0))
    else:
        C_cabs = Q @ U[:, keep] @ np.diag(1.0 / np.sqrt(evals[keep]))

    C_ri = np.hstack([C_obs, C_cabs])

    cabs_metric = C_cabs.T @ S @ C_cabs
    cross_metric = C_obs.T @ S @ C_cabs
    ri_metric = C_ri.T @ S @ C_ri

    return {
        "C_cabs": C_cabs,
        "C_ri": C_ri,
        "projected_cabs_overlap_evals": evals,
        "keep": keep,
        "obs_orth_error": float(obs_orth_error),
        "cabs_orth_error": maxabs(cabs_metric - np.eye(C_cabs.shape[1])) if C_cabs.shape[1] else 0.0,
        "obs_cabs_cross_error": maxabs(cross_metric),
        "ri_orth_error": maxabs(ri_metric - np.eye(C_ri.shape[1])),
        "nobs": int(nobs),
        "nao": int(nao),
        "ncabs": int(C_cabs.shape[1]),
        "nri": int(C_ri.shape[1]),
        "n_dropped": int(np.count_nonzero(~keep)),
        "min_kept_eval": float(np.min(evals[keep])) if np.any(keep) else None,
        "max_dropped_eval": float(np.max(evals[~keep])) if np.any(~keep) else None,
    }


def transform_1index(h_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    return C.T @ h_ao @ C


def transform_4index(I_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    # I[p,q,r,s] = C_ap C_bq I[a,b,c,d] C_cr C_ds
    return np.einsum("ap,bq,abcd,cr,ds->pqrs", C, C, I_ao, C, C, optimize=True)


def build_ab_pair_hamiltonian(h: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """
    Ordered pair basis |p alpha, q beta>.

    h uses spatial orbitals.
    eri[p,q,r,s] = (p q | r s), chemist notation.

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

    # Deterministic phase.
    maxpos = np.unravel_index(np.argmax(np.abs(Cab)), Cab.shape)
    if Cab[maxpos] < 0:
        Cab = -Cab
    Cab /= np.linalg.norm(Cab)

    return e_elec + enuc, e_elec, Cab, evals


def rdms_from_ab_pair(Cab: np.ndarray):
    """
    Spin-free RDM convention:
        dm1[p,q] = sum_sigma <a+_{p sigma} a_{q sigma}>
        dm2[p,q,r,s] = sum_{sigma,tau}
                       <a+_{p sigma} a+_{r tau} a_{s tau} a_{q sigma}>

    Energy:
        E = einsum('pq,pq', h, dm1)
            + 1/2 einsum('pqrs,pqrs', eri, dm2)
            + Enuc
    """
    dm1a = Cab @ Cab.T
    dm1b = Cab.T @ Cab
    dm1 = dm1a + dm1b

    dm2 = np.einsum("pr,qs->pqrs", Cab, Cab, optimize=True)
    dm2 += np.einsum("rp,sq->pqrs", Cab, Cab, optimize=True)

    return dm1, dm2, dm1a, dm1b


def reconstruct_energy(h: np.ndarray, eri: np.ndarray, dm1: np.ndarray, dm2: np.ndarray, enuc: float = 0.0):
    e1 = float(np.einsum("pq,pq", h, dm1, optimize=True))
    e2 = float(0.5 * np.einsum("pqrs,pqrs", eri, dm2, optimize=True))
    return e1 + e2 + enuc, e1, e2


def embed_rdm_to_ri(dm1_obs: np.ndarray, dm2_obs: np.ndarray, nri: int, nobs: int):
    dm1_ri = np.zeros((nri, nri), dtype=float)
    dm2_ri = np.zeros((nri, nri, nri, nri), dtype=float)
    dm1_ri[:nobs, :nobs] = dm1_obs
    dm2_ri[:nobs, :nobs, :nobs, :nobs] = dm2_obs
    return dm1_ri, dm2_ri


def rdm_diagnostics(dm1: np.ndarray, dm2: np.ndarray, nelec: int = 2) -> Dict[str, Any]:
    occ = np.linalg.eigvalsh(sym(dm1))[::-1]
    return {
        "trace_dm1": float(np.trace(dm1)),
        "trace_dm2": float(np.einsum("pprr->", dm2, optimize=True)),
        "target_trace_dm1": float(nelec),
        "target_trace_dm2": float(nelec * (nelec - 1)),
        "max_dm1_asym": maxabs(dm1 - dm1.T),
        "max_dm2_bra_ket_error": maxabs(dm2 - dm2.transpose(2, 3, 0, 1)),
        "natural_occupations": occ.tolist(),
    }


def tensor_diag(T: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(T.shape),
        "has_nan": bool(np.isnan(T).any()),
        "has_inf": bool(np.isinf(T).any()),
        "norm": float(np.linalg.norm(T.reshape(-1))) if T.size else 0.0,
        "max_abs": float(np.max(np.abs(T))) if T.size else 0.0,
        "bra_ket_error": maxabs(T - T.transpose(2, 3, 0, 1)),
    }


def main():
    args = parse_args()
    label = safe_label(args.parent_basis)

    if args.psi4_output is None:
        args.psi4_output = f"psi4_he_{label}_step4b.out"
    if args.out is None:
        args.out = f"he_{label}_step4b_obs_fci_rdm.npz"
    if args.summary is None:
        args.summary = f"he_{label}_step4b_obs_fci_rdm_summary.txt"

    corr = parse_corr(args.corr) if args.corr is not None else [(args.gamma, -1.0 / args.gamma)]

    print("=" * 80)
    print("Step 4b | Parent-basis OBS-FCI/RDM reconstruction | He")
    print("=" * 80)
    print(f"parent basis = {args.parent_basis}")
    print(f"nobs         = {args.nobs}")
    print(f"gamma        = {args.gamma}")
    print(f"corr         = {corr}")
    print(f"threshold    = {args.thresh:.3e}")

    psi4, mol, wfn, mints, e_scf = build_psi4_he(
        args.parent_basis, args.memory, args.nthreads, args.psi4_output
    )
    enuc = float(mol.nuclear_repulsion_energy())
    nao = int(wfn.basisset().nbf())
    if args.nobs < 1 or args.nobs >= nao:
        raise ValueError(f"--nobs must be between 1 and nao-1 for a non-empty CABS. Got nobs={args.nobs}, nao={nao}")

    ao = compute_parent_integrals(wfn, mints, corr)
    S = ao["S"]
    C_mo = asarray_psi4(wfn.Ca())
    C_obs = C_mo[:, : args.nobs]

    cabs_info = construct_parent_cabs_plus(C_obs, S, args.thresh)
    C_cabs = cabs_info["C_cabs"]
    C_ri = cabs_info["C_ri"]
    nobs = cabs_info["nobs"]
    ncabs = cabs_info["ncabs"]
    nri = cabs_info["nri"]

    # Transform ordinary and F12 tensors.
    h_obs = transform_1index(ao["h"], C_obs)
    eri_obs = transform_4index(ao["eri"], C_obs)

    h_ri = transform_1index(ao["h"], C_ri)
    eri_ri = transform_4index(ao["eri"], C_ri)
    f12_ri = transform_4index(ao["f12"], C_ri)
    f12sq_ri = transform_4index(ao["f12sq"], C_ri)
    f12g12_ri = transform_4index(ao["f12g12"], C_ri)
    f12dc_ri = transform_4index(ao["f12dc"], C_ri)

    # Full parent FCI in all RHF MOs, for reference only.
    h_mo_full = transform_1index(ao["h"], C_mo)
    eri_mo_full = transform_4index(ao["eri"], C_mo)
    E_full_fci, E_full_elec, Cab_full, evals_full = solve_two_electron_ab_fci(h_mo_full, eri_mo_full, enuc=enuc)

    # OBS-FCI and RDM.
    E_obs_fci, E_obs_elec, Cab_obs, evals_obs = solve_two_electron_ab_fci(h_obs, eri_obs, enuc=enuc)
    dm1_obs, dm2_obs, dm1a_obs, dm1b_obs = rdms_from_ab_pair(Cab_obs)
    E_obs_rdm, E1_obs, E2_obs = reconstruct_energy(h_obs, eri_obs, dm1_obs, dm2_obs, enuc=enuc)

    dm1_ri, dm2_ri = embed_rdm_to_ri(dm1_obs, dm2_obs, nri=nri, nobs=nobs)
    E_ri_rdm, E1_ri, E2_ri = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc=enuc)

    diag_obs = rdm_diagnostics(dm1_obs, dm2_obs, nelec=2)
    diag_ri = rdm_diagnostics(dm1_ri, dm2_ri, nelec=2)

    print("\n[Dimensions]")
    print(f"nao(parent) = {nao}")
    print(f"nobs        = {nobs}")
    print(f"ncabs       = {ncabs}")
    print(f"nri         = {nri}")

    print("\n[CABS+ orthogonality]")
    print(f"Max|C_obs^T S C_obs - I|   = {cabs_info['obs_orth_error']:.3e}")
    print(f"Max|C_cabs^T S C_cabs - I| = {cabs_info['cabs_orth_error']:.3e}")
    print(f"Max|C_obs^T S C_cabs|      = {cabs_info['obs_cabs_cross_error']:.3e}")
    print(f"Max|C_ri^T S C_ri - I|     = {cabs_info['ri_orth_error']:.3e}")

    print("\n[Energies]")
    print(f"RHF energy in parent basis       = {e_scf: .14f} Eh")
    print(f"Full parent FCI energy           = {E_full_fci: .14f} Eh")
    print(f"OBS-FCI energy                   = {E_obs_fci: .14f} Eh")
    print(f"OBS-RDM reconstructed energy     = {E_obs_rdm: .14f} Eh")
    print(f"RI-embedded RDM energy           = {E_ri_rdm: .14f} Eh")
    print(f"Delta(OBS-RDM - OBS-FCI)         = {E_obs_rdm - E_obs_fci: .3e} Eh")
    print(f"Delta(RI-RDM - OBS-FCI)          = {E_ri_rdm - E_obs_fci: .3e} Eh")
    print(f"OBS truncation vs full parent FCI = {E_obs_fci - E_full_fci: .6e} Eh")
    print(f"E1_obs                           = {E1_obs: .14f} Eh")
    print(f"E2_obs                           = {E2_obs: .14f} Eh")

    print("\n[RDM diagnostics: OBS]")
    print(f"Tr(dm1_obs)                      = {diag_obs['trace_dm1']:.12f}")
    print(f"Tr(dm2_obs)                      = {diag_obs['trace_dm2']:.12f}")
    print(f"Max|dm1-dm1.T|                   = {diag_obs['max_dm1_asym']:.3e}")
    print(f"Max dm2 bra-ket error            = {diag_obs['max_dm2_bra_ket_error']:.3e}")
    occ4 = diag_obs["natural_occupations"][:4] + [float("nan")] * 4
    print("Natural occupations first 4       = " + ", ".join(f"{x:.10f}" for x in occ4[:4]))

    print("\n[RI tensor diagnostics]")
    ri_tensor_diags = {
        "eri_ri": tensor_diag(eri_ri),
        "f12_ri": tensor_diag(f12_ri),
        "f12sq_ri": tensor_diag(f12sq_ri),
        "f12g12_ri": tensor_diag(f12g12_ri),
        "f12dc_ri": tensor_diag(f12dc_ri),
    }
    for name, d in ri_tensor_diags.items():
        print(f"{name:<10s} shape={d['shape']} norm={d['norm']:.6e} nan={d['has_nan']} bra-ket={d['bra_ket_error']:.3e}")

    metadata = {
        "system": "He",
        "parent_basis": args.parent_basis,
        "nao": int(nao),
        "nobs": int(nobs),
        "ncabs": int(ncabs),
        "nri": int(nri),
        "gamma": float(args.gamma),
        "corr_convention": "list[(Gaussian exponent, coefficient)]",
        "corr": [[float(a), float(c)] for a, c in corr],
        "threshold": float(args.thresh),
        "enuc": float(enuc),
        "E_scf_parent": float(e_scf),
        "E_full_parent_fci": float(E_full_fci),
        "E_obs_fci": float(E_obs_fci),
        "E_obs_rdm": float(E_obs_rdm),
        "E_ri_embedded_rdm": float(E_ri_rdm),
        "delta_obs_rdm_minus_obs_fci": float(E_obs_rdm - E_obs_fci),
        "delta_ri_rdm_minus_obs_fci": float(E_ri_rdm - E_obs_fci),
        "obs_truncation_error_vs_full_parent_fci": float(E_obs_fci - E_full_fci),
        "E1_obs": float(E1_obs),
        "E2_obs": float(E2_obs),
        "E1_ri": float(E1_ri),
        "E2_ri": float(E2_ri),
        "rdm_obs_diag": diag_obs,
        "rdm_ri_diag": diag_ri,
        "cabs_info": {
            k: v for k, v in cabs_info.items()
            if k not in ["C_cabs", "C_ri", "projected_cabs_overlap_evals", "keep"]
        },
        "ri_tensor_diags": ri_tensor_diags,
    }

    save_dict = {
        "S_ao": S,
        "h_ao": ao["h"],
        "C_mo": C_mo,
        "C_obs": C_obs,
        "C_cabs": C_cabs,
        "C_ri": C_ri,
        "h_obs": h_obs,
        "eri_obs": eri_obs,
        "h_ri": h_ri,
        "eri_ri": eri_ri,
        "f12_ri": f12_ri,
        "f12sq_ri": f12sq_ri,
        "f12g12_ri": f12g12_ri,
        "f12dc_ri": f12dc_ri,
        "dm1_obs": dm1_obs,
        "dm2_obs": dm2_obs,
        "dm1_alpha_obs": dm1a_obs,
        "dm1_beta_obs": dm1b_obs,
        "dm1_ri": dm1_ri,
        "dm2_ri": dm2_ri,
        "pair_coeff_ab_obs": Cab_obs,
        "pair_coeff_ab_full": Cab_full,
        "obs_fci_eigenvalues": evals_obs,
        "full_parent_fci_eigenvalues": evals_full,
        "projected_cabs_overlap_evals": cabs_info["projected_cabs_overlap_evals"],
        "keep": cabs_info["keep"],
        "E_obs_fci": np.array(E_obs_fci),
        "E_obs_rdm": np.array(E_obs_rdm),
        "E_ri_embedded_rdm": np.array(E_ri_rdm),
        "E_full_parent_fci": np.array(E_full_fci),
        "E_scf_parent": np.array(e_scf),
        "Enuc": np.array(enuc),
        "metadata_json": np.array(json.dumps(metadata, indent=2)),
    }

    if args.save_ao_integrals:
        save_dict.update({
            "eri_ao": ao["eri"],
            "f12_ao": ao["f12"],
            "f12sq_ao": ao["f12sq"],
            "f12g12_ao": ao["f12g12"],
            "f12dc_ao": ao["f12dc"],
        })

    np.savez(args.out, **save_dict)
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, indent=2))
        f.write("\n")

    print("\n[Saved]")
    print(" ", args.out)
    print(" ", args.summary)

    ok = (
        cabs_info["obs_orth_error"] < 1e-8
        and cabs_info["cabs_orth_error"] < 1e-8
        and cabs_info["obs_cabs_cross_error"] < 1e-8
        and cabs_info["ri_orth_error"] < 1e-8
        and abs(E_obs_rdm - E_obs_fci) < 1e-10
        and abs(E_ri_rdm - E_obs_fci) < 1e-10
        and abs(diag_obs["trace_dm1"] - 2.0) < 1e-10
        and abs(diag_obs["trace_dm2"] - 2.0) < 1e-10
        and all(not d["has_nan"] for d in ri_tensor_diags.values())
    )

    if not ok:
        print("\nERROR: Step 4b consistency check failed.")
        print("Inspect the summary file for RDM traces, energy reconstruction, orthogonality, or NaN tensors.")
        sys.exit(2)

    print("\nStatus: Step 4b parent-basis OBS-FCI/RDM check passed.")


if __name__ == "__main__":
    main()
