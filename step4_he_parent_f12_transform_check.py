#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Parent-basis F12 integral transformation check for He.

Purpose
-------
This script switches from the earlier mixed-basis OBS/CABS test to a single
parent AO basis.  This avoids the unstable Psi4 mixed-basis calls such as

    mints.ao_f12(corr, obs, obs, cabs, cabs)

and follows the route needed for ECG-NO:

    parent AO basis
        -> choose OBS orbitals inside this parent basis
        -> construct CABS+ as the S-orthogonal complement
        -> compute F12 AO integrals in the parent AO basis
        -> transform them to the RI basis [OBS, CABS]

This is still NOT the full [2]R12 correction.  It verifies that the integral
and orbital-transformation infrastructure works.

Default test
------------
    He atom
    parent_basis = cc-pVDZ
    nobs = 2
    corr = [(1.4, -1/1.4)]

Psi4 convention found in Step 3c
--------------------------------
    corr = [(Gaussian exponent, coefficient), ...]

For example, the one-Gaussian surrogate to
    f12 = -1/gamma exp(-gamma r12)
is represented here only as a smoke-test approximation:
    corr = [(gamma, -1/gamma)]

A real fitted Slater expansion should later replace this one-term surrogate.

Usage
-----
    python step4_he_parent_f12_transform_check.py

Optional:
    python step4_he_parent_f12_transform_check.py --parent-basis cc-pvtz --nobs 2
    python step4_he_parent_f12_transform_check.py --corr "1.4,-0.7142857142857143"
    python step4_he_parent_f12_transform_check.py --save-ao-integrals
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parent-basis", default="cc-pvdz", help="Single parent AO basis.")
    p.add_argument("--nobs", type=int, default=2, help="Number of lowest RHF MOs used as OBS.")
    p.add_argument("--gamma", type=float, default=1.4, help="Used only for default corr.")
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
    p.add_argument("--save-ao-integrals", action="store_true", help="Also save large AO integral tensors.")
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


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def maxabs(A: np.ndarray) -> float:
    return float(np.max(np.abs(A))) if A.size else 0.0


def asarray_psi4(x):
    return np.array(np.asarray(x), dtype=float, copy=True)


def ensure_4d_tensor(x, nbf: int, label: str) -> np.ndarray:
    arr = asarray_psi4(x)
    if arr.shape == (nbf, nbf, nbf, nbf):
        return arr
    if arr.size == nbf**4:
        return arr.reshape(nbf, nbf, nbf, nbf)
    raise RuntimeError(f"{label}: unexpected shape {arr.shape}, size={arr.size}, nbf={nbf}")


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


def construct_parent_cabs_plus(C_obs: np.ndarray, S: np.ndarray, thresh: float) -> Dict[str, Any]:
    """
    CABS+ in a single parent AO basis.

    raw CABS = identity AO functions.
    Q = (1 - P_obs) raw_AO, with P_obs = C_obs C_obs^T S because C_obs^T S C_obs = I.
    Orthogonalize Q under S and discard near-zero directions.
    """
    nao, nobs = C_obs.shape
    Iao = np.eye(nao)

    obs_metric = C_obs.T @ S @ C_obs
    obs_orth_error = maxabs(obs_metric - np.eye(nobs))

    # Project all AO functions against OBS.
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


def transform_4index(I_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Transform a four-index AO tensor to an orthonormal orbital basis C.

    I[p,q,r,s] = C_mu,p C_nu,q C_lam,r C_sig,s I[mu,nu,lam,sig]
    """
    return np.einsum("mp,nq,mnls,lr,ssig->pqrs", C, C, I_ao, C, C, optimize=True)


def transform_4index_safe(I_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    # Same as transform_4index but with clearer index names.  Kept separate to
    # avoid accidental reuse of the string 's' as both an index and variable.
    return np.einsum("ap,bq,abcd,cr,ds->pqrs", C, C, I_ao, C, C, optimize=True)


def tensor_symmetry_diagnostics(T: np.ndarray) -> Dict[str, float]:
    return {
        "shape": list(T.shape),
        "has_nan": bool(np.isnan(T).any()),
        "has_inf": bool(np.isinf(T).any()),
        "norm": float(np.linalg.norm(T.reshape(-1))) if T.size else 0.0,
        "max_abs": float(np.max(np.abs(T))) if T.size else 0.0,
        "pair_bra_error": maxabs(T - T.transpose(1, 0, 2, 3)),
        "pair_ket_error": maxabs(T - T.transpose(0, 1, 3, 2)),
        "bra_ket_error": maxabs(T - T.transpose(2, 3, 0, 1)),
    }


def compute_parent_integrals(psi4, wfn, mints, corr: List[Tuple[float, float]]) -> Dict[str, np.ndarray]:
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


def main():
    args = parse_args()

    parent_label = safe_label(args.parent_basis)
    if args.psi4_output is None:
        args.psi4_output = f"psi4_he_{parent_label}_parent_f12_step4.out"
    if args.out is None:
        args.out = f"he_{parent_label}_parent_f12_ri_transform.npz"
    if args.summary is None:
        args.summary = f"he_{parent_label}_parent_f12_ri_transform_summary.txt"

    corr = parse_corr(args.corr) if args.corr is not None else [(args.gamma, -1.0 / args.gamma)]

    print("=" * 80)
    print("Step 4 | Parent-basis F12 -> RI transform check | He")
    print("=" * 80)
    print(f"parent basis = {args.parent_basis}")
    print(f"nobs         = {args.nobs}")
    print(f"gamma        = {args.gamma}")
    print(f"corr         = {corr}")
    print(f"threshold    = {args.thresh:.3e}")

    psi4, mol, wfn, mints, e_scf = build_psi4_he(
        args.parent_basis,
        args.memory,
        args.nthreads,
        args.psi4_output,
    )
    nao = int(wfn.basisset().nbf())
    if args.nobs < 1 or args.nobs >= nao:
        raise ValueError(f"--nobs must be between 1 and nao-1 for a non-empty CABS. Got nobs={args.nobs}, nao={nao}")

    integrals_ao = compute_parent_integrals(psi4, wfn, mints, corr)
    S = integrals_ao["S"]

    C_mo = asarray_psi4(wfn.Ca())
    C_obs = C_mo[:, : args.nobs]

    cabs_info = construct_parent_cabs_plus(C_obs, S, args.thresh)
    C_cabs = cabs_info["C_cabs"]
    C_ri = cabs_info["C_ri"]
    nobs = cabs_info["nobs"]
    ncabs = cabs_info["ncabs"]
    nri = cabs_info["nri"]

    print("\n[Dimensions]")
    print(f"nao(parent)  = {nao}")
    print(f"nobs         = {nobs}")
    print(f"ncabs        = {ncabs}")
    print(f"nri          = {nri}")
    print(f"dropped      = {cabs_info['n_dropped']}")

    print("\n[CABS+ orthogonality]")
    print(f"Max|C_obs^T S C_obs - I|   = {cabs_info['obs_orth_error']:.3e}")
    print(f"Max|C_cabs^T S C_cabs - I| = {cabs_info['cabs_orth_error']:.3e}")
    print(f"Max|C_obs^T S C_cabs|      = {cabs_info['obs_cabs_cross_error']:.3e}")
    print(f"Max|C_ri^T S C_ri - I|     = {cabs_info['ri_orth_error']:.3e}")

    evals = cabs_info["projected_cabs_overlap_evals"]
    print("\n[Projected complement eigenvalues]")
    print("largest 10 =", np.array2string(evals[:10], precision=6, suppress_small=False))
    print("smallest 10 =", np.array2string(evals[-10:], precision=6, suppress_small=False))

    print("\n[AO F12 integral diagnostics]")
    ao_diag = {}
    for key in ["eri", "f12", "f12sq", "f12g12", "f12dc"]:
        diag = tensor_symmetry_diagnostics(integrals_ao[key])
        ao_diag[key] = diag
        print(f"{key:<8s} shape={diag['shape']} norm={diag['norm']:.6e} max={diag['max_abs']:.6e} nan={diag['has_nan']}")

    print("\n[Transforming AO tensors to RI basis]")
    ri_tensors = {}
    ri_diag = {}
    for key in ["eri", "f12", "f12sq", "f12g12", "f12dc"]:
        T_ri = transform_4index_safe(integrals_ao[key], C_ri)
        ri_tensors[key + "_ri"] = T_ri
        diag = tensor_symmetry_diagnostics(T_ri)
        ri_diag[key] = diag
        print(
            f"{key + '_ri':<10s} shape={diag['shape']} norm={diag['norm']:.6e} "
            f"max={diag['max_abs']:.6e} nan={diag['has_nan']} "
            f"bra-ket={diag['bra_ket_error']:.3e}"
        )

    # OBS block comparison: RI tensor OBS block should be identical to direct transform with C_obs.
    print("\n[OBS-block consistency checks]")
    obs_block_errors = {}
    for key in ["eri", "f12", "f12sq", "f12g12", "f12dc"]:
        direct_obs = transform_4index_safe(integrals_ao[key], C_obs)
        obs_from_ri = ri_tensors[key + "_ri"][:nobs, :nobs, :nobs, :nobs]
        err = maxabs(direct_obs - obs_from_ri)
        obs_block_errors[key] = float(err)
        print(f"{key:<8s} Max|direct OBS - RI OBS block| = {err:.3e}")

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
        "scf_energy": float(e_scf),
        "threshold": float(args.thresh),
        "cabs_info": {k: v for k, v in cabs_info.items() if k not in ["C_cabs", "C_ri", "projected_cabs_overlap_evals", "keep"]},
        "ao_diag": ao_diag,
        "ri_diag": ri_diag,
        "obs_block_errors": obs_block_errors,
    }

    save_dict = {
        "S_ao": S,
        "h_ao": integrals_ao["h"],
        "C_mo": C_mo,
        "C_obs": C_obs,
        "C_cabs": C_cabs,
        "C_ri": C_ri,
        "projected_cabs_overlap_evals": evals,
        "keep": cabs_info["keep"],
        "eri_ri": ri_tensors["eri_ri"],
        "f12_ri": ri_tensors["f12_ri"],
        "f12sq_ri": ri_tensors["f12sq_ri"],
        "f12g12_ri": ri_tensors["f12g12_ri"],
        "f12dc_ri": ri_tensors["f12dc_ri"],
        "metadata_json": np.array(json.dumps(metadata, indent=2)),
    }
    if args.save_ao_integrals:
        save_dict.update({
            "eri_ao": integrals_ao["eri"],
            "f12_ao": integrals_ao["f12"],
            "f12sq_ao": integrals_ao["f12sq"],
            "f12g12_ao": integrals_ao["f12g12"],
            "f12dc_ao": integrals_ao["f12dc"],
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
        and all(not d["has_nan"] for d in ri_diag.values())
        and all(err < 1e-10 for err in obs_block_errors.values())
    )

    if not ok:
        print("\nERROR: Step 4 consistency check failed.")
        print("Inspect the summary file for orthogonality, NaN, or OBS-block mismatch.")
        sys.exit(2)

    print("\nStatus: Step 4 parent-basis F12 transform check passed.")


if __name__ == "__main__":
    main()
