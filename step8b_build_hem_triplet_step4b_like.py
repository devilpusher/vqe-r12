#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8b: Build a Step-4b-like parent/RI file for metastable He triplet."""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

from r12_common import (
    assert_finite,
    embed_rdm_to_ri,
    maxabs,
    rdm_diagnostics,
    reconstruct_energy,
    tensor_diagnostics,
)
from step4b_he_parent_obs_fci_rdm_check import (
    asarray_psi4,
    compute_parent_integrals,
    construct_parent_cabs_plus,
    ensure_4d_tensor,
    parse_corr,
    transform_1index,
    transform_4index,
)
from step6a_fit_slater_corr import fit_gaussian_expansion
from step7b_export_ecg_no_orbitals import (
    L_INFO,
    build_ao_index,
    build_ecg_no_cobs,
    build_pyscf_mol,
    even_tempered_exponents,
    load_no_mats,
    parse_int_list,
    psi4_basis_string,
)


HEM_GTO_PARAMS = {
    "s": (0.01485521, 2.31960559),
    "p": (0.01520197, 2.31236633),
    "d": (0.014, 2.32),
    "f": (0.014, 2.32),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-dir", default="local_external/he-meta")
    p.add_argument("--rdm-inp", default="step8a_hem_triplet_ecg_no_fci_rdm_export.npz")
    p.add_argument("--channels", default="s,p")
    p.add_argument("--s-pick", default="0,1")
    p.add_argument("--p-pick", default="0,1")
    p.add_argument("--d-pick", default="")
    p.add_argument("--f-pick", default="")
    p.add_argument("--basis-name", default="hemetecgnosp")
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--corr", default=None)
    p.add_argument("--thresh", type=float, default=1e-10)
    p.add_argument("--memory", default="4 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--psi4-output", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--energy-tol", type=float, default=1e-9)
    p.add_argument("--orth-tol", type=float, default=1e-8)
    p.add_argument("--save-compressed", action="store_true")
    p.add_argument("--r12-only", action="store_true")
    return p.parse_args()


def safe_tag(channels: List[str], picks: Dict[str, List[int]], fitn: int | None) -> str:
    parts = ["".join(channels)]
    for ch in channels:
        parts.append(f"{ch}{''.join(str(i) for i in picks[ch])}")
    if fitn is not None:
        parts.append(f"fitN{fitn}")
    return "_".join(parts)


def get_corr(args) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    if args.corr is not None:
        corr = parse_corr(args.corr)
        return corr, {"source": "manual --corr", "corr_string": args.corr, "corr": [[float(a), float(c)] for a, c in corr]}
    fit_args = SimpleNamespace(
        gamma=args.gamma,
        nterms=args.fitN,
        alpha_min=0.08,
        alpha_max=60.0,
        rmin=0.0,
        rmax=8.0,
        ngrid=2000,
        grid="quadratic",
        weight="short",
        ridge=0.0,
        nonpositive_coeff=False,
    )
    fit = fit_gaussian_expansion(fit_args)
    corr = [(float(a), float(c)) for a, c in fit["corr"]]
    return corr, {
        "source": "step6a local least-squares fitted Slater Gaussian expansion",
        "fitN": int(args.fitN),
        "gamma": float(args.gamma),
        "corr_string": fit["corr_string"],
        "corr": [[float(a), float(c)] for a, c in corr],
        "fit_metrics": fit["metrics"],
        "cusp_note": "Finite Gaussian expansions have f'(0)=0 and cannot exactly reproduce the Slater cusp derivative.",
    }


def build_psi4_parent(name: str, basis_text: str, memory: str, nthreads: int, output_file: str):
    import psi4

    psi4.core.clean()
    psi4.set_memory(memory)
    psi4.set_num_threads(nthreads)
    psi4.core.set_output_file(output_file, False)
    mol = psi4.geometry(
        """
        0 3
        He 0.0 0.0 0.0
        symmetry c1
        units bohr
        """
    )
    psi4.basis_helper(basis_text, name=name, set_option=True)
    psi4.set_options({"basis": name, "reference": "uhf", "scf_type": "pk"})
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    mints = psi4.core.MintsHelper(wfn.basisset())
    return mol, wfn, mints


def compute_parent_integrals_r12_optional(wfn, mints, corr, r12_only: bool) -> Dict[str, np.ndarray]:
    if not r12_only:
        return compute_parent_integrals(wfn, mints, corr)
    nao = int(wfn.basisset().nbf())
    S = asarray_psi4(mints.ao_overlap())
    T = asarray_psi4(mints.ao_kinetic())
    V = asarray_psi4(mints.ao_potential())
    return {
        "S": S,
        "h": T + V,
        "eri": ensure_4d_tensor(mints.ao_eri(), nao, "ao_eri"),
        "f12": ensure_4d_tensor(mints.ao_f12(corr), nao, "ao_f12"),
    }


def symmetrize_bra_ket(T: np.ndarray) -> np.ndarray:
    return 0.5 * (T + T.transpose(2, 3, 0, 1))


def check_finite_many(arrays: Dict[str, np.ndarray]) -> None:
    for name, arr in arrays.items():
        assert_finite(name, np.asarray(arr))


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
    for ch, vals in picks.items():
        if vals and ch not in channels:
            raise ValueError(f"{ch}_pick={vals} but channel {ch} is not included")

    tag = safe_tag(channels, picks, None if args.corr else args.fitN)
    args.psi4_output = args.psi4_output or f"psi4_step8b_hem_triplet_{tag}.out"
    args.out = args.out or f"step8b_hem_triplet_{tag}_step4b_like.npz"
    args.summary = args.summary or f"step8b_hem_triplet_{tag}_step4b_like_summary.txt"
    args.json = args.json or f"step8b_hem_triplet_{tag}_step4b_like.json"

    rdm_data = np.load(args.rdm_inp, allow_pickle=True)
    rdm_meta = json.loads(str(rdm_data["metrics_json"]))
    dm1_obs = np.array(rdm_data["dm1_obs"], dtype=float)
    dm2_obs = np.array(rdm_data["dm2_obs"], dtype=float)
    input_energy = float(rdm_meta["input_metrics"]["energy_Ha"])

    ns, mats = load_no_mats(args.no_dir)
    exponents = {ch: even_tempered_exponents(*HEM_GTO_PARAMS[ch], ns) for ch in L_INFO}
    mol_pyscf = build_pyscf_mol(ns, channels, exponents)
    S_pyscf = mol_pyscf.intor("int1e_ovlp")
    ao_index = build_ao_index(ns, channels)
    C_obs, labels = build_ecg_no_cobs(mats, ns, mol_pyscf, S_pyscf, ao_index, channels, picks)
    nobs = C_obs.shape[1]
    if dm1_obs.shape != (nobs, nobs) or dm2_obs.shape != (nobs, nobs, nobs, nobs):
        raise ValueError(f"RDM dimension {dm1_obs.shape}/{dm2_obs.shape} does not match selected nobs={nobs}")

    basis_text = psi4_basis_string(args.basis_name, channels, exponents)
    mol_psi4, wfn, mints = build_psi4_parent(args.basis_name, basis_text, args.memory, args.nthreads, args.psi4_output)
    enuc = float(mol_psi4.nuclear_repulsion_energy())
    nao = int(wfn.basisset().nbf())
    if nao != mol_pyscf.nao:
        raise RuntimeError(f"PySCF/Psi4 AO dimension mismatch: {mol_pyscf.nao} vs {nao}")

    S_psi4 = asarray_psi4(mints.ao_overlap())
    direct_overlap_error = maxabs(S_pyscf - S_psi4)
    cobs_orth_pyscf = maxabs(C_obs.T @ S_pyscf @ C_obs - np.eye(nobs))
    cobs_orth_psi4 = maxabs(C_obs.T @ S_psi4 @ C_obs - np.eye(nobs))
    if direct_overlap_error > 1e-8 or cobs_orth_psi4 > 1e-8:
        raise RuntimeError(
            f"PySCF/Psi4 AO order/overlap mismatch: S error={direct_overlap_error:.3e}, C orth={cobs_orth_psi4:.3e}"
        )

    corr, corr_info = get_corr(args)
    ao = compute_parent_integrals_r12_optional(wfn, mints, corr, args.r12_only)
    cabs_info = construct_parent_cabs_plus(C_obs, ao["S"], args.thresh)
    C_cabs = cabs_info["C_cabs"]
    C_ri = cabs_info["C_ri"]
    nri = int(cabs_info["nri"])
    ncabs = int(cabs_info["ncabs"])

    h_obs = transform_1index(ao["h"], C_obs)
    eri_obs = symmetrize_bra_ket(transform_4index(ao["eri"], C_obs))
    E_obs_rdm, E1_obs, E2_obs = reconstruct_energy(h_obs, eri_obs, dm1_obs, dm2_obs, enuc=enuc)

    h_ri = transform_1index(ao["h"], C_ri)
    eri_ri = symmetrize_bra_ket(transform_4index(ao["eri"], C_ri))
    f12_ri = symmetrize_bra_ket(transform_4index(ao["f12"], C_ri))
    f12sq_ri = None if args.r12_only else symmetrize_bra_ket(transform_4index(ao["f12sq"], C_ri))
    f12g12_ri = None if args.r12_only else symmetrize_bra_ket(transform_4index(ao["f12g12"], C_ri))
    f12dc_ri = None if args.r12_only else symmetrize_bra_ket(transform_4index(ao["f12dc"], C_ri))
    dm1_ri, dm2_ri = embed_rdm_to_ri(dm1_obs, dm2_obs, nri=nri, nobs=nobs)
    E_ri_rdm, E1_ri, E2_ri = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc=enuc)

    diag_obs = rdm_diagnostics(dm1_obs, dm2_obs)
    diag_ri = rdm_diagnostics(dm1_ri, dm2_ri)
    tensor_diags = {"eri_ri": tensor_diagnostics(eri_ri), "f12_ri": tensor_diagnostics(f12_ri)}
    if not args.r12_only:
        tensor_diags.update(
            {
                "f12sq_ri": tensor_diagnostics(f12sq_ri),
                "f12g12_ri": tensor_diagnostics(f12g12_ri),
                "f12dc_ri": tensor_diagnostics(f12dc_ri),
            }
        )
    check_finite_many(
        {
            "h_obs": h_obs,
            "eri_obs": eri_obs,
            "h_ri": h_ri,
            "eri_ri": eri_ri,
            "f12_ri": f12_ri,
            "dm1_obs": dm1_obs,
            "dm2_obs": dm2_obs,
            "dm1_ri": dm1_ri,
            "dm2_ri": dm2_ri,
        }
    )
    if not args.r12_only:
        check_finite_many({"f12sq_ri": f12sq_ri, "f12g12_ri": f12g12_ri, "f12dc_ri": f12dc_ri})

    checks = {
        "direct_overlap_error": float(direct_overlap_error),
        "cobs_orth_pyscf": float(cobs_orth_pyscf),
        "cobs_orth_psi4_same_order": float(cobs_orth_psi4),
        "obs_orth_error": float(cabs_info["obs_orth_error"]),
        "cabs_orth_error": float(cabs_info["cabs_orth_error"]),
        "obs_cabs_cross_error": float(cabs_info["obs_cabs_cross_error"]),
        "ri_orth_error": float(cabs_info["ri_orth_error"]),
        "delta_obs_rdm_minus_input_energy": float(E_obs_rdm - input_energy),
        "delta_ri_rdm_minus_input_energy": float(E_ri_rdm - input_energy),
        "trace_dm1_error": float(diag_obs["trace_dm1"] - 2.0),
        "trace_dm2_error": float(diag_obs["trace_dm2"] - 2.0),
        "dm2_bra_ket_error": float(diag_obs["max_dm2_bra_ket_error"]),
    }
    failures = []
    for key in ["obs_orth_error", "cabs_orth_error", "obs_cabs_cross_error", "ri_orth_error"]:
        if checks[key] > args.orth_tol:
            failures.append(key)
    if abs(checks["delta_obs_rdm_minus_input_energy"]) > args.energy_tol:
        failures.append("obs_energy_reconstruction")
    if abs(checks["delta_ri_rdm_minus_input_energy"]) > args.energy_tol:
        failures.append("ri_energy_reconstruction")
    if abs(checks["trace_dm1_error"]) > 1e-10:
        failures.append("dm1_trace")
    if abs(checks["trace_dm2_error"]) > 1e-10:
        failures.append("dm2_trace")
    if checks["dm2_bra_ket_error"] > 1e-10:
        failures.append("dm2_bra_ket")
    for name, d in tensor_diags.items():
        if d["has_nan"] or d["has_inf"]:
            failures.append(f"{name}_finite")

    metadata = {
        "step": "8b",
        "system": "He metastable triplet",
        "state": "1s2s 3S, Ms=1",
        "no_dir": str(args.no_dir),
        "rdm_inp": str(args.rdm_inp),
        "channels": channels,
        "picks": picks,
        "basis_name": args.basis_name,
        "basis_text": basis_text,
        "r12_only": bool(args.r12_only),
        "ns": int(ns),
        "nao": int(nao),
        "nobs": int(nobs),
        "ncabs": int(ncabs),
        "nri": int(nri),
        "labels": labels,
        "enuc": float(enuc),
        "corr_info": corr_info,
        "rdm_source": "Step8a alpha-alpha triplet ECG-NO-FCI RDM",
        "energies": {
            "E_obs_fci": float(input_energy),
            "E_obs_rdm": float(E_obs_rdm),
            "E_ri_embedded_rdm": float(E_ri_rdm),
            "E1_obs": float(E1_obs),
            "E2_obs": float(E2_obs),
            "E1_ri": float(E1_ri),
            "E2_ri": float(E2_ri),
        },
        "checks": checks,
        "passed": not failures,
        "failures": failures,
        "rdm_obs_diag": diag_obs,
        "rdm_ri_diag": diag_ri,
        "cabs_info": {
            k: v for k, v in cabs_info.items()
            if k not in ["C_cabs", "C_ri", "projected_cabs_overlap_evals", "keep"]
        },
        "ri_tensor_diags": tensor_diags,
    }

    save_dict = {
        "S_ao": ao["S"],
        "h_ao": ao["h"],
        "C_obs": C_obs,
        "C_obs_psi4_same_order": C_obs,
        "C_cabs": C_cabs,
        "C_ri": C_ri,
        "h_obs": h_obs,
        "eri_obs": eri_obs,
        "h_ri": h_ri,
        "eri_ri": eri_ri,
        "f12_ri": f12_ri,
        "dm1_obs": dm1_obs,
        "dm2_obs": dm2_obs,
        "dm1_ri": dm1_ri,
        "dm2_ri": dm2_ri,
        "projected_cabs_overlap_evals": cabs_info["projected_cabs_overlap_evals"],
        "keep": cabs_info["keep"],
        "labels": np.array(labels, dtype=object),
        "channels": np.array(channels, dtype=object),
        "basis_text": np.array(basis_text),
        "E_obs_fci": np.array(input_energy),
        "E_obs_rdm": np.array(E_obs_rdm),
        "E_ri_embedded_rdm": np.array(E_ri_rdm),
        "Enuc": np.array(enuc),
        "metadata_json": np.array(json.dumps(metadata, indent=2)),
    }
    if not args.r12_only:
        save_dict.update({"f12sq_ri": f12sq_ri, "f12g12_ri": f12g12_ri, "f12dc_ri": f12dc_ri})
    save_func = np.savez_compressed if args.save_compressed else np.savez
    save_func(args.out, **save_dict)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    lines = [
        "=" * 96,
        "Step 8b | HEM triplet selected-OBS Step-4b-like bridge",
        "=" * 96,
        f"no_dir       = {args.no_dir}",
        f"rdm_inp      = {args.rdm_inp}",
        f"channels     = {channels}",
        f"picks        = {picks}",
        f"ns / nao     = {ns} / {nao}",
        f"nobs/ncabs/nri = {nobs}/{ncabs}/{nri}  ({2 * nobs} qubits)",
        f"corr source  = {corr_info['source']}",
        "",
        "[AO / C_obs audit]",
        f"Max|S_pyscf-S_psi4|              = {direct_overlap_error:.3e}",
        f"Max|C_obs^T S_pyscf C_obs-I|    = {cobs_orth_pyscf:.3e}",
        f"Max|C_obs^T S_psi4 C_obs-I|     = {cobs_orth_psi4:.3e}",
        f"labels                          = {labels}",
        "",
        "[Energy reconstruction]",
        f"E_input_HEM_triplet_FCI          = {input_energy: .14f} Eh",
        f"E_OBS_RDM                        = {E_obs_rdm: .14f} Eh",
        f"E_RI_embedded_RDM                = {E_ri_rdm: .14f} Eh",
        f"Delta(OBS_RDM-input)             = {E_obs_rdm - input_energy: .3e} Eh",
        f"Delta(RI_RDM-input)              = {E_ri_rdm - input_energy: .3e} Eh",
        f"E1_obs / E2_obs                  = {E1_obs: .14f} / {E2_obs: .14f} Eh",
        "",
        "[RDM]",
        f"Tr(dm1_obs)                      = {diag_obs['trace_dm1']:.12f}",
        f"Tr(dm2_obs)                      = {diag_obs['trace_dm2']:.12f}",
        f"dm2 bra-ket error                = {diag_obs['max_dm2_bra_ket_error']:.3e}",
        "Natural occupations              = " + np.array2string(np.array(diag_obs["natural_occupations"]), precision=10),
        "",
        "[RI tensor finite/symmetry]",
    ]
    for name, d in tensor_diags.items():
        lines.append(f"{name:<10s} shape={d['shape']} nan={d['has_nan']} inf={d['has_inf']} bra-ket={d['bra_ket_error']:.3e}")
    lines.extend(["", f"passed                           = {metadata['passed']}"])
    if failures:
        lines.append(f"failures                         = {failures}")
    lines.extend(["", "[Saved]", f"  {args.out}", f"  {args.json}", f"  {args.summary}"])
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
