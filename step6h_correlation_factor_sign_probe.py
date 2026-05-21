#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6h: Correlation-factor sign convention probe.

The current fitted Slater convention is

    f12(r) = -1/gamma exp(-gamma r)

This script checks what changes when all Gaussian coefficients are multiplied by
-1.  It is designed to answer two separate questions:

1. Does Psi4's F12 integral API preserve the sign supplied in corr?
2. Does the 3C(FIX)/SP formula path in Step 6g need an f12-linear sign flip
   relative to the paper/Psi4 theory notation?
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from r12_common import maxabs, pair_matrix
from step6e_build_vxbc_intermediates import default_prefix
from step6g_audit_approxc_terms import (
    ab_space_indices,
    build_formula_matrices,
    build_tilde_terms,
    convention_variant_rows,
    make_unit_pair,
    orbital_energy_audit,
    pair_index,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="he_ccpvdz_nobs2_fitN7_step5a_r12_intermediates.npz")
    p.add_argument("--nocc", type=int, default=1)
    p.add_argument("--run-pipeline", action="store_true", help="Generate a true sign-flipped Psi4 tensor set.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--denom-thresh", type=float, default=1e-10)
    return p.parse_args()


def load_metadata(data) -> Dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    return json.loads(str(data["metadata_json"]))


def corr_from_metadata(meta: Dict[str, Any]) -> List[Tuple[float, float]]:
    if "step4b_metadata" in meta and "corr" in meta["step4b_metadata"]:
        return [(float(a), float(c)) for a, c in meta["step4b_metadata"]["corr"]]
    if "corr" in meta:
        return [(float(a), float(c)) for a, c in meta["corr"]]
    raise RuntimeError("Could not find corr in Step5a metadata.")


def corr_string(corr: List[Tuple[float, float]]) -> str:
    return ";".join(f"{a:.16g},{c:.16g}" for a, c in corr)


def flip_corr(corr: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(a, -c) for a, c in corr]


def run_command(cmd: List[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_sign_flip_pipeline(args, meta: Dict[str, Any], corr_pos: List[Tuple[float, float]]) -> str:
    prefix = default_prefix(args.inp) + "_signflip"
    step4b_out = f"{prefix}_step4b_obs_fci_rdm.npz"
    step4b_sum = f"{prefix}_step4b_obs_fci_rdm_summary.txt"
    step5a_out = f"{prefix}_step5a_r12_intermediates.npz"
    step5a_sum = f"{prefix}_step5a_r12_intermediates_summary.txt"
    parent_basis = meta.get("step4b_metadata", {}).get("parent_basis", "cc-pvdz")
    nobs = int(meta.get("nobs", 2))
    commands = [
        [
            args.python,
            "step4b_he_parent_obs_fci_rdm_check.py",
            "--parent-basis",
            parent_basis,
            "--nobs",
            str(nobs),
            "--corr",
            corr_string(corr_pos),
            "--out",
            step4b_out,
            "--summary",
            step4b_sum,
        ],
        [
            args.python,
            "step5a_he_r12_intermediate_check.py",
            "--inp",
            step4b_out,
            "--out",
            step5a_out,
            "--summary",
            step5a_sum,
        ],
    ]
    for cmd in commands:
        run_command(cmd)
    return step5a_out


def synthetic_sign_flipped(data) -> Dict[str, np.ndarray]:
    return {
        "f12_ri": -np.array(data["f12_ri"], dtype=float),
        "f12g12_ri": -np.array(data["f12g12_ri"], dtype=float),
        "f12sq_ri": np.array(data["f12sq_ri"], dtype=float),
        "f12dc_ri": np.array(data["f12dc_ri"], dtype=float),
    }


def compare_tensor_signs(negative_data, positive_data) -> Dict[str, Any]:
    out = {}
    for key in ["f12_ri", "f12g12_ri", "f12sq_ri", "f12dc_ri"]:
        neg = np.array(negative_data[key], dtype=float)
        pos = np.array(positive_data[key], dtype=float)
        out[key] = {
            "maxabs_pos_plus_neg": maxabs(pos + neg),
            "maxabs_pos_minus_neg": maxabs(pos - neg),
            "norm_neg": float(np.linalg.norm(neg)),
            "norm_pos": float(np.linalg.norm(pos)),
            "dot_pos_neg": float(np.vdot(pos.reshape(-1), neg.reshape(-1))),
        }
    return out


def audited_delta(data, nocc: int, thresh: float) -> Dict[str, Any]:
    meta = load_metadata(data)
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))
    nobs = int(meta.get("nobs", np.array(data["Cab_obs"]).shape[0]))
    F_ri = np.array(data["F_ri"], dtype=float)
    eps = np.array(orbital_energy_audit(F_ri, nobs, nocc)["eps_diag"], dtype=float)
    built = build_formula_matrices(data, nri, nobs, nocc)
    spaces = ab_space_indices(nri, nobs, nocc)
    kl_idx = np.array([pair_index(k, l, nri) for k in range(nocc) for l in range(nocc)], dtype=int)
    terms = build_tilde_terms(
        built["matrices"],
        eps,
        i=0,
        j=0,
        kl_indices=kl_idx,
        ab_indices=spaces["ri_external"],
        n=nri,
        thresh=thresh,
        B_source="B_fock_q3",
    )
    T = make_unit_pair(nocc, 0, 0, 0.5)
    linear = float(2.0 * (T @ terms["V_tilde"]))
    quadratic = float(T @ (terms["B_tilde"] @ T))
    variants = convention_variant_rows(
        built["matrices"],
        eps,
        0,
        0,
        kl_idx,
        spaces["ri_external"],
        nri,
        thresh,
    )
    return {
        "linear_2T_Vtilde": linear,
        "quadratic_T_Btilde_T": quadratic,
        "delta_E_baseline": linear + quadratic,
        "V_block": terms["V_block"].tolist(),
        "C_over_den_V": terms["C_over_den_V"].tolist(),
        "V_tilde": terms["V_tilde"].tolist(),
        "B_tilde_norm": float(np.linalg.norm(terms["B_tilde"])),
        "variant_rows": variants["rows"],
    }


def make_positive_view(base_data, synthetic: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out = {key: base_data[key] for key in base_data.files}
    out.update(synthetic)
    return out


def main():
    args = parse_args()
    prefix = default_prefix(args.inp)
    if args.out is None:
        args.out = f"{prefix}_step6h_corr_sign_probe.json"
    if args.summary is None:
        args.summary = f"{prefix}_step6h_corr_sign_probe_summary.txt"

    neg_data = np.load(args.inp, allow_pickle=True)
    meta = load_metadata(neg_data)
    corr_neg = corr_from_metadata(meta)
    corr_pos = flip_corr(corr_neg)

    true_pos_path = None
    if args.run_pipeline:
        true_pos_path = run_sign_flip_pipeline(args, meta, corr_pos)
        pos_data = np.load(true_pos_path, allow_pickle=True)
        sign_mode = "true_psi4_signflip_pipeline"
    else:
        pos_data = make_positive_view(neg_data, synthetic_sign_flipped(neg_data))
        sign_mode = "synthetic_linear_signflip"

    tensor_signs = compare_tensor_signs(neg_data, pos_data)
    neg_delta = audited_delta(neg_data, args.nocc, args.denom_thresh)
    pos_delta = audited_delta(pos_data, args.nocc, args.denom_thresh)

    diagnostics = {
        "input_negative_corr": args.inp,
        "positive_corr_step5a": true_pos_path,
        "mode": sign_mode,
        "corr_negative": corr_neg,
        "corr_positive": corr_pos,
        "corr_negative_string": corr_string(corr_neg),
        "corr_positive_string": corr_string(corr_pos),
        "tensor_sign_checks": tensor_signs,
        "negative_corr_audited_delta": neg_delta,
        "positive_corr_audited_delta": pos_delta,
        "decision": (
            "If true Psi4 sign-flipped tensors satisfy f12_pos ~= -f12_neg and "
            "f12g12_pos ~= -f12g12_neg while f12sq is unchanged, then the integral API "
            "preserves corr sign. If the positive-corr audited baseline is negative while "
            "the negative-corr baseline is positive, the Step-6f formula row should expose "
            "an f12_linear_sign convention switch rather than changing G bra/ket ordering."
        ),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    lines = []
    lines.append("=" * 100)
    lines.append("Step 6h | correlation-factor sign convention probe")
    lines.append("=" * 100)
    lines.append(f"mode       = {sign_mode}")
    lines.append(f"negative   = {args.inp}")
    lines.append(f"positive   = {true_pos_path}")
    lines.append("")
    lines.append("[Corr strings]")
    lines.append(f"negative: {diagnostics['corr_negative_string']}")
    lines.append(f"positive: {diagnostics['corr_positive_string']}")
    lines.append("")
    lines.append("[Tensor sign checks]")
    lines.append("| tensor | max|pos+neg| | max|pos-neg| | interpretation |")
    lines.append("|---|---:|---:|---|")
    for key, row in tensor_signs.items():
        if row["maxabs_pos_plus_neg"] < row["maxabs_pos_minus_neg"]:
            interp = "odd under corr sign"
        else:
            interp = "even under corr sign"
        lines.append(f"| {key} | {row['maxabs_pos_plus_neg']:.3e} | {row['maxabs_pos_minus_neg']:.3e} | {interp} |")
    lines.append("")
    lines.append("[Audited 3C(FIX)/SP baseline]")
    lines.append(
        f"negative corr: linear={neg_delta['linear_2T_Vtilde']:.8e}, "
        f"quadratic={neg_delta['quadratic_T_Btilde_T']:.8e}, "
        f"DeltaE={neg_delta['delta_E_baseline']:.8e}"
    )
    lines.append(
        f"positive corr: linear={pos_delta['linear_2T_Vtilde']:.8e}, "
        f"quadratic={pos_delta['quadratic_T_Btilde_T']:.8e}, "
        f"DeltaE={pos_delta['delta_E_baseline']:.8e}"
    )
    lines.append("")
    lines.append("[Decision]")
    lines.append(diagnostics["decision"])

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n")
        f.write(json.dumps(diagnostics, indent=2))
        f.write("\n")

    print("\n".join(lines))
    print("\n[Saved]")
    print(f"  {args.out}")
    print(f"  {args.summary}")


if __name__ == "__main__":
    main()
