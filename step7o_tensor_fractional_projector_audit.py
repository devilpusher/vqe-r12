#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7o: Tensor-level fractional NO projector audit.

Step7n tested fractional projectors by interpolating final energies between two
hard projector limits.  This script moves the fractional projector inside the
V/B/X/Delta tensor contractions.

We keep the full ECG-NO OBS pair/RDM space active, but replace every passive
index block by a weighted RI sum:

    q_i = 0      for fixed core OBS orbitals
    q_i = model(n_i) for non-core ECG-NO OBS orbitals
    q_i = 1      for CABS orbitals

Setting q=0 for all OBS and q=1 for CABS reproduces the conservative CABS-only
full-active formula.  Nonzero q on low-occupation ECG-NO orbitals tests whether
those orbitals can enter the complementary projector fractionally at tensor
level, without an external refscale.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from r12_common import build_sp_tensor
from r12_correction import block2, block4
from step7g_audit_ecg_no_r12_subterms import load_case
from step7h_dual_space_projector_prototype import fixed_core_indices, load_labels, parse_core_tokens
from step7i_residual_weighted_dual_space import DEFAULT_REFERENCE_ECG14
from step7j_scan_residual_weights import parse_fitn
from step7m_strict_projector_partition_audit import DEFAULT_HE_EXACT_NONREL, case_family


MODELS = [
    "cabs_only",
    "q_occ_over_occ_plus_alpha",
    "q_one_minus_exp_occ_over_alpha",
    "q_sqrt_occ_over_occ_plus_alpha",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=None)
    p.add_argument("--glob", default="step7c_*fitN*_r12only_step4b_like.npz")
    p.add_argument("--core", default="s0,s1,s2,p0")
    p.add_argument("--alphas", default="1e-4,2.5e-4,5e-4,1e-3")
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--exact-energy", type=float, default=DEFAULT_HE_EXACT_NONREL)
    p.add_argument("--out-json", default="step7o_tensor_fractional_projector_audit.json")
    p.add_argument("--out-csv", default="step7o_tensor_fractional_projector_audit.csv")
    p.add_argument("--stats-csv", default="step7o_tensor_fractional_projector_audit_stats.csv")
    p.add_argument("--summary", default="step7o_tensor_fractional_projector_audit_summary.txt")
    return p.parse_args()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def sorted_inputs(args) -> List[str]:
    paths = args.inputs if args.inputs else [str(p) for p in Path(".").glob(args.glob)]
    return sorted(paths, key=lambda p: (case_family(Path(p).name), parse_fitn(p), Path(p).name))


def wblock2(A: np.ndarray, i: List[int], j: List[int], wi: Optional[np.ndarray] = None, wj: Optional[np.ndarray] = None) -> np.ndarray:
    out = block2(A, i, j)
    if wi is not None:
        out = out * wi[:, None]
    if wj is not None:
        out = out * wj[None, :]
    return out


def wblock4(
    T: np.ndarray,
    i: List[int],
    j: List[int],
    k: List[int],
    l: List[int],
    wi: Optional[np.ndarray] = None,
    wj: Optional[np.ndarray] = None,
    wk: Optional[np.ndarray] = None,
    wl: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = block4(T, i, j, k, l)
    if wi is not None:
        out = out * wi[:, None, None, None]
    if wj is not None:
        out = out * wj[None, :, None, None]
    if wk is not None:
        out = out * wk[None, None, :, None]
    if wl is not None:
        out = out * wl[None, None, None, :]
    return out


def q_for_model(occ: np.ndarray, nobs: int, nri: int, core: List[int], model: str, alpha: float) -> np.ndarray:
    q = np.ones(nri, dtype=float)
    q[:nobs] = 0.0
    noncore = [i for i in range(nobs) if i not in set(core)]
    if model == "cabs_only":
        return q
    x = occ[noncore]
    if model == "q_occ_over_occ_plus_alpha":
        vals = x / (x + alpha)
    elif model == "q_one_minus_exp_occ_over_alpha":
        vals = 1.0 - np.exp(-x / alpha)
    elif model == "q_sqrt_occ_over_occ_plus_alpha":
        vals = np.sqrt(x / (x + alpha))
    else:
        raise ValueError(f"Unknown model {model}")
    q[noncore] = np.clip(vals, 0.0, 1.0)
    q[core] = 0.0
    return q


def tensor_fractional_components(g_phys, r_phys, fock, rdm1, rdm2, nobs: int, nri: int, q: np.ndarray) -> Dict[str, float]:
    a = list(range(nobs))
    f = list(range(nri))
    p = f
    qp = q[p]
    t = build_sp_tensor(nobs)

    gKLxy_rRSkl = np.einsum("klxy,rskl->rsxy", block4(g_phys, f, f, a, a), block4(r_phys, a, a, f, f), optimize=True)
    gTUxy_rRStu = np.einsum("tuxy,rstu->rsxy", block4(g_phys, a, a, a, a), block4(r_phys, a, a, a, a), optimize=True)
    gATxy_rdm1Ut_rRSau = np.einsum(
        "atxy,ut,rsau->rsxy",
        wblock4(g_phys, p, a, a, a, wi=qp),
        rdm1,
        wblock4(r_phys, a, a, p, a, wk=qp),
        optimize=True,
    )
    V = float(np.einsum("pqrs,xypq,rsxy", t, rdm2, gKLxy_rRSkl - gTUxy_rRStu - gATxy_rdm1Ut_rRSau, optimize=True))

    rZYpq_fockXy_rTUzx = np.einsum("zypq,xy,tuzx->tupq", block4(r_phys, a, a, a, a), block2(fock, a, a), block4(r_phys, a, a, a, a), optimize=True)
    rAYpq_fockXa_rTUxy = np.einsum(
        "aypq,xa,tuxy->tupq",
        wblock4(r_phys, p, a, a, a, wi=qp),
        wblock2(fock, a, p, wj=qp),
        block4(r_phys, a, a, a, a),
        optimize=True,
    )
    rYXpq_fockAx_rTUya = np.einsum(
        "yxpq,ax,tuya->tupq",
        block4(r_phys, a, a, a, a),
        wblock2(fock, p, a, wi=qp),
        wblock4(r_phys, a, a, a, p, wl=qp),
        optimize=True,
    )
    rMLpq_fockKl_rTUmk = np.einsum("mlpq,kl,tumk->tupq", block4(r_phys, f, f, a, a), block2(fock, f, f), block4(r_phys, a, a, f, f), optimize=True)
    rBYpq_rdm1Xy_fockAb_rTUax = np.einsum(
        "bypq,xy,ab,tuax->tupq",
        wblock4(r_phys, p, a, a, a, wi=qp),
        rdm1,
        wblock2(fock, p, p, wi=qp, wj=qp),
        wblock4(r_phys, a, a, p, a, wk=qp),
        optimize=True,
    )
    rAYpq_rdm1Xy_fockKx_rTUak = np.einsum(
        "aypq,xy,kx,tuak->tupq",
        wblock4(r_phys, p, a, a, a, wi=qp),
        rdm1,
        block2(fock, f, a),
        wblock4(r_phys, a, a, p, f, wk=qp),
        optimize=True,
    )
    B_mid = rMLpq_fockKl_rTUmk - rZYpq_fockXy_rTUzx - rAYpq_fockXa_rTUxy - rYXpq_fockAx_rTUya - 0.5 * rBYpq_rdm1Xy_fockAb_rTUax - 0.5 * rAYpq_rdm1Xy_fockKx_rTUak
    B = float(np.einsum("pqrs,vwtu,rsvw,tupq", t, t, rdm2, B_mid, optimize=True))

    rTUkl_rKLpq = np.einsum("tukl,klpq->tupq", block4(r_phys, a, a, f, f), block4(r_phys, f, f, a, a), optimize=True)
    rTUyz_rYZpq = np.einsum("tuyz,yzpq->tupq", block4(r_phys, a, a, a, a), block4(r_phys, a, a, a, a), optimize=True)
    rUTya_rdm1Yz_rAZpq = np.einsum(
        "utya,yz,azpq->tupq",
        wblock4(r_phys, a, a, a, p, wl=qp),
        rdm1,
        wblock4(r_phys, p, a, a, a, wi=qp),
        optimize=True,
    )
    rTUay_rdm1Yz_rAZqp = np.einsum(
        "tuay,yz,azqp->tupq",
        wblock4(r_phys, a, a, p, a, wk=qp),
        rdm1,
        wblock4(r_phys, p, a, a, a, wi=qp),
        optimize=True,
    )
    X_mid = rTUkl_rKLpq - rTUyz_rYZpq - 0.5 * rUTya_rdm1Yz_rAZpq - 0.5 * rTUay_rdm1Yz_rAZqp
    X = float(-np.einsum("pqrs,vwtu,rsvx,xw,tupq", t, t, rdm2, block2(fock, a, a), X_mid, optimize=True))

    r_paypq = wblock4(r_phys, p, a, a, a, wi=qp)
    r_payqp = wblock4(r_phys, p, a, a, a, wi=qp)
    r_utak = wblock4(r_phys, a, a, p, f, wk=qp)
    r_tuak = wblock4(r_phys, a, a, p, f, wk=qp)
    Delta1 = (
        -0.5 * np.einsum("pqrs,aypq,vwtu,xrvy,kx,sw,utak", t, r_paypq, t, rdm2, block2(fock, f, a), rdm1, r_utak, optimize=True)
        -0.5 * np.einsum("pqrs,aypq,vwtu,xryv,kx,sw,tuak", t, r_paypq, t, rdm2, block2(fock, f, a), rdm1, r_tuak, optimize=True)
        -0.5 * np.einsum("pqrs,aypq,vwtu,kx,rv,sw,xy,utak", t, r_paypq, t, block2(fock, f, a), rdm1, rdm1, rdm1, r_utak, optimize=True)
        + np.einsum("pqrs,aypq,vwtu,kx,rv,sw,xy,tuak", t, r_paypq, t, block2(fock, f, a), rdm1, rdm1, rdm1, r_tuak, optimize=True)
        +0.5 * np.einsum("pqrs,aypq,vwtu,kx,ry,sv,xw,tuak", t, r_paypq, t, block2(fock, f, a), rdm1, rdm1, rdm1, r_tuak, optimize=True)
        -0.25 * np.einsum("pqrs,aypq,vwtu,kx,ry,sv,xw,utak", t, r_paypq, t, block2(fock, f, a), rdm1, rdm1, rdm1, r_utak, optimize=True)
    )
    Delta2 = (
        np.einsum("pqrs,ayqp,vwtu,xrvy,kx,sw,utak", t, r_payqp, t, rdm2, block2(fock, f, a), rdm1, r_utak, optimize=True)
        -0.5 * np.einsum("pqrs,ayqp,vwtu,xrvy,kx,sw,tuak", t, r_payqp, t, rdm2, block2(fock, f, a), rdm1, r_tuak, optimize=True)
        - np.einsum("pqrs,ayqp,vwtu,kx,ry,sv,xw,tuak", t, r_payqp, t, block2(fock, f, a), rdm1, rdm1, rdm1, r_tuak, optimize=True)
        +0.5 * np.einsum("pqrs,ayqp,vwtu,kx,ry,sv,xw,utak", t, r_payqp, t, block2(fock, f, a), rdm1, rdm1, rdm1, r_utak, optimize=True)
    )
    Delta = float(Delta1 + Delta2)
    return {"V": V, "B": B, "X": X, "Delta": Delta, "correction": V + B + X + Delta}


def audit_case(path: str, args) -> List[Dict[str, Any]]:
    case = load_case(path)
    nobs = case["nobs"]
    nri = case["nri"]
    labels = load_labels(path, nobs)
    core = fixed_core_indices(labels, parse_core_tokens(args.core))
    if not core:
        core = list(range(min(6, nobs)))
    occ = np.array(case["natural_occupations"], dtype=float)
    rows = []
    for model in MODELS:
        alphas = [0.0] if model == "cabs_only" else parse_float_list(args.alphas)
        for alpha in alphas:
            q = q_for_model(occ, nobs, nri, core, model, alpha)
            comp = tensor_fractional_components(case["g_phys"], case["r_phys"], case["fock"], case["dm1_obs"], case["dm2_obs"], nobs, nri, q)
            delta = comp["correction"]
            E_total = case["E_obs"] + delta
            rows.append(
                {
                    "family": case_family(case["label"]),
                    "case": case["label"],
                    "path": path,
                    "fitN": parse_fitn(path),
                    "nobs": nobs,
                    "nqubits": 2 * nobs,
                    "model": model,
                    "alpha": alpha,
                    "q_obs_sum": float(np.sum(q[:nobs])),
                    "q_obs_max": float(np.max(q[:nobs])) if nobs else 0.0,
                    "q_cabs_min": float(np.min(q[nobs:nri])) if nri > nobs else 0.0,
                    "E_obs": case["E_obs"],
                    "delta_E_r12": delta,
                    "delta_E_r12_mEh": 1000.0 * delta,
                    "E_total": E_total,
                    "residual_to_reference_mEh": 1000.0 * (E_total - args.reference_energy),
                    "abs_residual_to_reference_mEh": 1000.0 * abs(E_total - args.reference_energy),
                    "above_exact_mEh": 1000.0 * (E_total - args.exact_energy),
                    "overcorrects_reference": E_total < args.reference_energy - 1.0e-10,
                    "below_exact_nonrel": E_total < args.exact_energy,
                    "V_mEh": 1000.0 * comp["V"],
                    "B_mEh": 1000.0 * comp["B"],
                    "X_mEh": 1000.0 * comp["X"],
                    "Delta_mEh": 1000.0 * comp["Delta"],
                }
            )
    return rows


def build_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["model"], row["alpha"])].append(row)
    stats = []
    for (family, model, alpha), group in sorted(grouped.items()):
        deltas = np.array([r["delta_E_r12_mEh"] for r in group], dtype=float)
        residuals = np.array([r["abs_residual_to_reference_mEh"] for r in group], dtype=float)
        stats.append(
            {
                "family": family,
                "model": model,
                "alpha": alpha,
                "npoints": len(group),
                "fitNs": ",".join(str(r["fitN"]) for r in sorted(group, key=lambda x: x["fitN"])),
                "delta_mean_mEh": float(np.mean(deltas)),
                "delta_std_mEh": float(np.std(deltas)),
                "delta_min_mEh": float(np.min(deltas)),
                "delta_max_mEh": float(np.max(deltas)),
                "delta_span_mEh": float(np.max(deltas) - np.min(deltas)),
                "abs_residual_mean_mEh": float(np.mean(residuals)),
                "abs_residual_max_mEh": float(np.max(residuals)),
                "q_obs_sum_mean": float(np.mean([r["q_obs_sum"] for r in group])),
                "q_obs_max_mean": float(np.mean([r["q_obs_max"] for r in group])),
                "any_overcorrects_reference": any(r["overcorrects_reference"] for r in group),
                "any_below_exact_nonrel": any(r["below_exact_nonrel"] for r in group),
            }
        )
    return stats


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args, inputs: List[str], rows: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> None:
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": inputs,
                "reference_energy": args.reference_energy,
                "exact_energy": args.exact_energy,
                "core": args.core,
                "alphas": parse_float_list(args.alphas),
                "models": MODELS,
                "rows": rows,
                "stats": stats,
                "notes": [
                    "This is a tensor-level fractional passive projector audit.",
                    "OBS q weights enter the V/B/X/Delta contractions wherever a passive index appears.",
                    "A final formula still needs a paper-level justification for q(n).",
                ],
            },
            f,
            indent=2,
        )
    write_csv(args.out_csv, rows)
    write_csv(args.stats_csv, stats)

    lines = []
    lines.append("=" * 138)
    lines.append("Step 7o | Tensor-level fractional NO passive projector audit")
    lines.append("=" * 138)
    lines.append(f"core = {args.core}")
    lines.append("")
    lines.append("[FitN stability by family/model/alpha]")
    lines.append("family                   model                              alpha      dE_mean  dE_span  resid_max overRef belowExact")
    for st in stats:
        lines.append(
            f"{st['family']:<24s} {st['model']:<34s} {st['alpha']:>8.1e} "
            f"{st['delta_mean_mEh']:>8.3f} {st['delta_span_mEh']:>8.3f} "
            f"{st['abs_residual_max_mEh']:>9.3f} "
            f"{str(st['any_overcorrects_reference']):>7s} {str(st['any_below_exact_nonrel']):>10s}"
        )
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out_json}")
    lines.append(f"  {args.out_csv}")
    lines.append(f"  {args.stats_csv}")
    lines.append(f"  {args.summary}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    inputs = sorted_inputs(args)
    if not inputs:
        raise SystemExit(f"No inputs matched {args.glob!r}")
    rows = []
    for path in inputs:
        rows.extend(audit_case(path, args))
    stats = build_stats(rows)
    write_outputs(args, inputs, rows, stats)


if __name__ == "__main__":
    main()
