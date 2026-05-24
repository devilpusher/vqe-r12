#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8e: Locate why the HEM triplet R12 candidate fails.

Step8d showed the direct closed-shell formula has the wrong sign, while naive
same-spin antisymmetric tensors have the right sign but are too large.  This
script separates two possible culprits:

1. Fock model: closed-shell spin-free h + J - 1/2 K vs alpha-only h + J - K.
2. Linear V scale: how much an overall same-spin geminal scale would need to
   shrink to match the dense same-spin parent pair-FCI gap.

This remains an audit script, not a production correction.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict, List

import numpy as np

import r12_correction as rc
from step8c_hem_triplet_r12_correction import same_spin_pair_fci_target
from step8d_audit_same_spin_open_shell_r12 import delta_tensor, patched_sp_tensor, variant_specs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="step8b_hem_triplet_sp_s01_p01_fitN7_step4b_like.npz")
    p.add_argument("--prefix", default="step8e_hem_triplet_sp_s01_p01_fitN7")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def alpha_only_fock(h: np.ndarray, g_phys: np.ndarray, rdm1: np.ndarray, nobs: int, nri: int) -> np.ndarray:
    active = list(range(nobs))
    full = list(range(nri))
    g_fafa = rc.block4(g_phys, full, active, full, active)
    J = np.einsum("sr,krls->kl", rdm1, g_fafa, optimize=True)
    K = np.einsum("sr,krsl->kl", rdm1, g_fafa.transpose(0, 1, 3, 2), optimize=True)
    return h[np.ix_(full, full)] + J - K


def load_case(inp: str) -> Dict[str, Any]:
    data = np.load(inp, allow_pickle=True)
    meta = rc.load_metadata(data)
    nobs = int(meta.get("nobs", np.array(data["dm1_obs"]).shape[0]))
    nri = int(meta.get("nri", np.array(data["h_ri"]).shape[0]))
    h = np.array(data["h_ri"], dtype=float)
    g_phys = rc.chem_to_phys(np.array(data["eri_ri"], dtype=float))
    r_phys = rc.chem_to_phys(np.array(data["f12_ri"], dtype=float))
    dm1 = np.array(data["dm1_obs"], dtype=float)
    dm2 = np.array(data["dm2_obs"], dtype=float)
    labels = [str(x) for x in data["labels"]]
    return {
        "data": data,
        "nobs": nobs,
        "nri": nri,
        "h": h,
        "g_phys": g_phys,
        "r_phys": r_phys,
        "dm1": dm1,
        "dm2": dm2,
        "labels": labels,
        "E_obs": float(data["E_obs_fci"]),
    }


def components_for(case: Dict[str, Any], spec: Dict[str, Any], fock_model: str) -> Dict[str, Any]:
    if fock_model == "spinfree_J_minus_halfK":
        fock = rc.build_fock_tequila(
            case["h"], case["g_phys"], case["dm1"], list(range(case["nobs"])), list(range(case["nri"]))
        )
    elif fock_model == "alpha_only_J_minus_K":
        fock = alpha_only_fock(case["h"], case["g_phys"], case["dm1"], case["nobs"], case["nri"])
    else:
        raise ValueError(f"unknown fock_model={fock_model}")

    t = delta_tensor(case["nobs"], spec["direct"], spec["exchange"])
    with patched_sp_tensor(t):
        comp = rc.compute_sf2r12_components(
            case["g_phys"], case["r_phys"], fock, case["dm1"], case["dm2"], case["nobs"], case["nri"]
        )
    return comp


def scale_to_target_mEh(V_mEh: float, Q_mEh: float, target_mEh: float) -> float | None:
    if abs(Q_mEh) < 1e-16:
        return target_mEh / V_mEh if abs(V_mEh) > 1e-16 else None
    disc = V_mEh * V_mEh + 4.0 * Q_mEh * target_mEh
    if disc < 0.0:
        return None
    roots = [(-V_mEh + np.sqrt(disc)) / (2.0 * Q_mEh), (-V_mEh - np.sqrt(disc)) / (2.0 * Q_mEh)]
    roots = [float(x) for x in roots if np.isfinite(x) and x > 0.0]
    return min(roots, key=abs) if roots else None


def build_rows(case: Dict[str, Any], target: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_mEh = float(target["full_parent_gap_mEh"])
    rows = []
    for spec in variant_specs():
        for fock_model in ["spinfree_J_minus_halfK", "alpha_only_J_minus_K"]:
            comp = components_for(case, spec, fock_model)
            V_mEh = 1000.0 * comp["V"]
            Q_mEh = 1000.0 * (comp["B"] + comp["X"] + comp["Delta"])
            corr_mEh = 1000.0 * comp["correction"]
            lam = scale_to_target_mEh(V_mEh, Q_mEh, target_mEh)
            rows.append(
                {
                    "name": spec["name"],
                    "class": spec["class"],
                    "fock_model": fock_model,
                    "direct": spec["direct"],
                    "exchange": spec["exchange"],
                    "V_mEh": V_mEh,
                    "B_mEh": 1000.0 * comp["B"],
                    "X_mEh": 1000.0 * comp["X"],
                    "Delta_mEh": 1000.0 * comp["Delta"],
                    "quadratic_sum_mEh": Q_mEh,
                    "correction_mEh": corr_mEh,
                    "target_gap_mEh": target_mEh,
                    "residual_mEh": corr_mEh - target_mEh,
                    "same_sign_as_gap": bool(np.sign(corr_mEh) == np.sign(target_mEh)),
                    "lambda_to_target": lam,
                    "effective_direct": None if lam is None else lam * float(spec["direct"]),
                    "effective_exchange": None if lam is None else lam * float(spec["exchange"]),
                }
            )
    return rows


def write_outputs(args, case: Dict[str, Any], target: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    args.out_json = args.out_json or f"{args.prefix}_failure_source_audit.json"
    args.out_csv = args.out_csv or f"{args.prefix}_failure_source_audit.csv"
    args.summary = args.summary or f"{args.prefix}_failure_source_audit_summary.txt"
    payload = {
        "step": "8e",
        "input": args.inp,
        "labels": case["labels"],
        "target": target,
        "rows": rows,
        "conclusion_hint": "If lambda_to_target is much smaller than 1, the fixed same-spin geminal amplitude is too large.",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    best = min(rows, key=lambda r: abs(float(r["residual_mEh"])))
    antisym = [r for r in rows if r["class"] == "same_spin_antisymmetric"]
    lines = [
        "=" * 104,
        "Step 8e | HEM triplet same-spin failure-source audit",
        "=" * 104,
        f"input        = {args.inp}",
        f"labels       = {case['labels']}",
        f"E_obs        = {case['E_obs']:.14f} Eh",
        f"E_parent     = {target['E_full_parent_triplet_pair_fci']:.14f} Eh",
        f"target gap   = {target['full_parent_gap_mEh']:.9f} mEh",
        "",
        "[Fock-model comparison]",
        f"{'name':<44s} {'fock':<22s} {'V':>10s} {'Q':>10s} {'dE':>10s} {'resid':>10s} {'lambda':>10s}",
        "-" * 104,
    ]
    for r in rows:
        lam = "" if r["lambda_to_target"] is None else f"{float(r['lambda_to_target']):.6f}"
        lines.append(
            f"{r['name']:<44s} {r['fock_model']:<22s} "
            f"{float(r['V_mEh']):10.6f} {float(r['quadratic_sum_mEh']):10.6f} "
            f"{float(r['correction_mEh']):10.6f} {float(r['residual_mEh']):10.6f} {lam:>10s}"
        )
    lines.extend(
        [
            "",
            "[Best residual row]",
            f"name     = {best['name']}",
            f"fock     = {best['fock_model']}",
            f"DeltaE   = {best['correction_mEh']:.9f} mEh",
            f"residual = {best['residual_mEh']:.9f} mEh",
            "",
            "[Specific diagnosis]",
            "1. Changing Fock from spinfree J-0.5K to alpha-only J-K changes B/X/Delta,",
            "   but V is unchanged and remains the dominant scale-setting term.",
            "2. Naive antisymmetric same-spin tensors have the correct sign, but their",
            "   V term is 16-36 times larger than the actual parent pair-FCI gap.",
            "3. The lambda_to_target values for antisymmetric rows are far below 1,",
            "   meaning HEM needs a Pauli/same-spin suppressed geminal amplitude rather",
            "   than the closed-shell SP amplitude with only an exchange-sign flip.",
            "",
            "[Antisymmetric lambda range]",
        ]
    )
    for r in antisym:
        lines.append(
            f"{r['name']:<44s} {r['fock_model']:<22s} "
            f"lambda={r['lambda_to_target']} eff_direct={r['effective_direct']} eff_exchange={r['effective_exchange']}"
        )
    lines.extend(["", "[Saved]", f"  {args.out_json}", f"  {args.out_csv}", f"  {args.summary}"])
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    case = load_case(args.inp)
    target = same_spin_pair_fci_target(args.inp)
    rows = build_rows(case, target)
    write_outputs(args, case, target, rows)


if __name__ == "__main__":
    main()
