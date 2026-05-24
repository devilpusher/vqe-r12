#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8f: Pauli/same-spin suppressed geminal audit for HEM triplet.

The HEM s01+p01 parent/OBS gap is only about -0.016 mEh.  Step8e showed that
standard same-spin antisymmetric SP amplitudes have the correct sign but a V
numerator that is one order of magnitude too large.  This script asks a narrower
question:

    How much must the same-spin geminal amplitude be suppressed, and do simple
    HEM state descriptors predict a similar scale?

No row in this script is a production formula.  The "target-fit" rows are
explicit diagnostics against the dense same-spin parent pair-FCI gap.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict, List

import numpy as np

import r12_correction as rc
from step8c_hem_triplet_r12_correction import same_spin_pair_fci_target
from step8d_audit_same_spin_open_shell_r12 import delta_tensor, patched_sp_tensor
from step8e_audit_hem_same_spin_failure_source import alpha_only_fock, load_case, scale_to_target_mEh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="step8b_hem_triplet_sp_s01_p01_fitN7_step4b_like.npz")
    p.add_argument("--pair-inp", default="step8a_hem_triplet_ecg_no_fci_rdm_export.npz")
    p.add_argument("--prefix", default="step8f_hem_triplet_sp_s01_p01_fitN7")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def pair_descriptors(pair_inp: str) -> Dict[str, Any]:
    data = np.load(pair_inp, allow_pickle=True)
    B = np.array(data["pair_coeff_upper"], dtype=float)
    labels = [str(x) for x in data["labels"]]
    weights = []
    for p in range(B.shape[0]):
        for q in range(p + 1, B.shape[1]):
            weights.append((float(B[p, q] ** 2), p, q, labels[p], labels[q], float(B[p, q])))
    weights.sort(reverse=True)
    leading = weights[0]
    residual_pair_weight = max(0.0, 1.0 - leading[0])
    return {
        "labels": labels,
        "leading_pair": {
            "weight": leading[0],
            "p": leading[1],
            "q": leading[2],
            "label_p": leading[3],
            "label_q": leading[4],
            "coefficient": leading[5],
        },
        "residual_pair_weight": residual_pair_weight,
        "sqrt_residual_pair_weight": float(np.sqrt(residual_pair_weight)),
        "top_pairs": [
            {
                "weight": w,
                "p": p,
                "q": q,
                "label_p": lp,
                "label_q": lq,
                "coefficient": c,
            }
            for w, p, q, lp, lq, c in weights[:8]
        ],
    }


def occupation_descriptors(dm1: np.ndarray) -> Dict[str, Any]:
    occ = np.linalg.eigvalsh(0.5 * (dm1 + dm1.T))[::-1]
    tail_sum = float(np.sum(occ[2:]))
    leading_hole_sum = float(np.sum(1.0 - occ[:2]))
    return {
        "natural_occupations": occ.tolist(),
        "tail_occupation_sum_after_two": tail_sum,
        "sqrt_tail_occupation_sum_after_two": float(np.sqrt(max(0.0, tail_sum))),
        "leading_two_hole_sum": leading_hole_sum,
        "sqrt_leading_two_hole_sum": float(np.sqrt(max(0.0, leading_hole_sum))),
        "linear_entropy_alpha": float(np.trace(dm1) - np.trace(dm1 @ dm1)),
    }


def components(case: Dict[str, Any], direct: float, exchange: float, fock_model: str) -> Dict[str, float]:
    if fock_model == "spinfree":
        fock = rc.build_fock_tequila(
            case["h"], case["g_phys"], case["dm1"], list(range(case["nobs"])), list(range(case["nri"]))
        )
    elif fock_model == "alpha":
        fock = alpha_only_fock(case["h"], case["g_phys"], case["dm1"], case["nobs"], case["nri"])
    else:
        raise ValueError(f"unknown fock_model={fock_model}")
    with patched_sp_tensor(delta_tensor(case["nobs"], direct, exchange)):
        return rc.compute_sf2r12_components(
            case["g_phys"], case["r_phys"], fock, case["dm1"], case["dm2"], case["nobs"], case["nri"]
        )


def suppression_models(desc: Dict[str, Any], pair: Dict[str, Any], lambda_exact: float | None) -> List[Dict[str, Any]]:
    tail_sqrt = float(desc["sqrt_tail_occupation_sum_after_two"])
    hole_sqrt = float(desc["sqrt_leading_two_hole_sum"])
    pair_sqrt = float(pair["sqrt_residual_pair_weight"])
    models = [
        ("none", 1.0, "No suppression; reproduces the naive same-spin row."),
        ("sqrt_tail_occ", tail_sqrt, "sqrt(sum occupations outside the two dominant alpha NOs)."),
        ("2sqrt_tail_occ", 2.0 * tail_sqrt, "A simple occupancy-scale stress test."),
        ("sqrt_leading_hole", hole_sqrt, "sqrt(hole sum in the two dominant alpha NOs)."),
        ("sqrt_residual_pair_weight", pair_sqrt, "sqrt(1 - largest alpha-alpha pair weight)."),
        ("3sqrt_residual_pair_weight", 3.0 * pair_sqrt, "Pair-weight stress test."),
    ]
    if lambda_exact is not None:
        models.append(("target_fit_lambda", float(lambda_exact), "Diagnostic lambda fitted to the parent pair-FCI gap."))
    return [{"model": name, "lambda": float(lam), "note": note} for name, lam, note in models]


def build_rows(case: Dict[str, Any], target: Dict[str, Any], desc: Dict[str, Any], pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_mEh = float(target["full_parent_gap_mEh"])
    base_direct = 1.0 / 8.0
    base_exchange = -1.0 / 8.0
    rows: List[Dict[str, Any]] = []
    for fock_model in ["spinfree", "alpha"]:
        base = components(case, base_direct, base_exchange, fock_model)
        V_mEh = 1000.0 * base["V"]
        Q_mEh = 1000.0 * (base["B"] + base["X"] + base["Delta"])
        lam_exact = scale_to_target_mEh(V_mEh, Q_mEh, target_mEh)
        for model in suppression_models(desc, pair, lam_exact):
            lam = model["lambda"]
            direct = lam * base_direct
            exchange = lam * base_exchange
            comp = components(case, direct, exchange, fock_model)
            corr_mEh = 1000.0 * comp["correction"]
            rows.append(
                {
                    "fock_model": fock_model,
                    "suppression_model": model["model"],
                    "lambda": lam,
                    "direct": direct,
                    "exchange": exchange,
                    "V_mEh": 1000.0 * comp["V"],
                    "B_mEh": 1000.0 * comp["B"],
                    "X_mEh": 1000.0 * comp["X"],
                    "Delta_mEh": 1000.0 * comp["Delta"],
                    "correction_mEh": corr_mEh,
                    "target_gap_mEh": target_mEh,
                    "residual_mEh": corr_mEh - target_mEh,
                    "same_sign_as_gap": bool(np.sign(corr_mEh) == np.sign(target_mEh)),
                    "note": model["note"],
                }
            )
    return rows


def write_outputs(
    args,
    case: Dict[str, Any],
    target: Dict[str, Any],
    desc: Dict[str, Any],
    pair: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    args.out_json = args.out_json or f"{args.prefix}_pauli_suppressed_geminal_audit.json"
    args.out_csv = args.out_csv or f"{args.prefix}_pauli_suppressed_geminal_audit.csv"
    args.summary = args.summary or f"{args.prefix}_pauli_suppressed_geminal_audit_summary.txt"

    payload = {
        "step": "8f",
        "input": args.inp,
        "pair_input": args.pair_inp,
        "target": target,
        "occupation_descriptors": desc,
        "pair_descriptors": pair,
        "rows": rows,
        "guardrail": "target_fit_lambda is diagnostic only and must not be treated as an ab initio formula.",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda r: abs(float(r["residual_mEh"])))
    lines = [
        "=" * 108,
        "Step 8f | HEM triplet Pauli/same-spin suppressed geminal audit",
        "=" * 108,
        f"input        = {args.inp}",
        f"pair_input   = {args.pair_inp}",
        f"E_obs        = {case['E_obs']:.14f} Eh",
        f"E_parent     = {target['E_full_parent_triplet_pair_fci']:.14f} Eh",
        f"target gap   = {target['full_parent_gap_mEh']:.9f} mEh",
        "",
        "[State descriptors]",
        "natural occupations = " + np.array2string(np.array(desc["natural_occupations"]), precision=10),
        f"tail occ sum after two        = {desc['tail_occupation_sum_after_two']:.12e}",
        f"sqrt tail occ sum             = {desc['sqrt_tail_occupation_sum_after_two']:.12e}",
        f"leading two hole sum           = {desc['leading_two_hole_sum']:.12e}",
        f"sqrt leading two hole sum      = {desc['sqrt_leading_two_hole_sum']:.12e}",
        f"leading pair weight            = {pair['leading_pair']['weight']:.12e} ({pair['leading_pair']['label_p']} ^ {pair['leading_pair']['label_q']})",
        f"sqrt residual pair weight      = {pair['sqrt_residual_pair_weight']:.12e}",
        "",
        "[Suppressed antisymmetric same-spin rows]",
        f"{'fock':<8s} {'model':<28s} {'lambda':>10s} {'dir':>10s} {'V':>12s} {'dE':>12s} {'resid':>12s}",
        "-" * 108,
    ]
    for r in rows:
        lines.append(
            f"{r['fock_model']:<8s} {r['suppression_model']:<28s} "
            f"{float(r['lambda']):10.6f} {float(r['direct']):10.6f} "
            f"{float(r['V_mEh']):12.6f} {float(r['correction_mEh']):12.6f} {float(r['residual_mEh']):12.6f}"
        )
    lines.extend(
        [
            "",
            "[Best residual row]",
            f"fock       = {best['fock_model']}",
            f"model      = {best['suppression_model']}",
            f"lambda     = {best['lambda']:.12e}",
            f"direct     = {best['direct']:.12e}",
            f"exchange   = {best['exchange']:.12e}",
            f"DeltaE     = {best['correction_mEh']:.9f} mEh",
            f"residual   = {best['residual_mEh']:.9f} mEh",
            "",
            "[Progress report]",
            "1. The HEM parent/OBS gap is extremely small, so a normal closed-shell-size",
            "   geminal amplitude is not a useful scale for the same-spin block.",
            "2. Matching the parent gap with an antisymmetric same-spin tensor requires",
            "   an effective direct/exchange magnitude near 7.1e-3, not 1/8 or 1/4.",
            "3. Simple state descriptors naturally live in this small-amplitude regime:",
            "   sqrt(tail occupation) is about 2.7e-2, and sqrt(residual pair weight)",
            "   is about 1.9e-2.  They are still not a final formula, but they show why",
            "   a Pauli-suppressed same-spin ansatz is the right next direction.",
            "4. The fitted target lambda is only a calibration diagnostic.  A publishable",
            "   formula needs a state-defined q_same_spin or pair-density suppression rule",
            "   that transfers across HEM spaces and, later, other open-shell states.",
            "",
            "[Saved]",
            f"  {args.out_json}",
            f"  {args.out_csv}",
            f"  {args.summary}",
        ]
    )
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    case = load_case(args.inp)
    target = same_spin_pair_fci_target(args.inp)
    desc = occupation_descriptors(case["dm1"])
    pair = pair_descriptors(args.pair_inp)
    rows = build_rows(case, target, desc, pair)
    write_outputs(args, case, target, desc, pair, rows)


if __name__ == "__main__":
    main()
