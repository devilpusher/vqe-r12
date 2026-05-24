#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8d: Audit same-spin/open-shell [2]R12 formula choices for HEM.

This is a diagnostic script, not a production energy formula.  The Step8c
result showed that directly reusing the closed-shell spin-free SP tensor gives a
positive correction for the He metastable triplet.  Here we keep the same
parent-basis/CABS-only V/B/X/Delta machinery but scan the suspicious
same-spin pieces explicitly:

* SP amplitude normalization and exchange sign;
* closed-shell vs antisymmetric same-spin tensor structure;
* V/B/X/Delta partial sums against a dense same-spin parent pair-FCI target.

The minimal HEM orbital space is s01+p01.  This script enforces that the active
ECG-NO labels include even numbers of selected s and p spatial orbitals.
"""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List

import numpy as np

import r12_correction
from r12_correction import compute_he_sf2r12_correction, validate_correction_result
from step8c_hem_triplet_r12_correction import same_spin_pair_fci_target


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="step8b_hem_triplet_sp_s01_p01_fitN7_step4b_like.npz")
    p.add_argument("--prefix", default="step8d_hem_triplet_sp_s01_p01_fitN7")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--scale-f12", type=float, default=1.0)
    p.add_argument("--strict-even-sp", action="store_true", default=True)
    return p.parse_args()


def delta_tensor(n: int, direct: float, exchange: float) -> np.ndarray:
    t = np.zeros((n, n, n, n), dtype=float)
    for p in range(n):
        for q in range(n):
            t[p, q, p, q] += direct
            t[p, q, q, p] += exchange
    return t


def variant_specs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "closed_shell_sp_current",
            "direct": 3.0 / 8.0,
            "exchange": 1.0 / 8.0,
            "class": "closed_shell_singlet_reference",
            "note": "Current Step8c row; included as a negative control for HEM triplet.",
        },
        {
            "name": "same_spin_antisym_1_4",
            "direct": 1.0 / 4.0,
            "exchange": -1.0 / 4.0,
            "class": "same_spin_antisymmetric",
            "note": "Antisymmetric delta pair with same-spin cusp prefactor candidate.",
        },
        {
            "name": "same_spin_antisym_1_8",
            "direct": 1.0 / 8.0,
            "exchange": -1.0 / 8.0,
            "class": "same_spin_antisymmetric",
            "note": "Half of the 1/4 antisymmetric candidate.",
        },
        {
            "name": "same_spin_antisym_1_2",
            "direct": 1.0 / 2.0,
            "exchange": -1.0 / 2.0,
            "class": "same_spin_antisymmetric",
            "note": "Larger normalization check; useful because B/X/Delta are quadratic in t.",
        },
        {
            "name": "same_spin_direct_only_1_4",
            "direct": 1.0 / 4.0,
            "exchange": 0.0,
            "class": "projector_exchange_control",
            "note": "Exchange removed; diagnoses whether antisymmetrizer/exchange is driving the sign.",
        },
        {
            "name": "same_spin_direct_only_1_8",
            "direct": 1.0 / 8.0,
            "exchange": 0.0,
            "class": "projector_exchange_control",
            "note": "Smaller direct-only control.",
        },
        {
            "name": "opposite_closed_shell_exchange",
            "direct": 3.0 / 8.0,
            "exchange": -1.0 / 8.0,
            "class": "exchange_sign_control",
            "note": "Keeps closed-shell direct prefactor but flips exchange sign.",
        },
    ]


@contextmanager
def patched_sp_tensor(tensor: np.ndarray):
    old = r12_correction.build_sp_tensor

    def builder(nobs: int) -> np.ndarray:
        if tensor.shape != (nobs, nobs, nobs, nobs):
            raise ValueError(f"patched tensor shape {tensor.shape} does not match nobs={nobs}")
        return np.array(tensor, copy=True)

    r12_correction.build_sp_tensor = builder
    try:
        yield
    finally:
        r12_correction.build_sp_tensor = old


def selected_channel_counts(labels: Iterable[Any]) -> Dict[str, int]:
    counts = {"s": 0, "p": 0, "d": 0, "f": 0, "other": 0}
    for x in labels:
        s = str(x)
        matched = False
        for ch in ("s", "p", "d", "f"):
            if f"ECG-{ch}" in s:
                counts[ch] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1
    return counts


def audit_space(inp: str, strict_even_sp: bool) -> Dict[str, Any]:
    data = np.load(inp, allow_pickle=True)
    labels = [str(x) for x in data["labels"]]
    counts = selected_channel_counts(labels)
    failures = []
    if strict_even_sp:
        if counts["s"] <= 0 or counts["s"] % 2 != 0:
            failures.append("selected_s_spatial_orbitals_not_positive_even")
        if counts["p"] <= 0 or counts["p"] % 2 != 0:
            failures.append("selected_p_spatial_orbitals_not_positive_even")
    return {
        "labels": labels,
        "counts": counts,
        "nobs": int(np.array(data["dm1_obs"]).shape[0]),
        "strict_even_sp": bool(strict_even_sp),
        "passed": not failures,
        "failures": failures,
    }


def row_from_result(spec: Dict[str, Any], result: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    comp = result["components"]
    E_obs = float(result["E_obs_fci"])
    E_full = float(target["E_full_parent_triplet_pair_fci"])
    gap = E_full - E_obs
    rowsum_v = comp["V"]
    rowsum_vb = comp["V"] + comp["B"]
    rowsum_vbx = comp["V"] + comp["B"] + comp["X"]
    rowsum_full = comp["correction"]
    same_sign = np.sign(rowsum_full) == np.sign(gap) if abs(gap) > 0.0 else False
    recovery = rowsum_full / gap if abs(gap) > 0.0 else None
    return {
        "name": spec["name"],
        "class": spec["class"],
        "direct": spec["direct"],
        "exchange": spec["exchange"],
        "T_0000": float(delta_tensor(1, spec["direct"], spec["exchange"])[0, 0, 0, 0]),
        "V": comp["V"],
        "B": comp["B"],
        "X": comp["X"],
        "Delta": comp["Delta"],
        "V_only": rowsum_v,
        "V_plus_B": rowsum_vb,
        "V_plus_B_plus_X": rowsum_vbx,
        "correction": rowsum_full,
        "correction_mEh": 1000.0 * rowsum_full,
        "E_total": E_obs + rowsum_full,
        "target_gap_mEh": 1000.0 * gap,
        "residual_to_parent_mEh": 1000.0 * (E_obs + rowsum_full - E_full),
        "recovery_ratio": recovery,
        "same_sign_as_parent_gap": bool(same_sign),
        "note": spec["note"],
    }


def run_variant(inp: str, spec: Dict[str, Any], nobs: int, target: Dict[str, Any], scale_f12: float) -> Dict[str, Any]:
    t = delta_tensor(nobs, spec["direct"], spec["exchange"])
    exchange_symmetry = float(np.max(np.abs(t + t.transpose(1, 0, 2, 3)))) if nobs else 0.0
    with patched_sp_tensor(t):
        result = compute_he_sf2r12_correction(inp, step4b_path=None, scale_f12=scale_f12)
    validate_correction_result(result)
    row = row_from_result(spec, result, target)
    row["pair_bra_antisym_error"] = exchange_symmetry
    return row


def write_outputs(args, space: Dict[str, Any], target: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    args.out_json = args.out_json or f"{args.prefix}_same_spin_open_shell_audit.json"
    args.out_csv = args.out_csv or f"{args.prefix}_same_spin_open_shell_audit.csv"
    args.summary = args.summary or f"{args.prefix}_same_spin_open_shell_audit_summary.txt"

    payload = {
        "step": "8d",
        "purpose": "same-spin/open-shell [2]R12 formula audit for HEM triplet",
        "input": args.inp,
        "space_audit": space,
        "target": target,
        "rows": rows,
        "interpretation_guardrail": [
            "Rows are formula diagnostics, not accepted production corrections.",
            "A physically useful candidate should have the same sign as the parent pair-FCI gap before any empirical fitting.",
            "The current closed-shell SP tensor is retained only as a negative control for the triplet.",
        ],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "name",
        "class",
        "direct",
        "exchange",
        "T_0000",
        "V",
        "B",
        "X",
        "Delta",
        "V_only",
        "V_plus_B",
        "V_plus_B_plus_X",
        "correction",
        "correction_mEh",
        "E_total",
        "target_gap_mEh",
        "residual_to_parent_mEh",
        "recovery_ratio",
        "same_sign_as_parent_gap",
        "pair_bra_antisym_error",
        "note",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda r: abs(float(r["residual_to_parent_mEh"])))
    lines = [
        "=" * 104,
        "Step 8d | HEM triplet same-spin/open-shell [2]R12 formula audit",
        "=" * 104,
        f"input             = {args.inp}",
        f"nobs              = {space['nobs']}",
        f"labels            = {space['labels']}",
        f"channel counts    = {space['counts']}",
        f"space audit passed= {space['passed']}",
        "",
        "[Parent same-spin pair-FCI target]",
        f"E_full_parent_triplet = {target['E_full_parent_triplet_pair_fci']:.14f} Eh",
        f"OBS -> parent gap      = {target['full_parent_gap_mEh']:.9f} mEh",
        f"pair dimension         = {target['pair_dimension']}",
        "",
        "[Variant scan]",
        f"{'name':<32s} {'dir':>8s} {'exch':>8s} {'dE/mEh':>12s} {'resid/mEh':>12s} {'recov':>12s} {'sign':>7s}",
        "-" * 104,
    ]
    for r in rows:
        rec = "" if r["recovery_ratio"] is None else f"{float(r['recovery_ratio']): .6f}"
        lines.append(
            f"{r['name']:<32s} {float(r['direct']):8.4f} {float(r['exchange']):8.4f} "
            f"{float(r['correction_mEh']):12.6f} {float(r['residual_to_parent_mEh']):12.6f} "
            f"{rec:>12s} {str(r['same_sign_as_parent_gap']):>7s}"
        )
    lines.extend(
        [
            "",
            "[Closest row by parent residual]",
            f"name       = {best['name']}",
            f"DeltaE     = {float(best['correction_mEh']):.9f} mEh",
            f"residual   = {float(best['residual_to_parent_mEh']):.9f} mEh",
            f"same sign  = {best['same_sign_as_parent_gap']}",
            "",
            "[Audit conclusion]",
            "The accepted HEM triplet formula is still open.  This scan only localizes which",
            "prefactor/exchange choices move the closed-shell result toward or away from the",
            "same-spin parent pair-FCI target.",
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
    space = audit_space(args.inp, strict_even_sp=args.strict_even_sp)
    if not space["passed"]:
        raise SystemExit(f"Space audit failed: {space['failures']}")
    target = same_spin_pair_fci_target(args.inp)
    rows = [run_variant(args.inp, spec, space["nobs"], target, args.scale_f12) for spec in variant_specs()]
    write_outputs(args, space, target, rows)


if __name__ == "__main__":
    main()
