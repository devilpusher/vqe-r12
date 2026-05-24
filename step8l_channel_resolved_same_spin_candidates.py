#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8l: Channel-resolved same-spin R12 candidate scan for HEM.

This script moves beyond a single scalar same-spin suppression factor.  It
builds antisymmetric SP tensors whose active pair blocks can be scaled by OBS
pair channel:

    s-s, s-p, p-p

The rows here are still candidates/audits.  Oracle rows use the parent same-spin
pair-FCI target only to diagnose the best possible channel family.
"""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import r12_correction as rc
from step8c_hem_triplet_r12_correction import same_spin_pair_fci_target
from step8e_audit_hem_same_spin_failure_source import alpha_only_fock, load_case, scale_to_target_mEh
from step8f_hem_pauli_suppressed_geminal_audit import occupation_descriptors, pair_descriptors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--prefix", default="step8l_hem_channel_resolved_same_spin")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def cases() -> List[str]:
    return [
        "sp_s01_p01",
        "sp_s0123_p01",
        "sp_s012345_p01",
        "sp_s01_p0123",
        "sp_s01_p012345",
        "sp_s0123_p0123",
        "sp_s012345_p0123",
        "sp_s012345_p012345",
    ]


def paths(case: str, fitn: int) -> Dict[str, str]:
    return {
        "rdm_npz": f"step8h_hem_triplet_{case}_rdm_export.npz",
        "bridge_npz": f"step8b_hem_triplet_{case}_fitN{fitn}_step4b_like.npz",
    }


def channel(label: str) -> str:
    for ch in ("s", "p", "d", "f"):
        if f"ECG-{ch}" in label:
            return ch
    return "other"


def pair_channel(a: str, b: str) -> str:
    return "-".join(sorted([a, b]))


def radial_counts(labels: List[str]) -> Dict[str, int]:
    out: Dict[str, set[str]] = {"s": set(), "p": set(), "d": set(), "f": set()}
    for lab in labels:
        for ch in out:
            token = f"ECG-{ch}"
            if token in lab:
                radial = lab.split("_m")[0]
                out[ch].add(radial)
    return {k: len(v) for k, v in out.items()}


def channel_tensor(labels: List[str], scale: Dict[str, float], direct: float = 1.0 / 8.0, exchange: float = -1.0 / 8.0) -> np.ndarray:
    n = len(labels)
    t = np.zeros((n, n, n, n), dtype=float)
    ch = [channel(x) for x in labels]
    for p in range(n):
        for q in range(n):
            key = pair_channel(ch[p], ch[q])
            lam = float(scale.get(key, 0.0))
            t[p, q, p, q] += lam * direct
            t[p, q, q, p] += lam * exchange
    return t


@contextmanager
def patched_tensor(tensor: np.ndarray):
    old = rc.build_sp_tensor

    def builder(nobs: int) -> np.ndarray:
        if tensor.shape != (nobs, nobs, nobs, nobs):
            raise ValueError(f"tensor shape {tensor.shape} does not match nobs={nobs}")
        return np.array(tensor, copy=True)

    rc.build_sp_tensor = builder
    try:
        yield
    finally:
        rc.build_sp_tensor = old


def compute(case_data: Dict[str, Any], labels: List[str], scale: Dict[str, float], fock_model: str = "alpha") -> Dict[str, float]:
    if fock_model == "alpha":
        fock = alpha_only_fock(case_data["h"], case_data["g_phys"], case_data["dm1"], case_data["nobs"], case_data["nri"])
    elif fock_model == "spinfree":
        fock = rc.build_fock_tequila(
            case_data["h"], case_data["g_phys"], case_data["dm1"], list(range(case_data["nobs"])), list(range(case_data["nri"]))
        )
    else:
        raise ValueError(f"unknown fock_model={fock_model}")
    with patched_tensor(channel_tensor(labels, scale)):
        return rc.compute_sf2r12_components(
            case_data["g_phys"],
            case_data["r_phys"],
            fock,
            case_data["dm1"],
            case_data["dm2"],
            case_data["nobs"],
            case_data["nri"],
        )


def root_for_family(case_data: Dict[str, Any], labels: List[str], family: Dict[str, float], fock_model: str, target_mEh: float) -> float | None:
    comp = compute(case_data, labels, family, fock_model)
    v = 1000.0 * comp["V"]
    q = 1000.0 * (comp["B"] + comp["X"] + comp["Delta"])
    return scale_to_target_mEh(v, q, target_mEh)


def heuristic_vsat_q(case_name: str, labels: List[str], ref_vss: float, this_vss: float) -> float:
    counts = radial_counts(labels)
    ns = max(1, counts["s"])
    np_ = max(1, counts["p"])
    q = np.sqrt(abs(ref_vss) / max(abs(this_vss), 1e-30))
    if np_ > 2:
        q *= np.sqrt(2.0 / np_)
    if np_ > 2 and ns > 2:
        q *= 2.0 / ns
    return float(q)


def vss_unit(case_data: Dict[str, Any], labels: List[str]) -> float:
    comp = compute(case_data, labels, {"s-s": 1.0}, "alpha")
    return 1000.0 * comp["V"]


def add_row(
    rows: List[Dict[str, Any]],
    case: str,
    case_data: Dict[str, Any],
    labels: List[str],
    target: Dict[str, Any],
    desc: Dict[str, Any],
    pair: Dict[str, Any],
    model: str,
    scale: Dict[str, float],
    fock_model: str = "alpha",
) -> None:
    comp = compute(case_data, labels, scale, fock_model)
    corr_mEh = 1000.0 * comp["correction"]
    target_mEh = float(target["full_parent_gap_mEh"])
    rows.append(
        {
            "case": case,
            "nobs": case_data["nobs"],
            "fock_model": fock_model,
            "model": model,
            "scale_ss": scale.get("s-s", 0.0),
            "scale_sp": scale.get("p-s", 0.0),
            "scale_pp": scale.get("p-p", 0.0),
            "tail_sqrt": desc["sqrt_tail_occupation_sum_after_two"],
            "pair_sqrt": pair["sqrt_residual_pair_weight"],
            "target_gap_mEh": target_mEh,
            "correction_mEh": corr_mEh,
            "residual_mEh": corr_mEh - target_mEh,
            "V_mEh": 1000.0 * comp["V"],
            "B_mEh": 1000.0 * comp["B"],
            "X_mEh": 1000.0 * comp["X"],
            "Delta_mEh": 1000.0 * comp["Delta"],
        }
    )


def build_rows(fitn: int) -> List[Dict[str, Any]]:
    loaded = {}
    for case in cases():
        p = paths(case, fitn)
        if not Path(p["rdm_npz"]).exists() or not Path(p["bridge_npz"]).exists():
            raise FileNotFoundError(f"Missing prerequisite files for {case}; run make step8i first.")
        data = load_case(p["bridge_npz"])
        labels = [str(x) for x in np.load(p["bridge_npz"], allow_pickle=True)["labels"]]
        loaded[case] = {
            "paths": p,
            "data": data,
            "labels": labels,
            "target": same_spin_pair_fci_target(p["bridge_npz"]),
            "desc": occupation_descriptors(data["dm1"]),
            "pair": pair_descriptors(p["rdm_npz"]),
            "vss_unit": vss_unit(data, labels),
        }
    ref_vss = loaded["sp_s01_p01"]["vss_unit"]

    rows: List[Dict[str, Any]] = []
    for case, item in loaded.items():
        data = item["data"]
        labels = item["labels"]
        target = item["target"]
        desc = item["desc"]
        pair = item["pair"]
        target_mEh = float(target["full_parent_gap_mEh"])
        tail2 = 2.0 * desc["sqrt_tail_occupation_sum_after_two"]
        pair3 = 3.0 * pair["sqrt_residual_pair_weight"]
        vsat = heuristic_vsat_q(case, labels, ref_vss, item["vss_unit"]) * tail2
        candidates = [
            ("scalar_tail2_all", {"s-s": tail2, "p-s": tail2, "p-p": tail2}),
            ("scalar_pair3_all", {"s-s": pair3, "p-s": pair3, "p-p": pair3}),
            ("tail2_ss_only", {"s-s": tail2}),
            ("tail2_ss_sp", {"s-s": tail2, "p-s": tail2}),
            ("vsat_tail2_ss_only", {"s-s": vsat}),
            ("vsat_tail2_ss_sp", {"s-s": vsat, "p-s": vsat}),
        ]
        for name, scale in candidates:
            add_row(rows, case, data, labels, target, desc, pair, name, scale, "alpha")
        for fam_name, fam in [("oracle_ss_only", {"s-s": 1.0}), ("oracle_ss_sp", {"s-s": 1.0, "p-s": 1.0})]:
            lam = root_for_family(data, labels, fam, "alpha", target_mEh)
            if lam is not None:
                add_row(
                    rows,
                    case,
                    data,
                    labels,
                    target,
                    desc,
                    pair,
                    fam_name,
                    {k: lam * v for k, v in fam.items()},
                    "alpha",
                )
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        abs_res = np.array([abs(float(r["residual_mEh"])) for r in sub])
        out[model] = {
            "mean_abs_residual_mEh": float(np.mean(abs_res)),
            "max_abs_residual_mEh": float(np.max(abs_res)),
        }
    return out


def main():
    args = parse_args()
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    rows = build_rows(args.fitN)
    summary = summarize(rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"step": "8l", "fitN": args.fitN, "model_summary": summary, "rows": rows}, f, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=" * 112,
        "Step 8l | HEM channel-resolved same-spin candidate scan",
        "=" * 112,
        f"fitN = {args.fitN}",
        "",
        "[Model residual summary]",
    ]
    for model, vals in summary.items():
        lines.append(
            f"{model:<24s} mean_abs={vals['mean_abs_residual_mEh']:.6f} mEh  "
            f"max_abs={vals['max_abs_residual_mEh']:.6f} mEh"
        )
    lines.extend(
        [
            "",
            "[Rows]",
            f"{'case':<18s} {'model':<24s} {'ss':>9s} {'sp':>9s} {'pp':>9s} {'gap':>11s} {'dE':>11s} {'resid':>11s}",
            "-" * 112,
        ]
    )
    for r in rows:
        lines.append(
            f"{r['case']:<18s} {r['model']:<24s} {float(r['scale_ss']):9.5f} {float(r['scale_sp']):9.5f} "
            f"{float(r['scale_pp']):9.5f} {float(r['target_gap_mEh']):11.6f} {float(r['correction_mEh']):11.6f} "
            f"{float(r['residual_mEh']):11.6f}"
        )
    lines.extend(
        [
            "",
            "[Conclusion]",
            "Channel resolution separates the dominant s-s numerator from p-s cancellation.",
            "Oracle rows show that an s-s or s-s+s-p family can represent each residual,",
            "but the simple vsat heuristic is still not a transferable final rule.",
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


if __name__ == "__main__":
    main()
