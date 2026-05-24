#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8k: HEM pair-channel weights and R12 V-numerator audit.

This is a diagnostic script.  It decomposes the same-spin antisymmetric V
numerator by active OBS pair labels/channels and compares it with the
alpha-alpha pair coefficient weights from the selected HEM FCI state.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import r12_correction as rc


@dataclass
class Case:
    name: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--prefix", default="step8k_hem_pair_channel_v_audit")
    p.add_argument("--direct", type=float, default=1.0 / 8.0)
    p.add_argument("--exchange", type=float, default=-1.0 / 8.0)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-channel-csv", default=None)
    p.add_argument("--out-label-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def cases() -> List[Case]:
    return [
        Case("sp_s01_p01"),
        Case("sp_s0123_p01"),
        Case("sp_s01_p0123"),
        Case("sp_s0123_p0123"),
    ]


def paths(case: Case, fitn: int) -> Dict[str, str]:
    return {
        "rdm_npz": f"step8h_hem_triplet_{case.name}_rdm_export.npz",
        "bridge_npz": f"step8b_hem_triplet_{case.name}_fitN{fitn}_step4b_like.npz",
    }


def channel(label: str) -> str:
    for ch in ("s", "p", "d", "f"):
        if f"ECG-{ch}" in label:
            return ch
    return "other"


def pair_key(a: str, b: str) -> str:
    return "-".join(sorted([a, b]))


def load_arrays(rdm_npz: str, bridge_npz: str) -> Dict[str, Any]:
    rdm = np.load(rdm_npz, allow_pickle=True)
    bridge = np.load(bridge_npz, allow_pickle=True)
    labels = [str(x) for x in bridge["labels"]]
    return {
        "labels": labels,
        "B": np.array(rdm["pair_coeff_upper"], dtype=float),
        "dm1": np.array(bridge["dm1_obs"], dtype=float),
        "dm2": np.array(bridge["dm2_obs"], dtype=float),
        "g_phys": rc.chem_to_phys(np.array(bridge["eri_ri"], dtype=float)),
        "r_phys": rc.chem_to_phys(np.array(bridge["f12_ri"], dtype=float)),
        "nobs": len(labels),
        "nri": np.array(bridge["h_ri"]).shape[0],
        "E_obs": float(bridge["E_obs_fci"]),
    }


def pair_weight_rows(case: str, labels: List[str], B: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows = []
    channel_weights: Dict[str, float] = {}
    n = B.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            w = float(B[i, j] ** 2)
            ck = pair_key(channel(labels[i]), channel(labels[j]))
            lk = pair_key(labels[i], labels[j])
            channel_weights[ck] = channel_weights.get(ck, 0.0) + w
            rows.append(
                {
                    "case": case,
                    "kind": "pair_weight",
                    "label_pair": lk,
                    "channel_pair": ck,
                    "p": i,
                    "q": j,
                    "coeff": float(B[i, j]),
                    "pair_weight": w,
                    "V_mEh": "",
                    "V_fraction": "",
                }
            )
    return rows, channel_weights


def build_v_mid(g_phys: np.ndarray, r_phys: np.ndarray, dm1: np.ndarray, nobs: int, nri: int) -> np.ndarray:
    a = list(range(nobs))
    p = list(range(nobs, nri))
    f = list(range(nri))
    gKLxy_rRSkl = np.einsum(
        "klxy,rskl->rsxy",
        rc.block4(g_phys, f, f, a, a),
        rc.block4(r_phys, a, a, f, f),
        optimize=True,
    )
    gTUxy_rRStu = np.einsum(
        "tuxy,rstu",
        rc.block4(g_phys, a, a, a, a),
        rc.block4(r_phys, a, a, a, a),
        optimize=True,
    )
    gATxy_rdm1Ut_rRSau = np.einsum(
        "atxy,ut,rsau",
        rc.block4(g_phys, p, a, a, a),
        dm1,
        rc.block4(r_phys, a, a, p, a),
        optimize=True,
    )
    return gKLxy_rRSkl - gTUxy_rRStu - gATxy_rdm1Ut_rRSau


def v_contribution_rows(
    case: str,
    labels: List[str],
    dm1: np.ndarray,
    dm2: np.ndarray,
    g_phys: np.ndarray,
    r_phys: np.ndarray,
    nobs: int,
    nri: int,
    direct: float,
    exchange: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    V_mid = build_v_mid(g_phys, r_phys, dm1, nobs, nri)
    label_acc: Dict[str, Dict[str, Any]] = {}
    channel_acc: Dict[str, float] = {}
    total = 0.0
    for p in range(nobs):
        for q in range(nobs):
            contrib = float(
                np.einsum(
                    "xy,xy->",
                    dm2[:, :, p, q],
                    direct * V_mid[p, q, :, :] + exchange * V_mid[q, p, :, :],
                    optimize=True,
                )
            )
            if abs(contrib) < 1e-18:
                continue
            lk = pair_key(labels[p], labels[q])
            ck = pair_key(channel(labels[p]), channel(labels[q]))
            total += contrib
            channel_acc[ck] = channel_acc.get(ck, 0.0) + contrib
            rec = label_acc.setdefault(
                lk,
                {
                    "case": case,
                    "kind": "V_contribution",
                    "label_pair": lk,
                    "channel_pair": ck,
                    "p": "",
                    "q": "",
                    "coeff": "",
                    "pair_weight": "",
                    "V_Eh": 0.0,
                },
            )
            rec["V_Eh"] += contrib
    rows = []
    for rec in label_acc.values():
        v = float(rec.pop("V_Eh"))
        rec["V_mEh"] = 1000.0 * v
        rec["V_fraction"] = v / total if abs(total) > 0.0 else None
        rows.append(rec)
    rows.sort(key=lambda r: abs(float(r["V_mEh"])), reverse=True)
    return rows, {k: 1000.0 * v for k, v in channel_acc.items()}


def channel_rows(case: str, channel_weights: Dict[str, float], channel_v_mEh: Dict[str, float]) -> List[Dict[str, Any]]:
    keys = sorted(set(channel_weights) | set(channel_v_mEh))
    total_v = sum(channel_v_mEh.values())
    rows = []
    for key in keys:
        v = channel_v_mEh.get(key, 0.0)
        rows.append(
            {
                "case": case,
                "channel_pair": key,
                "pair_weight": channel_weights.get(key, 0.0),
                "V_mEh": v,
                "V_fraction": v / total_v if abs(total_v) > 0.0 else None,
            }
        )
    return rows


def main():
    args = parse_args()
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_channel_csv = args.out_channel_csv or f"{args.prefix}_channel.csv"
    args.out_label_csv = args.out_label_csv or f"{args.prefix}_label.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"

    all_label_rows: List[Dict[str, Any]] = []
    all_channel_rows: List[Dict[str, Any]] = []
    payload_cases = []
    for case in cases():
        p = paths(case, args.fitN)
        if not Path(p["rdm_npz"]).exists() or not Path(p["bridge_npz"]).exists():
            raise FileNotFoundError(f"Missing Step8h/8b files for {case.name}: {p}")
        arr = load_arrays(p["rdm_npz"], p["bridge_npz"])
        weight_rows, ch_w = pair_weight_rows(case.name, arr["labels"], arr["B"])
        v_rows, ch_v = v_contribution_rows(
            case.name,
            arr["labels"],
            arr["dm1"],
            arr["dm2"],
            arr["g_phys"],
            arr["r_phys"],
            arr["nobs"],
            arr["nri"],
            args.direct,
            args.exchange,
        )
        ch_rows = channel_rows(case.name, ch_w, ch_v)
        all_label_rows.extend(weight_rows)
        all_label_rows.extend(v_rows)
        all_channel_rows.extend(ch_rows)
        payload_cases.append({"case": case.name, "nobs": arr["nobs"], "E_obs": arr["E_obs"], "channel_rows": ch_rows})

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "step": "8k",
                "fitN": args.fitN,
                "direct": args.direct,
                "exchange": args.exchange,
                "cases": payload_cases,
            },
            f,
            indent=2,
        )
    with open(args.out_channel_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_channel_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_channel_rows)
    with open(args.out_label_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_label_rows)

    lines = [
        "=" * 112,
        "Step 8k | HEM pair-channel weight and V-numerator audit",
        "=" * 112,
        f"fitN = {args.fitN}",
        f"same-spin tensor direct/exchange = {args.direct:.8f} / {args.exchange:.8f}",
        "",
        "[Channel summary]",
        f"{'case':<18s} {'channel':<8s} {'pair_w':>12s} {'V/mEh':>12s} {'V_frac':>12s}",
        "-" * 112,
    ]
    for r in all_channel_rows:
        vf = "" if r["V_fraction"] is None else f"{float(r['V_fraction']): .6f}"
        lines.append(
            f"{r['case']:<18s} {r['channel_pair']:<8s} {float(r['pair_weight']):12.6e} "
            f"{float(r['V_mEh']):12.6f} {vf:>12s}"
        )
    lines.extend(
        [
            "",
            "[Interpretation]",
            "Compare pair_weight against V_fraction.  A channel whose V_fraction changes",
            "strongly when s or p OBS is expanded is the likely source of scalar-rule",
            "failure in Step8j.",
            "",
            "[Saved]",
            f"  {args.out_json}",
            f"  {args.out_channel_csv}",
            f"  {args.out_label_csv}",
            f"  {args.summary}",
        ]
    )
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
