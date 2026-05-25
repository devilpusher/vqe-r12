#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 10e: Be pair/channel audit for selected-space R12 corrections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import r12_correction as rc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--cases", default="sp_s012_p0,sp_s0123_p0,sp_s012_p01,sp_s0123_p01")
    p.add_argument("--prefix", default="step10e_be_pair_channel_audit_fitN7")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-channel-csv", default=None)
    p.add_argument("--out-spin-csv", default=None)
    p.add_argument("--out-label-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def parse_cases(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def paths(case: str, fitn: int) -> Dict[str, str]:
    return {
        "rdm_npz": f"step10a_be_{case}_rdm_export.npz",
        "bridge_npz": f"step10b_be_{case}_fitN{fitn}_step4b_like.npz",
        "corr_json": f"step10c_be_{case}_fitN{fitn}_sf2r12_correction.json",
    }


def channel(label: str) -> str:
    for ch in ("s", "p", "d", "f"):
        if f"ECG-{ch}" in label:
            return ch
    return "other"


def pair_key(a: str, b: str) -> str:
    return "-".join(sorted([a, b]))


def spin_pair(pair: Tuple[int, int], nobs: int) -> str:
    p, q = pair
    if p < nobs and q < nobs:
        return "aa"
    if p >= nobs and q >= nobs:
        return "bb"
    return "ab"


def spinorb_channel(pair: Tuple[int, int], labels: List[str]) -> str:
    nobs = len(labels)
    p, q = pair
    return pair_key(channel(labels[p % nobs]), channel(labels[q % nobs]))


def load_case(case: str, fitn: int) -> Dict[str, Any]:
    p = paths(case, fitn)
    missing = [v for v in p.values() if not Path(v).exists()]
    if missing:
        raise FileNotFoundError(f"Missing prerequisite files for {case}: {missing}. Run step10d first.")
    rdm = np.load(p["rdm_npz"], allow_pickle=True)
    bridge = np.load(p["bridge_npz"], allow_pickle=True)
    with open(p["corr_json"], "r", encoding="utf-8") as f:
        corr = json.load(f)["result"]
    labels = [str(x) for x in bridge["labels"]]
    nobs = len(labels)
    eri_ri = np.array(bridge["eri_ri"], dtype=float)
    f12_ri = np.array(bridge["f12_ri"], dtype=float)
    dm1 = np.array(bridge["dm1_obs"], dtype=float)
    dm2 = np.array(bridge["dm2_obs"], dtype=float)
    g_phys = rc.chem_to_phys(eri_ri)
    r_phys = rc.chem_to_phys(f12_ri)
    fock = rc.build_fock_tequila(
        np.array(bridge["h_ri"], dtype=float),
        g_phys,
        dm1,
        list(range(nobs)),
        list(range(g_phys.shape[0])),
    )
    return {
        "case": case,
        "paths": p,
        "labels": labels,
        "nobs": nobs,
        "nri": g_phys.shape[0],
        "dm1": dm1,
        "dm2": dm2,
        "g_phys": g_phys,
        "r_phys": r_phys,
        "fock": fock,
        "D_pair": np.array(rdm["D_pair"], dtype=float),
        "pair_list": [tuple(map(int, x)) for x in np.array(rdm["pair_list"], dtype=int)],
        "corr": corr,
    }


def pair_population_rows(case: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels = case["labels"]
    nobs = case["nobs"]
    D = case["D_pair"]
    pair_list = case["pair_list"]
    total = float(np.trace(D))
    channel_acc: Dict[Tuple[str, str], float] = {}
    spin_acc: Dict[str, float] = {}
    label_rows = []
    for i, pair in enumerate(pair_list):
        w = float(D[i, i])
        sp = spin_pair(pair, nobs)
        ch = spinorb_channel(pair, labels)
        p, q = pair
        lp = labels[p % nobs]
        lq = labels[q % nobs]
        channel_acc[(sp, ch)] = channel_acc.get((sp, ch), 0.0) + w
        spin_acc[sp] = spin_acc.get(sp, 0.0) + w
        if w > 1e-8:
            label_rows.append(
                {
                    "case": case["case"],
                    "kind": "pair_population",
                    "spin_pair": sp,
                    "channel_pair": ch,
                    "label_pair": pair_key(lp, lq),
                    "p": p,
                    "q": q,
                    "weight": w,
                    "fraction": w / total if total else None,
                    "V_mEh": "",
                    "V_fraction": "",
                }
            )
    rows = []
    for (sp, ch), w in sorted(channel_acc.items()):
        rows.append(
            {
                "case": case["case"],
                "spin_pair": sp,
                "channel_pair": ch,
                "pair_weight": w,
                "pair_fraction": w / total if total else None,
                "V_mEh": "",
                "V_fraction": "",
            }
        )
    spin_rows = [
        {
            "case": case["case"],
            "spin_pair": sp,
            "pair_weight": w,
            "pair_fraction": w / total if total else None,
        }
        for sp, w in sorted(spin_acc.items())
    ]
    return rows + label_rows, spin_rows


def build_v_mid(case: Dict[str, Any]) -> np.ndarray:
    nobs = case["nobs"]
    nri = case["nri"]
    a = list(range(nobs))
    p = list(range(nobs, nri))
    f = list(range(nri))
    g = case["g_phys"]
    r = case["r_phys"]
    dm1 = case["dm1"]
    gKLxy_rRSkl = np.einsum("klxy,rskl->rsxy", rc.block4(g, f, f, a, a), rc.block4(r, a, a, f, f), optimize=True)
    gTUxy_rRStu = np.einsum("tuxy,rstu->rsxy", rc.block4(g, a, a, a, a), rc.block4(r, a, a, a, a), optimize=True)
    gATxy_rdm1Ut_rRSau = np.einsum(
        "atxy,ut,rsau->rsxy",
        rc.block4(g, p, a, a, a),
        dm1,
        rc.block4(r, a, a, p, a),
        optimize=True,
    )
    return gKLxy_rRSkl - gTUxy_rRStu - gATxy_rdm1Ut_rRSau


def v_channel_rows(case: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels = case["labels"]
    nobs = case["nobs"]
    dm2 = case["dm2"]
    V_mid = build_v_mid(case)
    direct = 3.0 / 8.0
    exchange = 1.0 / 8.0
    channel_acc: Dict[str, float] = {}
    label_acc: Dict[str, Dict[str, Any]] = {}
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
            ck = pair_key(channel(labels[p]), channel(labels[q]))
            lk = pair_key(labels[p], labels[q])
            total += contrib
            channel_acc[ck] = channel_acc.get(ck, 0.0) + contrib
            rec = label_acc.setdefault(
                lk,
                {
                    "case": case["case"],
                    "kind": "V_contribution",
                    "spin_pair": "spinfree",
                    "channel_pair": ck,
                    "label_pair": lk,
                    "p": "",
                    "q": "",
                    "weight": "",
                    "fraction": "",
                    "V_Eh": 0.0,
                },
            )
            rec["V_Eh"] += contrib
    channel_rows = []
    for ck, v in sorted(channel_acc.items()):
        channel_rows.append(
            {
                "case": case["case"],
                "spin_pair": "spinfree",
                "channel_pair": ck,
                "pair_weight": "",
                "pair_fraction": "",
                "V_mEh": 1000.0 * v,
                "V_fraction": v / total if abs(total) else None,
            }
        )
    label_rows = []
    for rec in label_acc.values():
        v = float(rec.pop("V_Eh"))
        rec["V_mEh"] = 1000.0 * v
        rec["V_fraction"] = v / total if abs(total) else None
        label_rows.append(rec)
    label_rows.sort(key=lambda r: abs(float(r["V_mEh"])), reverse=True)
    return channel_rows, label_rows


def write_outputs(args, channel_rows: List[Dict[str, Any]], spin_rows: List[Dict[str, Any]], label_rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> None:
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_channel_csv = args.out_channel_csv or f"{args.prefix}_channel.csv"
    args.out_spin_csv = args.out_spin_csv or f"{args.prefix}_spin.csv"
    args.out_label_csv = args.out_label_csv or f"{args.prefix}_label.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"step": "10e", "fitN": args.fitN, "summaries": summaries, "channel_rows": channel_rows}, f, indent=2)
    for path, rows in [(args.out_channel_csv, channel_rows), (args.out_spin_csv, spin_rows), (args.out_label_csv, label_rows)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "=" * 120,
        "Step 10e | Be pair population and R12 V-channel audit",
        "=" * 120,
        f"fitN = {args.fitN}",
        "",
        f"{'case':<15s} {'V_total/mEh':>12s} {'V_s-s':>12s} {'V_s-p':>12s} {'V_p-p':>12s} {'ab_frac':>9s} {'ss_pair_frac':>12s}",
        "-" * 120,
    ]
    for s in summaries:
        lines.append(
            f"{s['case']:<15s} {s['V_total_mEh']:12.6f} {s.get('V_s-s_mEh', 0.0):12.6f} "
            f"{s.get('V_p-s_mEh', 0.0):12.6f} {s.get('V_p-p_mEh', 0.0):12.6f} "
            f"{s.get('ab_pair_fraction', 0.0):9.6f} {s.get('s-s_pair_fraction', 0.0):12.6f}"
        )
    lines.extend(
        [
            "",
            "[Reading]",
            "For closed-shell Be, spin pair fractions should approach aa:ab:bb = 1/6:4/6:1/6.",
            "Compare V_s-s, V_s-p, and V_p-p to see whether Be is still radial s-s dominated or p-screened.",
            "",
            "[Saved]",
            f"  {args.out_json}",
            f"  {args.out_channel_csv}",
            f"  {args.out_spin_csv}",
            f"  {args.out_label_csv}",
            f"  {args.summary}",
        ]
    )
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    channel_rows: List[Dict[str, Any]] = []
    spin_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for case_name in parse_cases(args.cases):
        case = load_case(case_name, args.fitN)
        pop_rows, sp_rows = pair_population_rows(case)
        v_rows, v_label_rows = v_channel_rows(case)
        channel_rows.extend([r for r in pop_rows if "pair_weight" in r])
        channel_rows.extend(v_rows)
        spin_rows.extend(sp_rows)
        label_rows.extend([r for r in pop_rows if r.get("kind") == "pair_population"])
        label_rows.extend(v_label_rows[:24])

        total_v = sum(float(r["V_mEh"]) for r in v_rows)
        pop_channel = [r for r in pop_rows if "pair_weight" in r and r.get("spin_pair") != ""]
        summary = {"case": case_name, "V_total_mEh": total_v}
        for r in v_rows:
            summary[f"V_{r['channel_pair']}_mEh"] = float(r["V_mEh"])
        for r in sp_rows:
            summary[f"{r['spin_pair']}_pair_fraction"] = float(r["pair_fraction"])
        s_s_weight = sum(float(r["pair_weight"]) for r in pop_channel if r["channel_pair"] == "s-s")
        all_weight = sum(float(r["pair_weight"]) for r in pop_channel)
        summary["s-s_pair_fraction"] = s_s_weight / all_weight if all_weight else 0.0
        summary["delta_E_r12_mEh"] = 1000.0 * float(case["corr"]["delta_E_r12"])
        summaries.append(summary)
    write_outputs(args, channel_rows, spin_rows, label_rows, summaries)


if __name__ == "__main__":
    main()
