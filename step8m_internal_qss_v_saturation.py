#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 8m: Internal q_ss from computable s-channel V saturation.

Candidate diagnostic requested by the user:

    q_ss = sqrt(||V_ss[CABS_s]|| / ||V_ss[OBS_s + CABS_s]||)

The intent is to replace reference-space fitting by a current-space quantity.
Here V_ss is represented by the same-spin F12/Coulomb coupling vector from an
active s-s pair into same-channel RI pair blocks.  CABS_s is identified by the
dominant AO-channel weight of each passive RI vector.

This script is an audit/candidate scan.  It does not yet change production R12.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import r12_correction as rc
from step7b_export_ecg_no_orbitals import L_INFO, build_ao_index
from step8c_hem_triplet_r12_correction import same_spin_pair_fci_target
from step8e_audit_hem_same_spin_failure_source import alpha_only_fock, load_case
from step8f_hem_pauli_suppressed_geminal_audit import occupation_descriptors, pair_descriptors
from step8l_channel_resolved_same_spin_candidates import channel, channel_tensor, patched_tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--prefix", default="step8m_hem_internal_qss_v_saturation")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def cases() -> List[str]:
    return ["sp_s01_p01", "sp_s0123_p01", "sp_s01_p0123", "sp_s0123_p0123"]


def paths(case: str, fitn: int) -> Dict[str, str]:
    return {
        "rdm_npz": f"step8h_hem_triplet_{case}_rdm_export.npz",
        "bridge_npz": f"step8b_hem_triplet_{case}_fitN{fitn}_step4b_like.npz",
    }


def ao_channel_indices(channels: List[str], nao: int) -> Dict[str, np.ndarray]:
    # Existing HEM parent uses ns=16 and spherical s/p shells; infer ns from nao.
    deg_sum = sum(L_INFO[ch]["deg"] for ch in channels)
    ns = nao // deg_sum
    if ns * deg_sum != nao:
        raise ValueError(f"Cannot infer ns from nao={nao}, channels={channels}")
    return {ch: np.array(build_ao_index(ns, channels)[ch]["all"], dtype=int) for ch in channels}


def ri_channel_weights(S_ao: np.ndarray, C_ri: np.ndarray, channels: List[str]) -> np.ndarray:
    idx = ao_channel_indices(channels, S_ao.shape[0])
    W = np.zeros((C_ri.shape[1], len(channels)), dtype=float)
    for j in range(C_ri.shape[1]):
        c = C_ri[:, j]
        denom = float(c.T @ S_ao @ c)
        for k, ch in enumerate(channels):
            cp = np.zeros_like(c)
            cp[idx[ch]] = c[idx[ch]]
            W[j, k] = max(0.0, float(cp.T @ S_ao @ cp) / (denom + 1e-300))
        s = float(np.sum(W[j]))
        if s > 1e-14:
            W[j] /= s
    return W


def ri_channel_sets(bridge_npz: str) -> Dict[str, Any]:
    data = np.load(bridge_npz, allow_pickle=True)
    labels = [str(x) for x in data["labels"]]
    channels = [str(x) for x in data["channels"]]
    nobs = len(labels)
    nri = np.array(data["C_ri"]).shape[1]
    active_by_channel: Dict[str, List[int]] = {ch: [] for ch in channels}
    for i, lab in enumerate(labels):
        ch = channel(lab)
        active_by_channel.setdefault(ch, []).append(i)

    W = ri_channel_weights(np.array(data["S_ao"], dtype=float), np.array(data["C_ri"], dtype=float), channels)
    dom = [channels[int(np.argmax(W[j]))] for j in range(nri)]
    passive_by_channel: Dict[str, List[int]] = {ch: [] for ch in channels}
    for j in range(nobs, nri):
        passive_by_channel.setdefault(dom[j], []).append(j)
    ri_by_channel = {ch: sorted(active_by_channel.get(ch, []) + passive_by_channel.get(ch, [])) for ch in channels}
    return {
        "labels": labels,
        "channels": channels,
        "nobs": nobs,
        "nri": nri,
        "dominant_channels": dom,
        "channel_weights": W,
        "active_by_channel": active_by_channel,
        "passive_by_channel": passive_by_channel,
        "ri_by_channel": ri_by_channel,
    }


def pair_block_norm(T: np.ndarray, active_s: List[int], pair_set: List[int]) -> float:
    if not active_s or not pair_set:
        return 0.0
    block = T[np.ix_(active_s, active_s, pair_set, pair_set)]
    anti = block - block.transpose(1, 0, 2, 3)
    return float(np.linalg.norm(anti.reshape(-1)))


def qss_internal(bridge_npz: str) -> Dict[str, Any]:
    sets = ri_channel_sets(bridge_npz)
    data = np.load(bridge_npz, allow_pickle=True)
    g_phys = rc.chem_to_phys(np.array(data["eri_ri"], dtype=float))
    r_phys = rc.chem_to_phys(np.array(data["f12_ri"], dtype=float))
    active_s = sets["active_by_channel"].get("s", [])
    cabs_s = sets["passive_by_channel"].get("s", [])
    obs_s = sets["active_by_channel"].get("s", [])
    obs_plus_cabs_s = sets["ri_by_channel"].get("s", [])

    # Coupling vector from active s-s pair to RI pair blocks.  The antisymmetric
    # combination matches same-spin pair character.
    T = g_phys * r_phys
    num = pair_block_norm(T, active_s, cabs_s)
    obs_norm = pair_block_norm(T, active_s, obs_s)
    den = pair_block_norm(T, active_s, obs_plus_cabs_s)
    q = float(np.sqrt(num / den)) if den > 0.0 else 0.0
    q_norm = float(1.0 / np.sqrt(1.0 + obs_norm / num)) if num > 0.0 else 0.0
    return {
        "q_ss_internal": q,
        "q_ss_normalized": q_norm,
        "norm_cabs_s": num,
        "norm_obs_s": obs_norm,
        "norm_obs_plus_cabs_s": den,
        "n_active_s": len(active_s),
        "n_cabs_s": len(cabs_s),
        "n_obs_plus_cabs_s": len(obs_plus_cabs_s),
        "channel_sets": {
            "active_by_channel": sets["active_by_channel"],
            "passive_by_channel": sets["passive_by_channel"],
            "ri_by_channel": sets["ri_by_channel"],
        },
    }


def compute_candidate(case_data: Dict[str, Any], labels: List[str], scale: Dict[str, float]) -> Dict[str, float]:
    fock = alpha_only_fock(case_data["h"], case_data["g_phys"], case_data["dm1"], case_data["nobs"], case_data["nri"])
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


def add_row(rows: List[Dict[str, Any]], case: str, model: str, scale: Dict[str, float], case_data, labels, target, qinfo, desc, pair) -> None:
    comp = compute_candidate(case_data, labels, scale)
    corr = 1000.0 * comp["correction"]
    gap = float(target["full_parent_gap_mEh"])
    rows.append(
        {
            "case": case,
            "nobs": case_data["nobs"],
            "model": model,
            "q_ss_internal": qinfo["q_ss_internal"],
            "q_ss_normalized": qinfo["q_ss_normalized"],
            "scale_ss": scale.get("s-s", 0.0),
            "scale_sp": scale.get("p-s", 0.0),
            "scale_pp": scale.get("p-p", 0.0),
            "tail_sqrt": desc["sqrt_tail_occupation_sum_after_two"],
            "pair_sqrt": pair["sqrt_residual_pair_weight"],
            "n_active_s": qinfo["n_active_s"],
            "n_cabs_s": qinfo["n_cabs_s"],
            "norm_cabs_s": qinfo["norm_cabs_s"],
            "norm_obs_s": qinfo["norm_obs_s"],
            "norm_obs_plus_cabs_s": qinfo["norm_obs_plus_cabs_s"],
            "target_gap_mEh": gap,
            "correction_mEh": corr,
            "residual_mEh": corr - gap,
            "V_mEh": 1000.0 * comp["V"],
            "B_mEh": 1000.0 * comp["B"],
            "X_mEh": 1000.0 * comp["X"],
            "Delta_mEh": 1000.0 * comp["Delta"],
        }
    )


def build_rows(fitn: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases():
        p = paths(case, fitn)
        if not Path(p["bridge_npz"]).exists() or not Path(p["rdm_npz"]).exists():
            raise FileNotFoundError(f"Missing Step8i prerequisites for {case}; run make step8i")
        data = load_case(p["bridge_npz"])
        bridge = np.load(p["bridge_npz"], allow_pickle=True)
        labels = [str(x) for x in bridge["labels"]]
        target = same_spin_pair_fci_target(p["bridge_npz"])
        desc = occupation_descriptors(data["dm1"])
        pair = pair_descriptors(p["rdm_npz"])
        qinfo = qss_internal(p["bridge_npz"])
        tail2 = 2.0 * desc["sqrt_tail_occupation_sum_after_two"]
        pair3 = 3.0 * pair["sqrt_residual_pair_weight"]
        q = qinfo["q_ss_internal"]
        add_row(rows, case, "tail2_ss_only", {"s-s": tail2}, data, labels, target, qinfo, desc, pair)
        add_row(rows, case, "internal_q_tail2_ss_only", {"s-s": q * tail2}, data, labels, target, qinfo, desc, pair)
        add_row(rows, case, "normalized_q_tail2_ss_only", {"s-s": qinfo["q_ss_normalized"] * tail2}, data, labels, target, qinfo, desc, pair)
        add_row(rows, case, "internal_q_pair3_ss_only", {"s-s": q * pair3}, data, labels, target, qinfo, desc, pair)
        add_row(rows, case, "internal_q_tail2_ss_sp", {"s-s": q * tail2, "p-s": q * tail2}, data, labels, target, qinfo, desc, pair)
        add_row(rows, case, "normalized_q_tail2_ss_sp", {"s-s": qinfo["q_ss_normalized"] * tail2, "p-s": qinfo["q_ss_normalized"] * tail2}, data, labels, target, qinfo, desc, pair)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        abs_res = np.array([abs(float(r["residual_mEh"])) for r in sub], dtype=float)
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
        json.dump({"step": "8m", "fitN": args.fitN, "model_summary": summary, "rows": rows}, f, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=" * 112,
        "Step 8m | HEM internal q_ss V-saturation candidate",
        "=" * 112,
        f"fitN = {args.fitN}",
        "",
        "[Model residual summary]",
    ]
    for model, vals in summary.items():
        lines.append(f"{model:<32s} mean_abs={vals['mean_abs_residual_mEh']:.6f} mEh  max_abs={vals['max_abs_residual_mEh']:.6f} mEh")
    lines.extend(
        [
            "",
            "[Rows]",
            f"{'case':<18s} {'model':<30s} {'q':>8s} {'qnorm':>8s} {'ss':>9s} {'gap':>11s} {'dE':>11s} {'resid':>11s} {'nCs':>5s}",
            "-" * 112,
        ]
    )
    for r in rows:
        lines.append(
            f"{r['case']:<18s} {r['model']:<30s} {float(r['q_ss_internal']):8.5f} {float(r['q_ss_normalized']):8.5f} {float(r['scale_ss']):9.5f} "
            f"{float(r['target_gap_mEh']):11.6f} {float(r['correction_mEh']):11.6f} {float(r['residual_mEh']):11.6f} "
            f"{int(r['n_cabs_s']):5d}"
        )
    lines.extend(
        [
            "",
            "[Interpretation]",
            "This is the requested fully computable q_ss.  If it improves over tail2_ss_only",
            "without using the parent target, it is a candidate rule.  If it damps too much",
            "or too little, the internal V norm needs a more precise channel/block definition.",
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
