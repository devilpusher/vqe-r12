#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 11: cross-system summary from existing He/HEM/Li/Be audit files.

This script is read-only with respect to expensive data products: it does not
run FCI, Psi4, or R12 contractions.  It only reads existing JSON/CSV outputs and
builds summary tables plus article-facing notes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--he-scan", default="step7e_ecg_no_r12_scan_full_sp_spd_fitN3579.json")
    p.add_argument("--he-fallback", default="step7e_ecg_no_r12_scan.json")
    p.add_argument("--hem-candidate", default="step8p_hem_same_spin_candidate_stress_fitN7.json")
    p.add_argument("--hem-physical", default="step8q_hem_physical_q_law_audit_stress_fitN7.json")
    p.add_argument("--li-scan", default="step9d_li_selected_space_scan_fitN7.json")
    p.add_argument("--li-channel", default="step9e_li_pair_channel_audit_fitN7.json")
    p.add_argument("--be-scan", default="step10d_be_selected_space_scan_fitN7.json")
    p.add_argument("--be-channel", default="step10e_be_pair_channel_audit_fitN7.json")
    p.add_argument("--prefix", default="step11_cross_system_summary")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-systems-csv", default=None)
    p.add_argument("--out-spaces-csv", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def load_json(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fnum(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def fmt(x: Any, nd: int = 6) -> str:
    y = fnum(x)
    if y is None:
        return ""
    return f"{y:.{nd}f}"


def pick_best_abs_residual(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ok = [r for r in rows if r.get("status", "ok") == "ok"]
    if not ok:
        return None
    return min(ok, key=lambda r: abs(float(r.get("residual_mEh", r.get("abs_residual_to_reference_mEh", 1e99)))))


def pick_case(rows: List[Dict[str, Any]], case: str, fitn: int = 7) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r.get("case") == case and int(r.get("fitN", fitn)) == fitn and r.get("status", "ok") == "ok":
            return r
    return None


def channel_summary(channel_payload: Optional[Dict[str, Any]], preferred_case: str) -> Dict[str, Any]:
    if not channel_payload:
        return {}
    rows = channel_payload.get("summaries", [])
    chosen = next((r for r in rows if r.get("case") == preferred_case), None)
    if chosen is None and rows:
        chosen = rows[-1]
    return chosen or {}


def he_rows(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not payload:
        return []
    return [r for r in payload.get("rows", []) if r.get("status", "ok") == "ok"]


def normalize_scan_row(system: str, row: Dict[str, Any], reference_label: str = "") -> Dict[str, Any]:
    residual = fnum(row.get("residual_mEh"))
    if residual is None:
        residual = fnum(row.get("abs_residual_to_reference_mEh"))
    recovery = fnum(row.get("recovery_ratio"))
    if recovery is None:
        recovery = fnum(row.get("recovery_vs_reference"))
    return {
        "system": system,
        "case": row.get("case", ""),
        "fitN": row.get("fitN", ""),
        "nobs": row.get("nobs", ""),
        "nqubits": row.get("nqubits", ""),
        "E_obs_fci": row.get("E_obs_fci", ""),
        "delta_E_r12_mEh": fnum(row.get("delta_mEh")) if "delta_mEh" in row else 1000.0 * float(row.get("delta_E_r12", 0.0)),
        "E_total": row.get("E_total", ""),
        "reference_energy": row.get("reference_energy", ""),
        "recovery_ratio": recovery,
        "residual_mEh": residual,
        "V_mEh": fnum(row.get("V_mEh")) if "V_mEh" in row else 1000.0 * float(row.get("V", 0.0)),
        "B_mEh": fnum(row.get("B_mEh")) if "B_mEh" in row else 1000.0 * float(row.get("B", 0.0)),
        "X_mEh": fnum(row.get("X_mEh")) if "X_mEh" in row else 1000.0 * float(row.get("X", 0.0)),
        "Delta_mEh": fnum(row.get("Delta_mEh")) if "Delta_mEh" in row else 1000.0 * float(row.get("Delta", 0.0)),
        "reference_label": row.get("reference_label", reference_label),
    }


def hem_system_row(candidate: Optional[Dict[str, Any]], physical: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    model = "hem_q_shape_worst_case_ss_only"
    summary = ((candidate or {}).get("model_summary") or {}).get(model, {})
    params = (candidate or {}).get("hem_worst_case_params", {})
    branch_summary = (physical or {}).get("branch_summary", {})
    return {
        "system": "HEM",
        "state": "metastable He triplet, same-spin limit",
        "representative_space": "stress scan, multiple s/p spaces",
        "method_or_rule": model,
        "main_physics": "Pauli-suppressed same-spin correction; s growth boosts, p expansion screens, mixed s/p cancels.",
        "delta_E_r12_mEh": "",
        "recovery_ratio": "",
        "residual_mEh": summary.get("mean_abs_residual_mEh", ""),
        "max_abs_residual_mEh": summary.get("max_abs_residual_mEh", ""),
        "V_s_s_mEh": "",
        "V_s_p_mEh": "",
        "V_p_p_mEh": "",
        "s_s_pair_fraction": "",
        "ab_pair_fraction": "",
        "notes": f"q=(ns/2)^0.80*max((2/np)^0.35,0.76), cross cancel for mixed growth; branches={list(branch_summary.keys())}",
        "source": "step8p/step8q",
    }


def build(args) -> Dict[str, Any]:
    he = load_json(args.he_scan) or load_json(args.he_fallback)
    hem = load_json(args.hem_candidate)
    hem_phys = load_json(args.hem_physical)
    li = load_json(args.li_scan)
    li_ch = load_json(args.li_channel)
    be = load_json(args.be_scan)
    be_ch = load_json(args.be_channel)

    selected_rows: List[Dict[str, Any]] = []
    he_scan_rows = he_rows(he)
    for r in he_scan_rows:
        selected_rows.append(normalize_scan_row("He", r))
    for r in (li or {}).get("rows", []):
        selected_rows.append(normalize_scan_row("Li", r, (li or {}).get("reference_label", "")))
    for r in (be or {}).get("rows", []):
        selected_rows.append(normalize_scan_row("Be", r, (be or {}).get("reference_label", "")))

    he_rep = pick_case(he_scan_rows, "sp_s012_p0", 7) or pick_best_abs_residual(he_scan_rows)
    li_best = pick_best_abs_residual((li or {}).get("rows", []))
    be_best = pick_best_abs_residual((be or {}).get("rows", []))
    li_ch_best = channel_summary(li_ch, (li_best or {}).get("case", "sp_s012_p01"))
    be_ch_best = channel_summary(be_ch, (be_best or {}).get("case", "sp_s012_p01"))

    system_rows: List[Dict[str, Any]] = []
    if he_rep:
        h = normalize_scan_row("He", he_rep)
        system_rows.append(
            {
                "system": "He",
                "state": "closed-shell two-electron singlet",
                "representative_space": he_rep.get("case", ""),
                "method_or_rule": "paper_tequila_sf2r12",
                "main_physics": "opposite-spin short-range cusp/radial recovery; p expansion can over-screen in larger spaces.",
                "delta_E_r12_mEh": h["delta_E_r12_mEh"],
                "recovery_ratio": h["recovery_ratio"],
                "residual_mEh": h["residual_mEh"],
                "max_abs_residual_mEh": "",
                "V_s_s_mEh": "",
                "V_s_p_mEh": "",
                "V_p_p_mEh": "",
                "s_s_pair_fraction": "",
                "ab_pair_fraction": "1.0",
                "notes": "Clean two-electron sanity check for spin-free R12 conventions.",
                "source": args.he_scan if Path(args.he_scan).exists() else args.he_fallback,
            }
        )
    system_rows.append(hem_system_row(hem, hem_phys))
    if li_best:
        l = normalize_scan_row("Li", li_best, (li or {}).get("reference_label", ""))
        system_rows.append(
            {
                "system": "Li",
                "state": "open-shell doublet, 3 electrons",
                "representative_space": li_best.get("case", ""),
                "method_or_rule": "paper_tequila_sf2r12 audit",
                "main_physics": "s-s radial/core recovery dominates; p-space is weak angular screening.",
                "delta_E_r12_mEh": l["delta_E_r12_mEh"],
                "recovery_ratio": l["recovery_ratio"],
                "residual_mEh": l["residual_mEh"],
                "max_abs_residual_mEh": "",
                "V_s_s_mEh": li_ch_best.get("V_s-s_mEh", ""),
                "V_s_p_mEh": li_ch_best.get("V_p-s_mEh", ""),
                "V_p_p_mEh": li_ch_best.get("V_p-p_mEh", ""),
                "s_s_pair_fraction": li_ch_best.get("s-s_pair_fraction", ""),
                "ab_pair_fraction": li_ch_best.get("ab_pair_fraction", ""),
                "notes": "Adding third s radial NO sharply reduces R12 magnitude; Li is open-shell but not HEM-like same-spin dominated.",
                "source": "step9d/step9e",
            }
        )
    if be_best:
        b = normalize_scan_row("Be", be_best, (be or {}).get("reference_label", ""))
        system_rows.append(
            {
                "system": "Be",
                "state": "closed-shell four-electron singlet",
                "representative_space": be_best.get("case", ""),
                "method_or_rule": "paper_tequila_sf2r12 audit",
                "main_physics": "large s-s radial/core recovery with materially stronger p-shell screening than Li.",
                "delta_E_r12_mEh": b["delta_E_r12_mEh"],
                "recovery_ratio": b["recovery_ratio"],
                "residual_mEh": b["residual_mEh"],
                "max_abs_residual_mEh": "",
                "V_s_s_mEh": be_ch_best.get("V_s-s_mEh", ""),
                "V_s_p_mEh": be_ch_best.get("V_p-s_mEh", ""),
                "V_p_p_mEh": be_ch_best.get("V_p-p_mEh", ""),
                "s_s_pair_fraction": be_ch_best.get("s-s_pair_fraction", ""),
                "ab_pair_fraction": be_ch_best.get("ab_pair_fraction", ""),
                "notes": "Early exnot13 NO data; best residual in requested scan is p-extended s012+p01.",
                "source": "step10d/step10e",
            }
        )

    return {
        "step": "11",
        "inputs": vars(args),
        "system_rows": system_rows,
        "selected_space_rows": selected_rows,
        "notes": notes_text(system_rows, selected_rows),
    }


def notes_text(system_rows: List[Dict[str, Any]], selected_rows: List[Dict[str, Any]]) -> str:
    by_system = {r["system"]: r for r in system_rows}
    lines = [
        "# Cross-System ECG-NO + R12 Summary",
        "",
        "## Main Conclusion",
        "",
        "Across He, HEM, Li, and Be, the R12 correction behaves as a physically interpretable complement to compact ECG-NO selected spaces rather than as an arbitrary numerical shift.",
        "The common pattern is short-range radial/cusp recovery in opposite-spin or spin-free s-s channels, with p-space acting as angular screening. The metastable He triplet isolates the exceptional same-spin limit, where Pauli suppression must be treated explicitly.",
        "",
        "## System-Level Reading",
        "",
    ]
    for key in ["He", "HEM", "Li", "Be"]:
        r = by_system.get(key)
        if not r:
            lines.append(f"- **{key}**: missing input data.")
            continue
        bits = [f"- **{key}** ({r['state']}): {r['main_physics']}"]
        if r.get("representative_space"):
            bits.append(f"  Representative space: `{r['representative_space']}`.")
        if r.get("delta_E_r12_mEh") != "":
            bits.append(
                f"  R12 correction: {fmt(r['delta_E_r12_mEh'])} mEh; "
                f"recovery: {fmt(r['recovery_ratio'])}; residual: {fmt(r['residual_mEh'])} mEh."
            )
        if r.get("max_abs_residual_mEh") != "":
            bits.append(
                f"  Conservative candidate mean/max residual: {fmt(r['residual_mEh'])}/"
                f"{fmt(r['max_abs_residual_mEh'])} mEh."
            )
        if r.get("V_s_s_mEh") != "":
            bits.append(
                f"  V-channel: s-s={fmt(r['V_s_s_mEh'])} mEh, "
                f"s-p={fmt(r['V_s_p_mEh'])} mEh, p-p={fmt(r['V_p_p_mEh'])} mEh; "
                f"s-s pair fraction={fmt(r['s_s_pair_fraction'])}."
            )
        lines.extend(bits)
        lines.append("")
    lines.extend(
        [
            "## Article-Ready Statement",
            "",
            "The ECG-NO selected-space R12 correction is consistent across the tested systems with a channel-resolved physical picture: closed-shell He validates the opposite-spin cusp recovery limit; metastable He demonstrates the need for explicit Pauli-suppressed same-spin damping; Li retains an s-s radial/core recovery character despite being open-shell; and Be shows the same closed-shell s-s recovery with stronger p-shell angular screening.",
            "",
            "## Remaining Tests",
            "",
            "1. Run fitN=5/7/9 stability for the key Li and Be representative spaces if a final quantitative claim is needed.",
            "2. Treat Be as an early-data audit until the optimized exnot calculation is available.",
            "3. Keep HEM's conservative `hem_q_shape_worst_case_ss_only` rule separate from the opposite-spin dominated He/Li/Be interpretation.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args, payload: Dict[str, Any]) -> None:
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_systems_csv = args.out_systems_csv or f"{args.prefix}_systems.csv"
    args.out_spaces_csv = args.out_spaces_csv or f"{args.prefix}_selected_spaces.csv"
    args.notes = args.notes or f"{args.prefix}_notes.md"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(args.out_systems_csv, payload["system_rows"])
    write_csv(args.out_spaces_csv, payload["selected_space_rows"])
    with open(args.notes, "w", encoding="utf-8") as f:
        f.write(payload["notes"])
    lines = [
        "=" * 112,
        "Step 11 | Cross-system ECG-NO + R12 summary",
        "=" * 112,
        "",
        "[System Rows]",
        f"{'system':<6s} {'space':<16s} {'dR12/mEh':>12s} {'recovery':>10s} {'resid/mEh':>11s} {'main physics'}",
        "-" * 112,
    ]
    for r in payload["system_rows"]:
        lines.append(
            f"{r['system']:<6s} {str(r.get('representative_space','')):<16s} "
            f"{fmt(r.get('delta_E_r12_mEh')):>12s} {fmt(r.get('recovery_ratio')):>10s} "
            f"{fmt(r.get('residual_mEh')):>11s} {r.get('main_physics','')}"
        )
    lines.extend(
        [
            "",
            "[Saved]",
            f"  {args.out_json}",
            f"  {args.out_systems_csv}",
            f"  {args.out_spaces_csv}",
            f"  {args.notes}",
            f"  {args.summary}",
        ]
    )
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    payload = build(args)
    write_outputs(args, payload)


if __name__ == "__main__":
    main()
