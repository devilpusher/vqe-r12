#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7g: Subterm audit for ECG-NO SF-[2]R12 projector cancellation.

This script reads existing Step7c fitN=7 r12-only files and decomposes the
paper/Tequila SF-[2]R12 contraction into signed subterms.  The goal is to locate
why the CABS-only correction changes from a stable negative value in
s[0,1,2]+p[0] to near-zero or small-positive values in larger ECG-NO spaces.

The low-occupation variants are diagnostics only.  They remove low-occupation
OBS orbitals from the active summation and add them to the passive list, then
report the active RDM trace so the truncation is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from r12_common import build_sp_tensor, reconstruct_energy, rdm_diagnostics
from r12_correction import block2, block4, build_fock_tequila, chem_to_phys, load_metadata, metadata_energy
from step7f_audit_ecg_no_r12_projectors import DEFAULT_INPUTS, case_label


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--nocc", type=int, default=1)
    p.add_argument("--low-occ-thresholds", default="1e-4,5e-4,1e-3")
    p.add_argument("--out-json", default="step7g_ecg_no_r12_subterm_audit.json")
    p.add_argument("--out-csv", default="step7g_ecg_no_r12_subterm_audit.csv")
    p.add_argument("--diff-csv", default="step7g_ecg_no_r12_subterm_diffs.csv")
    p.add_argument("--summary", default="step7g_ecg_no_r12_subterm_audit_summary.txt")
    return p.parse_args()


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_case(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    meta = load_metadata(data)
    nobs = int(meta["nobs"])
    nri = int(meta["nri"])
    h_ri = np.array(data["h_ri"], dtype=float)
    eri_ri = np.array(data["eri_ri"], dtype=float)
    f12_ri = np.array(data["f12_ri"], dtype=float)
    dm1_obs = np.array(data["dm1_obs"], dtype=float)
    dm2_obs = np.array(data["dm2_obs"], dtype=float)
    dm1_ri = np.array(data["dm1_ri"], dtype=float)
    dm2_ri = np.array(data["dm2_ri"], dtype=float)
    enuc = float(metadata_energy(meta, "enuc", 0.0) or 0.0)
    E_obs = float(metadata_energy(meta, "E_obs_fci"))
    E_obs_rdm, _, _ = reconstruct_energy(h_ri[:nobs, :nobs], eri_ri[:nobs, :nobs, :nobs, :nobs], dm1_obs, dm2_obs, enuc)
    E_ri_rdm, _, _ = reconstruct_energy(h_ri, eri_ri, dm1_ri, dm2_ri, enuc)
    g_phys = chem_to_phys(eri_ri)
    r_phys = chem_to_phys(f12_ri)
    fock = build_fock_tequila(h_ri, g_phys, dm1_obs, list(range(nobs)), list(range(nri)))
    diag = rdm_diagnostics(dm1_obs, dm2_obs)
    return {
        "path": path,
        "label": case_label(path),
        "nobs": nobs,
        "nri": nri,
        "h_ri": h_ri,
        "g_phys": g_phys,
        "r_phys": r_phys,
        "fock": fock,
        "dm1_obs": dm1_obs,
        "dm2_obs": dm2_obs,
        "E_obs": E_obs,
        "energy_checks": {
            "delta_obs_rdm_minus_fci": E_obs_rdm - E_obs,
            "delta_ri_rdm_minus_fci": E_ri_rdm - E_obs,
        },
        "natural_occupations": diag["natural_occupations"],
    }


def local_rdms(dm1_obs: np.ndarray, dm2_obs: np.ndarray, active: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    ix = np.ix_(active, active)
    dm1 = dm1_obs[ix]
    dm2 = dm2_obs[np.ix_(active, active, active, active)]
    return dm1, dm2


def signed_subterms(
    g_phys: np.ndarray,
    r_phys: np.ndarray,
    fock: np.ndarray,
    dm1_obs: np.ndarray,
    dm2_obs: np.ndarray,
    active: List[int],
    passive: List[int],
    nri: int,
) -> Dict[str, float]:
    """Return signed subterms whose group sums reproduce V/B/X/Delta."""
    a = active
    p = passive
    f = list(range(nri))
    dm1, dm2 = local_rdms(dm1_obs, dm2_obs, a)
    t = build_sp_tensor(len(a))

    sub: Dict[str, float] = {}

    gKLxy_rRSkl = np.einsum("klxy,rskl->rsxy", block4(g_phys, f, f, a, a), block4(r_phys, a, a, f, f), optimize=True)
    gTUxy_rRStu = np.einsum("tuxy,rstu->rsxy", block4(g_phys, a, a, a, a), block4(r_phys, a, a, a, a), optimize=True)
    gATxy_rdm1Ut_rRSau = np.einsum("atxy,ut,rsau->rsxy", block4(g_phys, p, a, a, a), dm1, block4(r_phys, a, a, p, a), optimize=True)
    sub["V_KL_full"] = float(np.einsum("pqrs,xypq,rsxy", t, dm2, gKLxy_rRSkl, optimize=True))
    sub["V_TU_obs_subtraction"] = float(-np.einsum("pqrs,xypq,rsxy", t, dm2, gTUxy_rRStu, optimize=True))
    sub["V_AT_passive_subtraction"] = float(-np.einsum("pqrs,xypq,rsxy", t, dm2, gATxy_rdm1Ut_rRSau, optimize=True))

    rZYpq_fockXy_rTUzx = np.einsum("zypq,xy,tuzx->tupq", block4(r_phys, a, a, a, a), block2(fock, a, a), block4(r_phys, a, a, a, a), optimize=True)
    rAYpq_fockXa_rTUxy = np.einsum("aypq,xa,tuxy->tupq", block4(r_phys, p, a, a, a), block2(fock, a, p), block4(r_phys, a, a, a, a), optimize=True)
    rYXpq_fockAx_rTUya = np.einsum("yxpq,ax,tuya->tupq", block4(r_phys, a, a, a, a), block2(fock, p, a), block4(r_phys, a, a, a, p), optimize=True)
    rMLpq_fockKl_rTUmk = np.einsum("mlpq,kl,tumk->tupq", block4(r_phys, f, f, a, a), block2(fock, f, f), block4(r_phys, a, a, f, f), optimize=True)
    rBYpq_rdm1Xy_fockAb_rTUax = np.einsum("bypq,xy,ab,tuax->tupq", block4(r_phys, p, a, a, a), dm1, block2(fock, p, p), block4(r_phys, a, a, p, a), optimize=True)
    rAYpq_rdm1Xy_fockKx_rTUak = np.einsum("aypq,xy,kx,tuak->tupq", block4(r_phys, p, a, a, a), dm1, block2(fock, f, a), block4(r_phys, a, a, p, f), optimize=True)
    sub["B_ML_full"] = float(np.einsum("pqrs,vwtu,rsvw,tupq", t, t, dm2, rMLpq_fockKl_rTUmk, optimize=True))
    sub["B_ZY_obs_subtraction"] = float(-np.einsum("pqrs,vwtu,rsvw,tupq", t, t, dm2, rZYpq_fockXy_rTUzx, optimize=True))
    sub["B_AY_passive_subtraction"] = float(-np.einsum("pqrs,vwtu,rsvw,tupq", t, t, dm2, rAYpq_fockXa_rTUxy, optimize=True))
    sub["B_YX_passive_subtraction"] = float(-np.einsum("pqrs,vwtu,rsvw,tupq", t, t, dm2, rYXpq_fockAx_rTUya, optimize=True))
    sub["B_BY_half_subtraction"] = float(-0.5 * np.einsum("pqrs,vwtu,rsvw,tupq", t, t, dm2, rBYpq_rdm1Xy_fockAb_rTUax, optimize=True))
    sub["B_AK_half_subtraction"] = float(-0.5 * np.einsum("pqrs,vwtu,rsvw,tupq", t, t, dm2, rAYpq_rdm1Xy_fockKx_rTUak, optimize=True))

    rTUkl_rKLpq = np.einsum("tukl,klpq->tupq", block4(r_phys, a, a, f, f), block4(r_phys, f, f, a, a), optimize=True)
    rTUyz_rYZpq = np.einsum("tuyz,yzpq->tupq", block4(r_phys, a, a, a, a), block4(r_phys, a, a, a, a), optimize=True)
    rUTya_rdm1Yz_rAZpq = np.einsum("utya,yz,azpq->tupq", block4(r_phys, a, a, a, p), dm1, block4(r_phys, p, a, a, a), optimize=True)
    rTUay_rdm1Yz_rAZqp = np.einsum("tuay,yz,azqp->tupq", block4(r_phys, a, a, p, a), dm1, block4(r_phys, p, a, a, a), optimize=True)
    # X has an overall minus sign in the audited formula.
    sub["X_KL_full_after_energy_sign"] = float(-np.einsum("pqrs,vwtu,rsvx,xw,tupq", t, t, dm2, block2(fock, a, a), rTUkl_rKLpq, optimize=True))
    sub["X_YZ_obs_subtraction_after_energy_sign"] = float(+np.einsum("pqrs,vwtu,rsvx,xw,tupq", t, t, dm2, block2(fock, a, a), rTUyz_rYZpq, optimize=True))
    sub["X_UTya_half_subtraction_after_energy_sign"] = float(+0.5 * np.einsum("pqrs,vwtu,rsvx,xw,tupq", t, t, dm2, block2(fock, a, a), rUTya_rdm1Yz_rAZpq, optimize=True))
    sub["X_TUay_half_subtraction_after_energy_sign"] = float(+0.5 * np.einsum("pqrs,vwtu,rsvx,xw,tupq", t, t, dm2, block2(fock, a, a), rTUay_rdm1Yz_rAZqp, optimize=True))

    delta_terms = {
        "Delta1_a": -0.5 * np.einsum("pqrs,aypq,vwtu,xrvy,kx,sw,utak", t, block4(r_phys, p, a, a, a), t, dm2, block2(fock, f, a), dm1, block4(r_phys, a, a, p, f), optimize=True),
        "Delta1_b": -0.5 * np.einsum("pqrs,aypq,vwtu,xryv,kx,sw,tuak", t, block4(r_phys, p, a, a, a), t, dm2, block2(fock, f, a), dm1, block4(r_phys, a, a, p, f), optimize=True),
        "Delta1_c": -0.5 * np.einsum("pqrs,aypq,vwtu,kx,rv,sw,xy,utak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), dm1, dm1, dm1, block4(r_phys, a, a, p, f), optimize=True),
        "Delta1_d": np.einsum("pqrs,aypq,vwtu,kx,rv,sw,xy,tuak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), dm1, dm1, dm1, block4(r_phys, a, a, p, f), optimize=True),
        "Delta1_e": 0.5 * np.einsum("pqrs,aypq,vwtu,kx,ry,sv,xw,tuak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), dm1, dm1, dm1, block4(r_phys, a, a, p, f), optimize=True),
        "Delta1_f": -0.25 * np.einsum("pqrs,aypq,vwtu,kx,ry,sv,xw,utak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), dm1, dm1, dm1, block4(r_phys, a, a, p, f), optimize=True),
        "Delta2_a": np.einsum("pqrs,ayqp,vwtu,xrvy,kx,sw,utak", t, block4(r_phys, p, a, a, a), t, dm2, block2(fock, f, a), dm1, block4(r_phys, a, a, p, f), optimize=True),
        "Delta2_b": -0.5 * np.einsum("pqrs,ayqp,vwtu,xrvy,kx,sw,tuak", t, block4(r_phys, p, a, a, a), t, dm2, block2(fock, f, a), dm1, block4(r_phys, a, a, p, f), optimize=True),
        "Delta2_c": -np.einsum("pqrs,ayqp,vwtu,kx,ry,sv,xw,tuak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), dm1, dm1, dm1, block4(r_phys, a, a, p, f), optimize=True),
        "Delta2_d": 0.5 * np.einsum("pqrs,ayqp,vwtu,kx,ry,sv,xw,utak", t, block4(r_phys, p, a, a, a), t, block2(fock, f, a), dm1, dm1, dm1, block4(r_phys, a, a, p, f), optimize=True),
    }
    sub.update({k: float(v) for k, v in delta_terms.items()})

    sub["V_total"] = sub["V_KL_full"] + sub["V_TU_obs_subtraction"] + sub["V_AT_passive_subtraction"]
    sub["B_total"] = sum(v for k, v in sub.items() if k.startswith("B_"))
    sub["X_total"] = sum(v for k, v in sub.items() if k.startswith("X_"))
    sub["Delta_total"] = sum(v for k, v in sub.items() if k.startswith("Delta"))
    sub["correction_total"] = sub["V_total"] + sub["B_total"] + sub["X_total"] + sub["Delta_total"]
    sub["active_trace_dm1"] = float(np.trace(dm1))
    sub["active_trace_dm2"] = float(np.einsum("pprr->", dm2, optimize=True))
    return sub


def passive_variants(case: Dict[str, Any], nocc: int, thresholds: List[float]) -> List[Dict[str, Any]]:
    nobs = case["nobs"]
    nri = case["nri"]
    occ = np.array(case["natural_occupations"], dtype=float)
    obs = list(range(nobs))
    cabs = list(range(nobs, nri))
    variants = [
        {"mode": "cabs_only", "active": obs, "passive": cabs, "threshold": None},
        {"mode": "occupied_external", "active": obs, "passive": list(range(nocc, nri)), "threshold": None},
    ]
    for th in thresholds:
        low = [i for i in obs if i >= nocc and occ[i] < th]
        active = [i for i in obs if i not in low]
        passive = sorted(low + cabs)
        variants.append({"mode": f"low_occ_passive_lt_{th:g}", "active": active, "passive": passive, "threshold": th})
    return variants


def group_for_subterm(name: str) -> str:
    if name.startswith("V_"):
        return "V"
    if name.startswith("B_"):
        return "B"
    if name.startswith("X_"):
        return "X"
    if name.startswith("Delta"):
        return "Delta"
    if name == "correction_total":
        return "correction"
    return "diagnostic"


def audit_cases(cases: List[Dict[str, Any]], nocc: int, thresholds: List[float]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    summary_rows = []
    for case in cases:
        for variant in passive_variants(case, nocc, thresholds):
            sub = signed_subterms(
                case["g_phys"],
                case["r_phys"],
                case["fock"],
                case["dm1_obs"],
                case["dm2_obs"],
                variant["active"],
                variant["passive"],
                case["nri"],
            )
            common = {
                "case": case["label"],
                "nobs": case["nobs"],
                "nri": case["nri"],
                "nqubits": 2 * case["nobs"],
                "passive_mode": variant["mode"],
                "threshold": variant["threshold"],
                "active_size": len(variant["active"]),
                "passive_size": len(variant["passive"]),
                "active_trace_dm1": sub["active_trace_dm1"],
                "active_trace_dm2": sub["active_trace_dm2"],
            }
            summary_rows.append(
                {
                    **common,
                    "V_total": sub["V_total"],
                    "B_total": sub["B_total"],
                    "X_total": sub["X_total"],
                    "Delta_total": sub["Delta_total"],
                    "correction_total": sub["correction_total"],
                }
            )
            for name, value in sub.items():
                if name in ("active_trace_dm1", "active_trace_dm2"):
                    continue
                rows.append({**common, "group": group_for_subterm(name), "subterm": name, "value": value, "value_mEh": 1000.0 * value})
    return rows, summary_rows


def build_diffs(rows: List[Dict[str, Any]], transitions: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    by_key = {(r["case"], r["passive_mode"], r["subterm"]): r for r in rows}
    out = []
    for left, right in transitions:
        modes = sorted({r["passive_mode"] for r in rows if r["case"] in (left, right)})
        subterms = sorted({r["subterm"] for r in rows if r["case"] in (left, right)})
        for mode in modes:
            for subterm in subterms:
                a = by_key.get((left, mode, subterm))
                b = by_key.get((right, mode, subterm))
                if a is None or b is None:
                    continue
                delta = b["value"] - a["value"]
                out.append(
                    {
                        "transition": f"{left} -> {right}",
                        "passive_mode": mode,
                        "group": group_for_subterm(subterm),
                        "subterm": subterm,
                        "left_value_mEh": a["value_mEh"],
                        "right_value_mEh": b["value_mEh"],
                        "delta_mEh": 1000.0 * delta,
                    }
                )
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args, cases: List[Dict[str, Any]], rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], diffs: List[Dict[str, Any]]) -> None:
    payload = {
        "cases": [
            {
                "label": c["label"],
                "path": c["path"],
                "nobs": c["nobs"],
                "nri": c["nri"],
                "E_obs_fci": c["E_obs"],
                "energy_checks": c["energy_checks"],
                "natural_occupations": c["natural_occupations"],
            }
            for c in cases
        ],
        "summary_rows": summary_rows,
        "subterm_rows": rows,
        "diff_rows": diffs,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(args.out_csv, rows)
    write_csv(args.diff_csv, diffs)

    lines = []
    lines.append("=" * 118)
    lines.append("Step 7g | ECG-NO SF-[2]R12 signed subterm and projector-subtraction audit")
    lines.append("=" * 118)
    for c in cases:
        lines.append("")
        lines.append(f"[{c['label']}] nobs={c['nobs']} nri={c['nri']} E_obs={c['E_obs']:.14f} Eh")
        lines.append(
            f"energy checks: OBS-RDM {c['energy_checks']['delta_obs_rdm_minus_fci']:.3e}, "
            f"RI-RDM {c['energy_checks']['delta_ri_rdm_minus_fci']:.3e} Eh"
        )
        for s in [r for r in summary_rows if r["case"] == c["label"]]:
            lines.append(
                f"  {s['passive_mode']:<26s} active={s['active_size']:<3d} passive={s['passive_size']:<4d} "
                f"Tr1={s['active_trace_dm1']:.8f} "
                f"V={1000*s['V_total']: .8f} B={1000*s['B_total']: .8f} "
                f"X={1000*s['X_total']: .8f} D={1000*s['Delta_total']: .8f} "
                f"total={1000*s['correction_total']: .8f} mEh"
            )
            if s["passive_mode"] == "cabs_only":
                top = sorted(
                    [r for r in rows if r["case"] == c["label"] and r["passive_mode"] == "cabs_only" and not r["subterm"].endswith("_total")],
                    key=lambda x: abs(x["value_mEh"]),
                    reverse=True,
                )[:8]
                lines.append("    largest signed subterms:")
                for t in top:
                    lines.append(f"      {t['subterm']:<42s} {t['value_mEh']: .8f} mEh")
    lines.append("")
    lines.append("[CABS-only transition diffs: largest |delta|]")
    cabs_diffs = sorted([d for d in diffs if d["passive_mode"] == "cabs_only"], key=lambda x: abs(x["delta_mEh"]), reverse=True)[:16]
    for d in cabs_diffs:
        lines.append(f"  {d['transition']:<55s} {d['subterm']:<42s} {d['delta_mEh']: .8f} mEh")
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out_json}")
    lines.append(f"  {args.out_csv}")
    lines.append(f"  {args.diff_csv}")
    lines.append(f"  {args.summary}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    thresholds = parse_float_list(args.low_occ_thresholds)
    cases = [load_case(path) for path in args.inputs]
    rows, summary_rows = audit_cases(cases, args.nocc, thresholds)
    labels = [c["label"] for c in cases]
    transitions = list(zip(labels[:-1], labels[1:]))
    diffs = build_diffs(rows, transitions)
    write_outputs(args, cases, rows, summary_rows, diffs)


if __name__ == "__main__":
    main()
