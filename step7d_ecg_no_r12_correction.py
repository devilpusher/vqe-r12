#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7d: Apply the audited SF-[2]R12 correction to ECG-NO Step7c input.

Input is the Step7c Step-4b-like `.npz`, which already contains the selected
ECG-NO OBS RDMs, CABS+/RI tensors, and energy reconstruction checks.  This
script applies the same paper/Tequila spin-free [2]R12 candidate used by Step6m.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict

from r12_correction import compute_he_sf2r12_correction, validate_correction_result
from step7i_residual_weighted_dual_space import case_record, clipped01, parse_core_tokens, safe_ratio


DEFAULT_REF_EXCLUDED_OCC_SUM = 4.959159658929e-04
DEFAULT_REF_MISSING_TRACE = 5.316135369720e-04


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="step7c_ecg_no_sp_s012_p0_fitN7_step4b_like.npz")
    p.add_argument("--scale-f12", type=float, default=1.0)
    p.add_argument("--prefix", default="step7d_ecg_no_sp_s012_p0_fitN7")
    p.add_argument("--core", default="s0,s1,s2,p0")
    p.add_argument("--no-experimental-refscale", action="store_true")
    p.add_argument("--refscale-family", default="spd_s012_p01_d0")
    p.add_argument("--ref-excluded-occ-sum", type=float, default=DEFAULT_REF_EXCLUDED_OCC_SUM)
    p.add_argument("--ref-missing-trace", type=float, default=DEFAULT_REF_MISSING_TRACE)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def fmt(x: Any, prec: int = 12) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{prec}e}"
    except Exception:
        return str(x)


def build_candidate_rows(args, result: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return conservative and experimental rows without changing result."""
    components = result["components"]
    rows = [
        {
            "method": result["method"],
            "role": "conservative",
            "weight": 1.0,
            "raw_delta_E_r12": result["delta_E_r12"],
            "delta_E_r12": result["delta_E_r12"],
            "E_total": result["E_total"],
            "passive_space": result["passive_space"],
            "projector_core": "all_OBS_active",
            "refscale_family": "",
            "excluded_occ_sum": "",
            "ref_excluded_occ_sum": "",
            "missing_trace_dm1": "",
            "ref_missing_trace": "",
            "components": components,
        }
    ]
    if args.no_experimental_refscale:
        return rows

    rec = case_record(args.inp, parse_core_tokens(args.core))
    w_occ = clipped01(1.0 - safe_ratio(rec["excluded_occ_sum"], args.ref_excluded_occ_sum))
    delta = w_occ * rec["core_delta"]
    rows.append(
        {
            "method": "experimental_refscale_inverse_excluded_occ_sum",
            "role": "experimental",
            "weight": w_occ,
            "raw_delta_E_r12": rec["core_delta"],
            "delta_E_r12": delta,
            "E_total": rec["E_obs"] + delta,
            "passive_space": "dual_core_remaining_OBS_plus_CABS",
            "projector_core": args.core,
            "refscale_family": args.refscale_family,
            "excluded_occ_sum": rec["excluded_occ_sum"],
            "ref_excluded_occ_sum": args.ref_excluded_occ_sum,
            "missing_trace_dm1": rec["core_missing_trace_dm1"],
            "ref_missing_trace": args.ref_missing_trace,
            "components": {
                "V": rec["V_core"],
                "B": rec["B_core"],
                "X": rec["X_core"],
                "Delta": rec["Delta_core"],
            },
        }
    )
    return rows


def write_outputs(args, result: Dict[str, Any], candidate_rows: list[Dict[str, Any]]) -> None:
    args.out_json = args.out_json or f"{args.prefix}_sf2r12_correction.json"
    args.out_csv = args.out_csv or f"{args.prefix}_sf2r12_correction.csv"
    args.summary = args.summary or f"{args.prefix}_sf2r12_correction_summary.txt"

    payload = {
        "pipeline": {
            "source": "Step7c ECG-NO selected-OBS bridge",
            "input": args.inp,
            "scale_f12": args.scale_f12,
            "experimental_refscale_enabled": not args.no_experimental_refscale,
            "projector_core": args.core,
            "refscale_family": args.refscale_family,
        },
        "result": result,
        "candidate_rows": candidate_rows,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "method",
        "role",
        "nobs",
        "nri",
        "ncabs",
        "weight",
        "raw_delta_E_r12",
        "E_obs_fci",
        "delta_E_r12",
        "E_total",
        "passive_space",
        "projector_core",
        "excluded_occ_sum",
        "ref_excluded_occ_sum",
        "V",
        "B",
        "X",
        "Delta",
    ]
    rows = []
    for cand in candidate_rows:
        comp = cand["components"]
        rows.append(
            {
                "method": cand["method"],
                "role": cand["role"],
                "nobs": result["nobs"],
                "nri": result["nri"],
                "ncabs": result["ncabs"],
                "weight": cand["weight"],
                "raw_delta_E_r12": cand["raw_delta_E_r12"],
                "E_obs_fci": result["E_obs_fci"],
                "delta_E_r12": cand["delta_E_r12"],
                "E_total": cand["E_total"],
                "passive_space": cand["passive_space"],
                "projector_core": cand["projector_core"],
                "excluded_occ_sum": cand["excluded_occ_sum"],
                "ref_excluded_occ_sum": cand["ref_excluded_occ_sum"],
                "V": comp["V"],
                "B": comp["B"],
                "X": comp["X"],
                "Delta": comp["Delta"],
            }
        )
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    components = result["components"]
    checks = result["diagnostics"]["energy_checks"]
    rdm_obs = result["diagnostics"]["rdm_diagnostics"]["obs"]
    tensor = result["diagnostics"]["tensor_diagnostics"]
    fock = result["diagnostics"]["fock_diagnostics"]
    lines = []
    lines.append("=" * 100)
    lines.append("Step 7d | ECG-NO selected-OBS SF-[2]R12 correction")
    lines.append("=" * 100)
    lines.append(f"input        = {args.inp}")
    lines.append(f"method       = {result['method']} ({result['fock_model']})")
    lines.append(f"passive      = {result['passive_space']}")
    lines.append(f"nobs/ncabs/nri = {result['nobs']}/{result['ncabs']}/{result['nri']}")
    lines.append("")
    lines.append("[Energy]")
    lines.append(f"E_ECG_NO_OBS_FCI        = {result['E_obs_fci']:.14f} Eh")
    lines.append(f"DeltaE_R12              = {result['delta_E_r12']:.12e} Eh")
    lines.append(f"E_ECG_NO_OBS_plus_R12   = {result['E_total']:.14f} Eh")
    lines.append("full-parent target      = not available for Step7c ECG-NO bridge")
    lines.append("")
    lines.append("[Components]")
    lines.append(f"V      = {components['V']:.12e} Eh")
    lines.append(f"B      = {components['B']:.12e} Eh")
    lines.append(f"X      = {components['X']:.12e} Eh")
    lines.append(f"Delta  = {components['Delta']:.12e} Eh")
    lines.append("")
    lines.append("[Candidate Rows]")
    for cand in candidate_rows:
        lines.append(
            f"{cand['role']:<12s} {cand['method']:<48s} "
            f"w={float(cand['weight']):.6f} "
            f"DeltaE={float(cand['delta_E_r12']): .12e} Eh "
            f"E={float(cand['E_total']): .14f} Eh"
        )
    if not args.no_experimental_refscale:
        lines.append(
            f"experimental refscale: family={args.refscale_family}, "
            f"excluded_occ_ref={args.ref_excluded_occ_sum:.12e}"
        )
    lines.append("")
    lines.append("[Diagnostics]")
    lines.append(f"Delta OBS-RDM minus FCI = {checks['delta_obs_rdm_minus_fci']:.3e} Eh")
    lines.append(f"Delta RI-RDM minus FCI  = {checks['delta_ri_rdm_minus_fci']:.3e} Eh")
    lines.append(f"Tr(dm1_obs), Tr(dm2_obs) = {rdm_obs['trace_dm1']:.12f}, {rdm_obs['trace_dm2']:.12f}")
    lines.append(f"eri/f12 bra-ket error    = {tensor['eri_phys']['bra_ket_error']:.3e}, {tensor['f12_phys']['bra_ket_error']:.3e}")
    lines.append(f"Step5a Fock available    = {fock['fock_step5a_available']}")
    lines.append(f"max|F_tequila-F_step5a|  = {fmt(fock['maxabs_fock_tequila_minus_step5a'], 3)}")
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out_json}")
    lines.append(f"  {args.out_csv}")
    lines.append(f"  {args.summary}")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    result = compute_he_sf2r12_correction(args.inp, step4b_path=None, scale_f12=args.scale_f12)
    validate_correction_result(result)
    candidate_rows = build_candidate_rows(args, result)
    write_outputs(args, result, candidate_rows)


if __name__ == "__main__":
    main()
