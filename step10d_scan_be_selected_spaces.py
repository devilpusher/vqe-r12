#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 10d: scan Be selected-space ECG-NO + SF-[2]R12 stability."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_BE_ECG_REFERENCE = -14.6672938007836


@dataclass
class Case:
    name: str
    s_pick: str
    p_pick: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--be-dir", default="/mnt/d/vqecodex/be2")
    p.add_argument("--fitN", type=int, default=7)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--prefix", default="step10d_be_selected_space_scan_fitN7")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--memory", default="4 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--reference-energy", type=float, default=DEFAULT_BE_ECG_REFERENCE)
    p.add_argument("--reference-label", default="early local Be ECG reference from enerx.dat")
    p.add_argument("--cases", default="sp_s012_p0,sp_s0123_p0,sp_s012_p01,sp_s0123_p01")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def available_cases() -> Dict[str, Case]:
    return {
        "sp_s012_p0": Case("sp_s012_p0", "0,1,2", "0"),
        "sp_s0123_p0": Case("sp_s0123_p0", "0,1,2,3", "0"),
        "sp_s012_p01": Case("sp_s012_p01", "0,1,2", "0,1"),
        "sp_s0123_p01": Case("sp_s0123_p01", "0,1,2,3", "0,1"),
    }


def parse_cases(s: str) -> List[Case]:
    table = available_cases()
    out = []
    for name in [x.strip() for x in s.split(",") if x.strip()]:
        if name not in table:
            raise ValueError(f"Unknown case {name}; available: {', '.join(sorted(table))}")
        out.append(table[name])
    return out


def pick_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def nobs(case: Case) -> int:
    return len(pick_list(case.s_pick)) + 3 * len(pick_list(case.p_pick))


def run(cmd: List[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paths(case: Case, fitn: int) -> Dict[str, str]:
    tag = case.name
    return {
        "tag": tag,
        "step10a_npz": f"step10a_be_{tag}_rdm_export.npz",
        "step10a_json": f"step10a_be_{tag}_rdm_export.json",
        "step10a_summary": f"step10a_be_{tag}_rdm_export_summary.txt",
        "step10b_npz": f"step10b_be_{tag}_fitN{fitn}_step4b_like.npz",
        "step10b_json": f"step10b_be_{tag}_fitN{fitn}_step4b_like.json",
        "step10b_summary": f"step10b_be_{tag}_fitN{fitn}_step4b_like_summary.txt",
        "step10c_prefix": f"step10c_be_{tag}_fitN{fitn}",
        "step10c_json": f"step10c_be_{tag}_fitN{fitn}_sf2r12_correction.json",
        "step10c_csv": f"step10c_be_{tag}_fitN{fitn}_sf2r12_correction.csv",
        "step10c_summary": f"step10c_be_{tag}_fitN{fitn}_sf2r12_correction_summary.txt",
    }


def ensure_case(args, case: Case) -> Dict[str, str]:
    p = paths(case, args.fitN)
    if args.force or not Path(p["step10a_npz"]).exists():
        run(
            [
                args.python,
                "step10a_export_be_ecg_no_rdm_space.py",
                "--be-dir",
                args.be_dir,
                "--s-pick",
                case.s_pick,
                "--p-pick",
                case.p_pick,
                "--out",
                p["step10a_npz"],
                "--json",
                p["step10a_json"],
                "--summary",
                p["step10a_summary"],
            ],
            args.dry_run,
        )
    if args.force or not Path(p["step10b_npz"]).exists():
        run(
            [
                args.python,
                "step10b_build_be_step4b_like.py",
                "--be-dir",
                args.be_dir,
                "--rdm-inp",
                p["step10a_npz"],
                "--s-pick",
                case.s_pick,
                "--p-pick",
                case.p_pick,
                "--fitN",
                str(args.fitN),
                "--memory",
                args.memory,
                "--nthreads",
                str(args.nthreads),
                "--r12-only",
                "--out",
                p["step10b_npz"],
                "--json",
                p["step10b_json"],
                "--summary",
                p["step10b_summary"],
            ],
            args.dry_run,
        )
    if args.force or not Path(p["step10c_json"]).exists():
        run(
            [
                args.python,
                "step10c_be_r12_correction.py",
                "--inp",
                p["step10b_npz"],
                "--prefix",
                p["step10c_prefix"],
            ],
            args.dry_run,
        )
    return p


def row_from_outputs(args, case: Case, p: Dict[str, str]) -> Dict[str, Any]:
    a_meta = load_json(p["step10a_json"])
    b_meta = load_json(p["step10b_json"])
    c_payload = load_json(p["step10c_json"])
    result = c_payload["result"]
    comp = result["components"]
    E_obs = float(result["E_obs_fci"])
    delta = float(result["delta_E_r12"])
    E_total = float(result["E_total"])
    ref = float(args.reference_energy)
    gap = ref - E_obs
    residual = E_total - ref
    recovery = delta / gap if abs(gap) > 1e-14 else None
    diag = result["diagnostics"]["rdm_diagnostics"]["obs"]
    return {
        "case": case.name,
        "s_pick": case.s_pick,
        "p_pick": case.p_pick,
        "fitN": args.fitN,
        "nobs": result["nobs"],
        "nqubits": 2 * int(result["nobs"]),
        "ncabs": result["ncabs"],
        "nri": result["nri"],
        "E_obs_fci": E_obs,
        "delta_E_r12": delta,
        "E_total": E_total,
        "reference_energy": ref,
        "gap_ref_minus_obs": gap,
        "residual_total_minus_ref": residual,
        "recovery_ratio": recovery,
        "gap_mEh": 1000.0 * gap,
        "delta_mEh": 1000.0 * delta,
        "residual_mEh": 1000.0 * residual,
        "V": comp["V"],
        "B": comp["B"],
        "X": comp["X"],
        "Delta": comp["Delta"],
        "V_mEh": 1000.0 * float(comp["V"]),
        "B_mEh": 1000.0 * float(comp["B"]),
        "X_mEh": 1000.0 * float(comp["X"]),
        "Delta_mEh": 1000.0 * float(comp["Delta"]),
        "trace_dm1": diag["trace_dm1"],
        "trace_dm2": diag["trace_dm2"],
        "natural_occ": ";".join(f"{float(x):.12g}" for x in diag["natural_occupations"]),
        "step10a_delta_rdm_minus_fci": a_meta["checks"]["delta_rdm_minus_fci"],
        "step10b_delta_ri_minus_input": b_meta["checks"]["delta_ri_rdm_minus_input_energy"],
        "step10c_delta_ri_minus_fci": result["diagnostics"]["energy_checks"]["delta_ri_rdm_minus_fci"],
    }


def write_outputs(args, rows: List[Dict[str, Any]]) -> None:
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    payload = {
        "step": "10d",
        "fitN": args.fitN,
        "reference_energy": args.reference_energy,
        "reference_label": args.reference_label,
        "rows": rows,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=" * 118,
        "Step 10d | Be selected-space ECG-NO + SF-[2]R12 stability scan",
        "=" * 118,
        f"fitN              = {args.fitN}",
        f"reference energy  = {args.reference_energy:.14f} Eh ({args.reference_label})",
        "",
        f"{'case':<15s} {'nobs':>4s} {'E_obs':>18s} {'dR12/mEh':>11s} {'E+R12':>18s} {'recov':>9s} {'resid/mEh':>11s}",
        "-" * 118,
    ]
    for r in rows:
        rec = "" if r["recovery_ratio"] is None else f"{float(r['recovery_ratio']):.6f}"
        lines.append(
            f"{r['case']:<15s} {int(r['nobs']):4d} "
            f"{float(r['E_obs_fci']):18.12f} {float(r['delta_mEh']):11.6f} "
            f"{float(r['E_total']):18.12f} {rec:>9s} {float(r['residual_mEh']):11.6f}"
        )
    lines.extend(
        [
            "",
            "[Components, mEh]",
            f"{'case':<15s} {'V':>12s} {'B':>12s} {'X':>12s} {'Delta':>12s}",
            "-" * 67,
        ]
    )
    for r in rows:
        lines.append(
            f"{r['case']:<15s} {float(r['V_mEh']):12.6f} {float(r['B_mEh']):12.6f} "
            f"{float(r['X_mEh']):12.6f} {float(r['Delta_mEh']):12.6f}"
        )
    lines.extend(["", "[Saved]", f"  {args.out_json}", f"  {args.out_csv}", f"  {args.summary}"])
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for case in parse_cases(args.cases):
        p = ensure_case(args, case)
        if args.dry_run:
            rows.append({"case": case.name, "status": "dry-run"})
        else:
            rows.append(row_from_outputs(args, case, p))
    if not args.dry_run:
        write_outputs(args, rows)


if __name__ == "__main__":
    main()
