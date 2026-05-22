#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7e: Scan ECG-NO SF-[2]R12 fitN and p/d active-space convergence.

The default scan is intentionally moderate:

* fitN = 3,5,7,9
* s[0,1,2] + p[0]
* s[0,1,2] + p[0,1]

The d[0] extension is included as an optional large case.  With the current
even-tempered parent basis, adding d raises nao from 64 to 144; full RI
four-index tensors then become multi-GB objects.  Use --include-d --allow-large
only when you really want to spend that memory/disk budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


DEG = {"s": 1, "p": 3, "d": 5, "f": 7}
NS = 16
DEFAULT_REFERENCE_ECG14 = -2.9017962843565535


@dataclass
class ScanCase:
    name: str
    channels: List[str]
    picks: Dict[str, List[int]]
    basis_name: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fitNs", default="3,5,7,9")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--prefix", default="step7e_ecg_no_r12_scan")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-d", action="store_true")
    p.add_argument("--allow-large", action="store_true")
    p.add_argument("--max-nao", type=int, default=80, help="Skip cases above this nao unless --allow-large.")
    p.add_argument("--memory", default="4 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--reference-energy", type=float, default=DEFAULT_REFERENCE_ECG14)
    p.add_argument("--reference-label", default="local 14-orbital ECG-NO-FCI from he_2rdm_compare.py")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def pick_arg(xs: List[int]) -> str:
    return ",".join(str(x) for x in xs)


def nao_for_channels(channels: List[str]) -> int:
    return NS * sum(DEG[ch] for ch in channels)


def nobs_for_case(case: ScanCase) -> int:
    return sum(DEG[ch] * len(case.picks.get(ch, [])) for ch in case.channels)


def tensor_gib(nao: int) -> float:
    return nao**4 * 8.0 / (1024.0**3)


def cases(include_d: bool) -> List[ScanCase]:
    out = [
        ScanCase("sp_s012_p0", ["s", "p"], {"s": [0, 1, 2], "p": [0], "d": [], "f": []}, "heecgnosp"),
        ScanCase("sp_s012_p01", ["s", "p"], {"s": [0, 1, 2], "p": [0, 1], "d": [], "f": []}, "heecgnosp"),
    ]
    if include_d:
        out.append(
            ScanCase(
                "spd_s012_p01_d0",
                ["s", "p", "d"],
                {"s": [0, 1, 2], "p": [0, 1], "d": [0], "f": []},
                "heecgnospd",
            )
        )
    return out


def run(cmd: List[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def output_paths(case: ScanCase, fitn: int) -> Dict[str, str]:
    tag = f"{case.name}_fitN{fitn}"
    return {
        "tag": tag,
        "step7c_npz": f"step7c_ecg_no_{tag}_r12only_step4b_like.npz",
        "step7c_json": f"step7c_ecg_no_{tag}_r12only_step4b_like.json",
        "step7c_summary": f"step7c_ecg_no_{tag}_r12only_step4b_like_summary.txt",
        "step7d_prefix": f"step7d_ecg_no_{tag}_r12only",
        "step7d_json": f"step7d_ecg_no_{tag}_r12only_sf2r12_correction.json",
        "step7d_csv": f"step7d_ecg_no_{tag}_r12only_sf2r12_correction.csv",
        "step7d_summary": f"step7d_ecg_no_{tag}_r12only_sf2r12_correction_summary.txt",
    }


def run_point(args, case: ScanCase, fitn: int) -> Dict[str, Any]:
    paths = output_paths(case, fitn)
    nao = nao_for_channels(case.channels)
    nobs = nobs_for_case(case)
    estimated_tensor_gib = tensor_gib(nao)
    row: Dict[str, Any] = {
        "case": case.name,
        "channels": ",".join(case.channels),
        "s_pick": pick_arg(case.picks.get("s", [])),
        "p_pick": pick_arg(case.picks.get("p", [])),
        "d_pick": pick_arg(case.picks.get("d", [])),
        "fitN": fitn,
        "nao": nao,
        "nobs": nobs,
        "nqubits": 2 * nobs,
        "estimated_single_tensor_GiB": estimated_tensor_gib,
        "status": "pending",
        "skip_reason": "",
    }
    if nao > args.max_nao and not args.allow_large:
        row["status"] = "skipped"
        row["skip_reason"] = f"nao={nao} exceeds --max-nao={args.max_nao}; single tensor ~{estimated_tensor_gib:.2f} GiB"
        return row

    if args.force or not Path(paths["step7c_npz"]).exists():
        run(
            [
                args.python,
                "step7c_build_ecg_no_step4b_like.py",
                "--channels",
                ",".join(case.channels),
                "--s-pick",
                pick_arg(case.picks.get("s", [])),
                "--p-pick",
                pick_arg(case.picks.get("p", [])),
                "--d-pick",
                pick_arg(case.picks.get("d", [])),
                "--f-pick",
                pick_arg(case.picks.get("f", [])),
                "--basis-name",
                case.basis_name,
                "--fitN",
                str(fitn),
                "--memory",
                args.memory,
                "--nthreads",
                str(args.nthreads),
                "--r12-only",
                "--out",
                paths["step7c_npz"],
                "--json",
                paths["step7c_json"],
                "--summary",
                paths["step7c_summary"],
            ],
            args.dry_run,
        )
    if args.force or not Path(paths["step7d_json"]).exists():
        run(
            [
                args.python,
                "step7d_ecg_no_r12_correction.py",
                "--inp",
                paths["step7c_npz"],
                "--prefix",
                paths["step7d_prefix"],
            ],
            args.dry_run,
        )
    if args.dry_run:
        row["status"] = "dry-run"
        return row

    c_meta = load_json(paths["step7c_json"])
    d_payload = load_json(paths["step7d_json"])
    result = d_payload["result"]
    fit_metrics = c_meta["corr_info"].get("fit_metrics", {})
    reference = args.reference_energy
    E_obs = result["E_obs_fci"]
    delta = result["delta_E_r12"]
    E_total = result["E_total"]
    gap_to_ref = reference - E_obs if reference is not None else None
    recovery = delta / gap_to_ref if gap_to_ref is not None and abs(gap_to_ref) > 1e-10 else None
    residual = E_total - reference if reference is not None else None
    row.update(
        {
            "status": "ok",
            "nri": result["nri"],
            "ncabs": result["ncabs"],
            "fit_rms": fit_metrics.get("rms_abs"),
            "fit_rel_rms": fit_metrics.get("rel_rms"),
            "fit_f0_error": fit_metrics.get("f0_error"),
            "E_obs_fci": E_obs,
            "delta_E_r12": delta,
            "E_total": E_total,
            "reference_energy": reference,
            "reference_label": args.reference_label,
            "gap_obs_to_reference": gap_to_ref,
            "residual_to_reference": residual,
            "abs_residual_to_reference_mEh": None if residual is None else abs(residual) * 1000.0,
            "recovery_vs_reference": recovery,
            "V": result["components"]["V"],
            "B": result["components"]["B"],
            "X": result["components"]["X"],
            "Delta": result["components"]["Delta"],
            "step7c_npz": paths["step7c_npz"],
            "step7d_json": paths["step7d_json"],
        }
    )
    return row


def write_outputs(args, rows: List[Dict[str, Any]]) -> None:
    args.out_csv = args.out_csv or f"{args.prefix}.csv"
    args.out_json = args.out_json or f"{args.prefix}.json"
    args.summary = args.summary or f"{args.prefix}_summary.txt"
    keys = [
        "case",
        "channels",
        "s_pick",
        "p_pick",
        "d_pick",
        "fitN",
        "nao",
        "nobs",
        "nqubits",
        "nri",
        "ncabs",
        "status",
        "skip_reason",
        "fit_rms",
        "fit_rel_rms",
        "fit_f0_error",
        "E_obs_fci",
        "delta_E_r12",
        "E_total",
        "reference_energy",
        "gap_obs_to_reference",
        "residual_to_reference",
        "abs_residual_to_reference_mEh",
        "recovery_vs_reference",
        "V",
        "B",
        "X",
        "Delta",
        "estimated_single_tensor_GiB",
        "step7c_npz",
        "step7d_json",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    lines = []
    lines.append("=" * 110)
    lines.append("Step 7e | ECG-NO SF-[2]R12 fitN and p/d convergence scan")
    lines.append("=" * 110)
    lines.append(f"reference = {args.reference_energy} Eh ({args.reference_label})")
    lines.append("")
    lines.append("case              fitN nqubits status     E_obs_FCI          Delta_R12(mEh)   E_total            residual_ref(mEh) recovery")
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"{row['case']:<17s} {row['fitN']:>4} {row['nqubits']:>7} {row['status']:<10s} {row['skip_reason']}")
            continue
        lines.append(
            f"{row['case']:<17s} {row['fitN']:>4} {row['nqubits']:>7} ok         "
            f"{row['E_obs_fci']: .12f} {1000.0 * row['delta_E_r12']: .8f} "
            f"{row['E_total']: .12f} {1000.0 * row['residual_to_reference']: .8f} "
            f"{'' if row['recovery_vs_reference'] is None else f'{row['recovery_vs_reference']: .8f}'}"
        )
    lines.append("")
    lines.append("[Saved]")
    lines.append(f"  {args.out_csv}")
    lines.append(f"  {args.out_json}")
    lines.append(f"  {args.summary}")
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


def main():
    args = parse_args()
    fitns = parse_int_list(args.fitNs)
    rows = []
    for case in cases(args.include_d):
        for fitn in fitns:
            rows.append(run_point(args, case, fitn))
    write_outputs(args, rows)


if __name__ == "__main__":
    main()
