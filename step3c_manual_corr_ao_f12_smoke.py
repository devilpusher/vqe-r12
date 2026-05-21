#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3c: Manual Gaussian-correlation ao_f12 smoke test.

Reason
------
Step 3b showed that Psi4's MintsHelper.ao_f12 expects:

    corr: list[tuple[float, float]]

rather than a FittedSlaterCorrelationFactor object.  The fitted Slater object
does not expose the full Gaussian expansion in the Python API on this build;
only "exponent" is public.

This script therefore tests the F12 integral backend directly with manually
provided Gaussian correlation-factor lists.  It does not yet implement the
article's fitted Slater factor.  Its purpose is only to verify:

1. mints.ao_f12(corr) works;
2. mixed-basis calls such as ao_f12(corr, obs, obs, cabs, cabs) work;
3. related F12 intermediates are callable:
      ao_f12_squared
      ao_f12g12
      ao_f12_double_commutator

Each call runs in a separate child process with a timeout.

Usage
-----
    python step3c_manual_corr_ao_f12_smoke.py

Optional:
    python step3c_manual_corr_ao_f12_smoke.py --timeout 10
    python step3c_manual_corr_ao_f12_smoke.py --obs-basis 6-31g --cabs-basis cc-pvdz
    python step3c_manual_corr_ao_f12_smoke.py --corr "1.0,1.0;-0.714285714,1.96"

The --corr string is interpreted as:
    coefficient,exponent;coefficient,exponent;...

Important
---------
The tuple order expected by Psi4 is inferred from the accepted API:
    list[tuple[float, float]]
but the names are not exposed by the error message.  In this script the first
entry is treated as coefficient and the second as Gaussian exponent.  For a
true [2]R12 implementation this convention must still be verified against
Psi4/Libint source or a known reference value.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--obs-basis", default="6-31g")
    p.add_argument("--cabs-basis", default="cc-pvdz")
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--memory", default="1 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--corr", default=None, help="Manual corr list: 'coef,exp;coef,exp;...'.")
    p.add_argument("--out-json", default="step3c_manual_corr_ao_f12_smoke.json")
    p.add_argument("--summary", default="step3c_manual_corr_ao_f12_smoke_summary.txt")
    return p.parse_args()


def parse_corr(s: str) -> List[Tuple[float, float]]:
    pairs = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        xs = [float(x.strip()) for x in part.split(",")]
        if len(xs) != 2:
            raise ValueError(f"Bad corr pair: {part!r}")
        pairs.append((xs[0], xs[1]))
    if not pairs:
        raise ValueError("Empty --corr")
    return pairs


def default_corr_sets(gamma: float) -> Dict[str, List[Tuple[float, float]]]:
    return {
        # Pure smoke-test Gaussian.  This is not intended as fitted Slater.
        "one_gaussian_c1_a1": [(1.0, 1.0)],
        # Crude single-Gaussian surrogate with the article's prefactor -1/gamma.
        "one_gaussian_minus_inv_gamma_a_gamma": [(-1.0 / gamma, gamma)],
        "one_gaussian_minus_inv_gamma_a_gamma2": [(-1.0 / gamma, gamma * gamma)],
        # Two harmless Gaussians, only for additivity/API testing.
        "two_gaussian_test": [(0.5, 0.8), (-0.2, 2.0)],
    }


def run_child(label: str, code: str, timeout: float) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "label": label,
            "ok": proc.returncode == 0,
            "timeout": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "label": label,
            "ok": False,
            "timeout": True,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"TimeoutExpired after {timeout} s",
        }


def child_code(obs_basis: str, cabs_basis: str, memory: str, nthreads: int,
               corr: List[Tuple[float, float]], method: str, call_form: str) -> str:
    corr_literal = repr([(float(c), float(a)) for c, a in corr])
    return f"""
import json
import numpy as np
import psi4

psi4.core.clean()
psi4.set_memory({memory!r})
psi4.set_num_threads({nthreads})
psi4.core.set_output_file("psi4_step3c_child.out", False)

mol = psi4.geometry(\"\"\"
0 1
He 0.0 0.0 0.0
symmetry c1
units bohr
\"\"\")

obs = psi4.core.BasisSet.build(mol, "BASIS", {obs_basis!r})
cabs = psi4.core.BasisSet.build(mol, "BASIS", {cabs_basis!r})
mints = psi4.core.MintsHelper(obs)
corr = {corr_literal}
method = {method!r}
call_form = {call_form!r}
func = getattr(mints, method)

result = {{
    "obs_basis": {obs_basis!r},
    "cabs_basis": {cabs_basis!r},
    "corr": corr,
    "method": method,
    "call_form": call_form,
    "nao_obs": int(obs.nbf()),
    "nao_cabs": int(cabs.nbf()),
}}

try:
    if call_form == "default":
        val = func(corr)
    elif call_form == "obsobsobsobs":
        val = func(corr, obs, obs, obs, obs)
    elif call_form == "obsobscabscabs":
        val = func(corr, obs, obs, cabs, cabs)
    elif call_form == "obscabsobscabs":
        val = func(corr, obs, cabs, obs, cabs)
    elif call_form == "cabscabscabscabs":
        val = func(corr, cabs, cabs, cabs, cabs)
    else:
        raise ValueError("unknown call_form")
    arr = np.asarray(val)
    result["call_ok"] = True
    result["shape"] = list(arr.shape)
    result["size"] = int(arr.size)
    result["norm"] = float(np.linalg.norm(arr.reshape(-1))) if arr.size else 0.0
    result["max_abs"] = float(np.max(np.abs(arr))) if arr.size else 0.0
    result["min"] = float(np.min(arr)) if arr.size else 0.0
    result["max"] = float(np.max(arr)) if arr.size else 0.0
    if arr.size <= 64:
        result["values"] = arr.tolist()
    else:
        result["first10"] = arr.reshape(-1)[:10].tolist()
except Exception as e:
    result["call_ok"] = False
    result["error"] = repr(e)
    print(json.dumps(result, indent=2))
    raise SystemExit(2)

print(json.dumps(result, indent=2))
"""


def extract_json(stdout: str) -> Optional[Dict[str, Any]]:
    text = stdout.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start:end+1])
    except Exception:
        return None


def main():
    args = parse_args()

    if args.corr:
        corr_sets = {"manual": parse_corr(args.corr)}
    else:
        corr_sets = default_corr_sets(args.gamma)

    methods = [
        "ao_f12",
        "ao_f12_squared",
        "ao_f12g12",
        "ao_f12_double_commutator",
    ]
    call_forms = [
        "default",
        "obsobsobsobs",
        "obsobscabscabs",
        "obscabsobscabs",
        "cabscabscabscabs",
    ]

    report = {
        "obs_basis": args.obs_basis,
        "cabs_basis": args.cabs_basis,
        "gamma": args.gamma,
        "timeout_s": args.timeout,
        "corr_sets": {k: v for k, v in corr_sets.items()},
        "probes": [],
    }

    print("=" * 80)
    print("Step 3c | Manual corr-list ao_f12 smoke test")
    print("=" * 80)
    print(f"OBS basis  = {args.obs_basis}")
    print(f"CABS basis = {args.cabs_basis}")
    print(f"gamma      = {args.gamma}")
    print(f"timeout    = {args.timeout} s per call")
    print("")

    for corr_name, corr in corr_sets.items():
        print(f"[corr set] {corr_name}: {corr}")
        for method in methods:
            for form in call_forms:
                label = f"{corr_name} | {method} | {form}"
                code = child_code(
                    args.obs_basis,
                    args.cabs_basis,
                    args.memory,
                    args.nthreads,
                    corr,
                    method,
                    form,
                )
                rec = run_child(label, code, args.timeout)
                payload = extract_json(rec.get("stdout", ""))
                rec["payload"] = payload
                report["probes"].append(rec)

                status = "OK" if rec["ok"] else ("TIMEOUT" if rec.get("timeout") else "FAIL")
                if payload and payload.get("call_ok"):
                    print(f"  {status:<8s} {method:<28s} {form:<18s} shape={payload.get('shape')} norm={payload.get('norm'):.6e}")
                else:
                    err = ""
                    if payload and payload.get("error"):
                        err = payload["error"].split("\\n")[0]
                    elif rec.get("error"):
                        err = rec["error"]
                    elif rec.get("stderr"):
                        err = rec["stderr"].strip().splitlines()[-1] if rec["stderr"].strip() else ""
                    print(f"  {status:<8s} {method:<28s} {form:<18s} {err}")
        print("")

    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.summary).write_text(make_summary(report), encoding="utf-8")

    print("[Saved]")
    print(" ", args.out_json)
    print(" ", args.summary)


def make_summary(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("Step 3c summary")
    lines.append("=" * 80)
    lines.append(f"OBS basis = {report.get('obs_basis')}")
    lines.append(f"CABS basis = {report.get('cabs_basis')}")
    lines.append(f"gamma = {report.get('gamma')}")
    lines.append("")

    ok_count = 0
    fail_count = 0
    timeout_count = 0
    lines.append("[Successful calls]")
    for rec in report.get("probes", []):
        payload = rec.get("payload")
        if rec.get("ok") and payload and payload.get("call_ok"):
            ok_count += 1
            lines.append(
                f"  OK {rec['label']} shape={payload.get('shape')} "
                f"norm={payload.get('norm'):.12e} max_abs={payload.get('max_abs'):.12e}"
            )
        elif rec.get("timeout"):
            timeout_count += 1
        else:
            fail_count += 1

    lines.append("")
    lines.append("[Counts]")
    lines.append(f"ok = {ok_count}")
    lines.append(f"fail = {fail_count}")
    lines.append(f"timeout = {timeout_count}")

    lines.append("")
    lines.append("[Interpretation]")
    if ok_count > 0:
        lines.append("Psi4 accepts corr as list[tuple[float,float]] and can compute at least some F12 AO integrals.")
        lines.append("Next: determine tuple convention, tensor flattening/order, and whether mixed-basis F12 calls are stable enough for CABS+/RI transformations.")
    else:
        lines.append("No manual corr-list F12 call succeeded. Need to inspect Psi4 source/build configuration.")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
