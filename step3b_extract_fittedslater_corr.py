#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3b: Inspect Psi4 FittedSlaterCorrelationFactor and extract Gaussian expansion.

Background
----------
The previous minimal probe showed that Psi4 1.10 exposes

    psi4.core.FittedSlaterCorrelationFactor(gamma)

but MintsHelper.ao_f12 does NOT accept this object directly.  It expects

    corr: list[tuple[float, float]]

i.e. a Gaussian expansion representation of the Slater-type correlation
factor.

This script tries to discover how to extract that list from the Psi4 object.

Safety
------
Every method/property probe is executed in a separate child process with a
timeout, so a bad C++ bound method cannot hang the main process indefinitely.

Usage
-----
    python step3b_extract_fittedslater_corr.py

Optional:
    python step3b_extract_fittedslater_corr.py --timeout 8
    python step3b_extract_fittedslater_corr.py --gamma 1.4

Outputs
-------
    step3b_fittedslater_corr_probe.json
    step3b_fittedslater_corr_probe_summary.txt

Next expected outcome
---------------------
If the script finds a valid corr list, it will immediately test:

    mints.ao_f12(corr)

and report the returned tensor/matrix shape and norm.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional


COMMON_METHOD_CANDIDATES = [
    # likely accessors
    "nprimitive", "n_primitives", "nprimitive_", "nprim", "n_gaussians", "ngaussian",
    "coefficients", "coefs", "coeffs", "coef", "coeff", "get_coefficients", "get_coeffs",
    "exponents", "exps", "exp", "exponent", "get_exponents", "get_exps",
    "params", "parameters", "get_params", "terms", "get_terms",
    "cgtg", "get_cgtg", "gaussians", "get_gaussians",
    "aslist", "to_list", "tolist", "to_vector", "vector",
    # possible pair accessors
    "coefficient", "get_coefficient", "alpha", "get_alpha", "zeta", "get_zeta",
    # possible conversion helpers
    "fit", "form_f12", "f12", "correlation_factor",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--basis", default="6-31g", help="Basis for final ao_f12(corr) smoke test.")
    p.add_argument("--timeout", type=float, default=8.0, help="Timeout per child probe in seconds.")
    p.add_argument("--memory", default="1 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--out-json", default="step3b_fittedslater_corr_probe.json")
    p.add_argument("--summary", default="step3b_fittedslater_corr_probe_summary.txt")
    return p.parse_args()


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


def base_code(gamma: float, basis: str, memory: str, nthreads: int) -> str:
    return f"""
import json
import numpy as np
import psi4

psi4.core.clean()
psi4.set_memory({memory!r})
psi4.set_num_threads({nthreads})
psi4.core.set_output_file("psi4_step3b_child.out", False)

gamma = {gamma!r}
cf = psi4.core.FittedSlaterCorrelationFactor(gamma)

def compact_repr(x, maxlen=300):
    try:
        s = repr(x)
    except Exception as e:
        s = "<repr failed: " + repr(e) + ">"
    if len(s) > maxlen:
        s = s[:maxlen] + "..."
    return s

def serialize_obj(x):
    out = {{"type": str(type(x)), "repr": compact_repr(x)}}

    # Try numpy conversion
    try:
        arr = np.asarray(x)
        out["np_shape"] = list(arr.shape)
        out["np_dtype"] = str(arr.dtype)
        if arr.size <= 100:
            out["np_values"] = arr.tolist()
        else:
            out["np_first10"] = arr.reshape(-1)[:10].tolist()
    except Exception as e:
        out["np_error"] = repr(e)

    # Try Python list conversion
    try:
        lx = list(x)
        out["list_len"] = len(lx)
        out["list_preview"] = [compact_repr(v, 120) for v in lx[:10]]
    except Exception as e:
        out["list_error"] = repr(e)

    # Try common scalar conversions
    for conv_name, conv in [("float", float), ("int", int)]:
        try:
            out[conv_name] = conv(x)
        except Exception as e:
            out[conv_name + "_error"] = repr(e)

    return out

def looks_like_corr_list(obj):
    try:
        lst = list(obj)
    except Exception:
        return False, None
    pairs = []
    for item in lst:
        try:
            if len(item) != 2:
                return False, None
            c = float(item[0])
            a = float(item[1])
            pairs.append((c, a))
        except Exception:
            return False, None
    return len(pairs) > 0, pairs

def psi4_vector_to_list(v):
    # Psi4 Vector usually supports len and get(i), but builds differ.
    vals = []
    try:
        n = len(v)
    except Exception:
        try:
            n = v.dim()
        except Exception:
            try:
                n = v.np.shape[0]
            except Exception:
                return None
    for i in range(int(n)):
        try:
            vals.append(float(v[i]))
        except Exception:
            try:
                vals.append(float(v.get(i)))
            except Exception:
                return None
    return vals
"""


def dir_probe_code(gamma: float, basis: str, memory: str, nthreads: int) -> str:
    return base_code(gamma, basis, memory, nthreads) + """
names = [n for n in dir(cf) if not n.startswith("_")]
result = {
    "class": str(type(cf)),
    "repr": compact_repr(cf),
    "public_names": names,
    "f12_related_names": [n for n in names if any(k in n.lower() for k in ["coef", "exp", "gauss", "fit", "term", "prim", "cgtg", "vector", "list", "f12", "corr"])],
}
print(json.dumps(result, indent=2))
"""


def attr_probe_code(gamma: float, basis: str, memory: str, nthreads: int, attr: str, call: bool) -> str:
    action = "call" if call else "get"
    return base_code(gamma, basis, memory, nthreads) + f"""
result = {{"attr": {attr!r}, "action": {action!r}}}
try:
    obj = getattr(cf, {attr!r})
    result["getattr_ok"] = True
    result["attr_type"] = str(type(obj))
    result["attr_repr"] = compact_repr(obj)
except Exception as e:
    result["getattr_ok"] = False
    result["getattr_error"] = repr(e)
    print(json.dumps(result, indent=2))
    raise SystemExit(1)

if {call!r}:
    try:
        val = obj()
        result["call_ok"] = True
        result["value"] = serialize_obj(val)
        is_corr, pairs = looks_like_corr_list(val)
        result["looks_like_corr_list"] = is_corr
        if pairs is not None:
            result["corr_pairs"] = pairs
    except Exception as e:
        result["call_ok"] = False
        result["call_error"] = repr(e)
        print(json.dumps(result, indent=2))
        raise SystemExit(2)
else:
    result["value"] = serialize_obj(obj)
    is_corr, pairs = looks_like_corr_list(obj)
    result["looks_like_corr_list"] = is_corr
    if pairs is not None:
        result["corr_pairs"] = pairs

print(json.dumps(result, indent=2))
"""


def indexed_pair_probe_code(gamma: float, basis: str, memory: str, nthreads: int, coeff_method: str, exp_method: str, n: int) -> str:
    return base_code(gamma, basis, memory, nthreads) + f"""
result = {{"coeff_method": {coeff_method!r}, "exp_method": {exp_method!r}, "n": {n}}}
pairs = []
try:
    cm = getattr(cf, {coeff_method!r})
    em = getattr(cf, {exp_method!r})
except Exception as e:
    result["getattr_error"] = repr(e)
    print(json.dumps(result, indent=2))
    raise SystemExit(1)

try:
    for i in range({n}):
        c = float(cm(i))
        a = float(em(i))
        pairs.append((c, a))
    result["ok"] = True
    result["corr_pairs"] = pairs
except Exception as e:
    result["ok"] = False
    result["error"] = repr(e)
    print(json.dumps(result, indent=2))
    raise SystemExit(2)

print(json.dumps(result, indent=2))
"""


def corr_from_vectors_probe_code(gamma: float, basis: str, memory: str, nthreads: int, coeff_attr: str, exp_attr: str) -> str:
    return base_code(gamma, basis, memory, nthreads) + f"""
result = {{"coeff_attr": {coeff_attr!r}, "exp_attr": {exp_attr!r}}}
try:
    cv = getattr(cf, {coeff_attr!r})
    ev = getattr(cf, {exp_attr!r})
    if callable(cv):
        cv = cv()
    if callable(ev):
        ev = ev()
    coeffs = psi4_vector_to_list(cv)
    exps = psi4_vector_to_list(ev)
    result["coeffs"] = coeffs
    result["exps"] = exps
    if coeffs is None or exps is None or len(coeffs) != len(exps) or len(coeffs) == 0:
        result["ok"] = False
        result["error"] = "could not extract matching coefficient/exponent vectors"
        print(json.dumps(result, indent=2))
        raise SystemExit(2)
    result["ok"] = True
    result["corr_pairs"] = list(zip(coeffs, exps))
except Exception as e:
    result["ok"] = False
    result["error"] = repr(e)
    print(json.dumps(result, indent=2))
    raise SystemExit(1)

print(json.dumps(result, indent=2))
"""


def ao_f12_corr_test_code(gamma: float, basis: str, memory: str, nthreads: int, corr_pairs: List[List[float]]) -> str:
    # corr_pairs inserted as a literal list of tuples.
    corr_literal = repr([tuple(map(float, p)) for p in corr_pairs])
    return base_code(gamma, basis, memory, nthreads) + f"""
result = {{"basis": {basis!r}, "corr": {corr_literal}}}
try:
    mol = psi4.geometry(\"\"\"
0 1
He 0.0 0.0 0.0
symmetry c1
units bohr
\"\"\")
    bs = psi4.core.BasisSet.build(mol, "BASIS", {basis!r})
    mints = psi4.core.MintsHelper(bs)
    val = mints.ao_f12({corr_literal})
    arr = np.asarray(val)
    result["call_ok"] = True
    result["shape"] = list(arr.shape)
    result["size"] = int(arr.size)
    result["norm"] = float(np.linalg.norm(arr.reshape(-1))) if arr.size else 0.0
    result["max_abs"] = float(np.max(np.abs(arr))) if arr.size else 0.0
except Exception as e:
    result["call_ok"] = False
    result["error"] = repr(e)
    print(json.dumps(result, indent=2))
    raise SystemExit(2)

print(json.dumps(result, indent=2))
"""


def candidate_attrs_from_dir(public_names: List[str]) -> List[str]:
    names = []
    keys = ["coef", "coeff", "exp", "gauss", "fit", "term", "prim", "cgtg", "vector", "list", "f12", "corr", "alpha", "zeta"]
    for n in public_names:
        if any(k in n.lower() for k in keys):
            names.append(n)
    for n in COMMON_METHOD_CANDIDATES:
        if n not in names:
            names.append(n)
    return sorted(set(names))


def parse_payload(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return extract_json(rec.get("stdout", ""))


def find_corr_pairs_from_payloads(records: List[Dict[str, Any]]) -> Optional[List[List[float]]]:
    for rec in records:
        payload = parse_payload(rec)
        if not payload:
            continue
        pairs = payload.get("corr_pairs")
        if pairs and isinstance(pairs, list):
            try:
                return [[float(x[0]), float(x[1])] for x in pairs]
            except Exception:
                pass
    return None


def main():
    args = parse_args()
    report: Dict[str, Any] = {
        "gamma": args.gamma,
        "basis": args.basis,
        "timeout_s": args.timeout,
        "dir_probe": None,
        "attribute_probes": [],
        "vector_pair_probes": [],
        "indexed_pair_probes": [],
        "ao_f12_corr_test": None,
    }

    print("=" * 80)
    print("Step 3b | Extract FittedSlaterCorrelationFactor Gaussian expansion")
    print("=" * 80)
    print(f"gamma   = {args.gamma}")
    print(f"basis   = {args.basis}")
    print(f"timeout = {args.timeout} s per probe")

    print("\n[1] dir(cf)")
    rec = run_child(
        "dir(cf)",
        dir_probe_code(args.gamma, args.basis, args.memory, args.nthreads),
        args.timeout,
    )
    report["dir_probe"] = rec
    payload = parse_payload(rec)
    if not rec["ok"] or payload is None:
        print("FAILED to inspect dir(cf).")
        print(rec.get("stdout", ""))
        print(rec.get("stderr", ""))
        write_outputs(args, report)
        return

    public_names = payload.get("public_names", [])
    related = payload.get("f12_related_names", [])
    print(f"public method/property count = {len(public_names)}")
    print("F12/fit/coef/exp related names:")
    for n in related:
        print("  ", n)

    attrs = candidate_attrs_from_dir(public_names)

    print("\n[2] zero-argument attribute/method probes")
    for attr in attrs:
        # Probe as property/getattr first if it exists.
        if attr not in public_names:
            continue

        get_rec = run_child(
            f"getattr cf.{attr}",
            attr_probe_code(args.gamma, args.basis, args.memory, args.nthreads, attr, call=False),
            args.timeout,
        )
        report["attribute_probes"].append(get_rec)

        get_payload = parse_payload(get_rec)
        callable_attr = False
        if get_payload:
            callable_attr = "method" in get_payload.get("attr_type", "").lower() or "instancemethod" in get_payload.get("attr_type", "").lower()
            if get_payload.get("looks_like_corr_list"):
                print(f"FOUND possible corr list from property cf.{attr}")
                corr_pairs = get_payload.get("corr_pairs")
                test_ao_f12(args, report, corr_pairs)
                write_outputs(args, report)
                return

        status = "OK" if get_rec["ok"] else ("TIMEOUT" if get_rec.get("timeout") else "FAIL")
        print(f"{status:<8s} getattr cf.{attr}")

        # If callable, try zero-arg call.
        if callable_attr:
            call_rec = run_child(
                f"call cf.{attr}()",
                attr_probe_code(args.gamma, args.basis, args.memory, args.nthreads, attr, call=True),
                args.timeout,
            )
            report["attribute_probes"].append(call_rec)
            call_payload = parse_payload(call_rec)
            status = "OK" if call_rec["ok"] else ("TIMEOUT" if call_rec.get("timeout") else "FAIL")
            print(f"{status:<8s} call    cf.{attr}()")
            if call_payload and call_payload.get("looks_like_corr_list"):
                print(f"FOUND possible corr list from cf.{attr}()")
                corr_pairs = call_payload.get("corr_pairs")
                test_ao_f12(args, report, corr_pairs)
                write_outputs(args, report)
                return

    print("\n[3] vector-pair extraction attempts")
    coeff_names = [n for n in public_names if "coef" in n.lower() or "coeff" in n.lower()]
    exp_names = [n for n in public_names if "exp" in n.lower() or "alpha" in n.lower() or "zeta" in n.lower()]
    for cn in coeff_names:
        for en in exp_names:
            rec = run_child(
                f"vector-pair {cn}/{en}",
                corr_from_vectors_probe_code(args.gamma, args.basis, args.memory, args.nthreads, cn, en),
                args.timeout,
            )
            report["vector_pair_probes"].append(rec)
            status = "OK" if rec["ok"] else ("TIMEOUT" if rec.get("timeout") else "FAIL")
            print(f"{status:<8s} {cn} + {en}")
            payload = parse_payload(rec)
            if payload and payload.get("ok") and payload.get("corr_pairs"):
                print(f"FOUND possible corr list from {cn} and {en}")
                test_ao_f12(args, report, payload["corr_pairs"])
                write_outputs(args, report)
                return

    print("\n[4] indexed coefficient/exponent attempts")
    coeff_index_methods = [n for n in public_names if any(k in n.lower() for k in ["coef", "coeff"])]
    exp_index_methods = [n for n in public_names if any(k in n.lower() for k in ["exp", "alpha", "zeta"])]
    for cn in coeff_index_methods:
        for en in exp_index_methods:
            for ntrial in [3, 6, 10]:
                rec = run_child(
                    f"indexed {cn}/{en}/{ntrial}",
                    indexed_pair_probe_code(args.gamma, args.basis, args.memory, args.nthreads, cn, en, ntrial),
                    args.timeout,
                )
                report["indexed_pair_probes"].append(rec)
                status = "OK" if rec["ok"] else ("TIMEOUT" if rec.get("timeout") else "FAIL")
                print(f"{status:<8s} {cn}(i) + {en}(i), n={ntrial}")
                payload = parse_payload(rec)
                if payload and payload.get("ok") and payload.get("corr_pairs"):
                    print(f"FOUND possible corr list from indexed {cn}/{en}, n={ntrial}")
                    test_ao_f12(args, report, payload["corr_pairs"])
                    write_outputs(args, report)
                    return

    print("\nNo Gaussian expansion list was found automatically.")
    write_outputs(args, report)


def test_ao_f12(args, report: Dict[str, Any], corr_pairs):
    print("\n[5] Testing mints.ao_f12(corr)")
    rec = run_child(
        "mints.ao_f12(corr)",
        ao_f12_corr_test_code(args.gamma, args.basis, args.memory, args.nthreads, corr_pairs),
        args.timeout,
    )
    report["ao_f12_corr_test"] = rec
    status = "OK" if rec["ok"] else ("TIMEOUT" if rec.get("timeout") else "FAIL")
    print(f"{status:<8s} mints.ao_f12(corr)")
    payload = parse_payload(rec)
    if payload:
        print(textwrap.indent(json.dumps(payload, indent=2), "    "))
    elif rec.get("stderr"):
        print(textwrap.indent(rec["stderr"].splitlines()[-1], "    "))


def write_outputs(args, report: Dict[str, Any]):
    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.summary).write_text(make_summary(report), encoding="utf-8")
    print("\n[Saved]")
    print(" ", args.out_json)
    print(" ", args.summary)


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


def make_summary(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("Step 3b summary")
    lines.append("=" * 80)
    lines.append(f"gamma = {report.get('gamma')}")
    lines.append(f"basis = {report.get('basis')}")
    lines.append(f"timeout_s = {report.get('timeout_s')}")
    lines.append("")

    dir_payload = parse_payload(report.get("dir_probe", {}) or {})
    if dir_payload:
        lines.append("[F12-related public names]")
        for n in dir_payload.get("f12_related_names", []):
            lines.append(f"  {n}")
        lines.append("")

    def summarize_records(title: str, records: List[Dict[str, Any]], max_show: int = 80):
        lines.append(title)
        shown = 0
        for rec in records:
            if shown >= max_show:
                lines.append(f"  ... omitted {len(records)-shown} records")
                break
            status = "OK" if rec.get("ok") else ("TIMEOUT" if rec.get("timeout") else "FAIL")
            lines.append(f"  {status:<8s} {rec.get('label')}")
            payload = parse_payload(rec)
            if payload:
                if payload.get("looks_like_corr_list") or payload.get("corr_pairs"):
                    lines.append(f"    corr_pairs = {payload.get('corr_pairs')}")
                elif payload.get("call_ok"):
                    lines.append(f"    value = {json.dumps(payload.get('value', {}), ensure_ascii=False)[:500]}")
                elif payload.get("call_error"):
                    lines.append(f"    call_error = {payload.get('call_error')}")
            shown += 1
        lines.append("")

    summarize_records("[Attribute probes]", report.get("attribute_probes", []))
    summarize_records("[Vector pair probes]", report.get("vector_pair_probes", []))
    summarize_records("[Indexed pair probes]", report.get("indexed_pair_probes", []))

    lines.append("[ao_f12(corr) test]")
    rec = report.get("ao_f12_corr_test")
    if rec is None:
        lines.append("  not performed")
    else:
        status = "OK" if rec.get("ok") else ("TIMEOUT" if rec.get("timeout") else "FAIL")
        lines.append(f"  {status:<8s} {rec.get('label')}")
        payload = parse_payload(rec)
        if payload:
            lines.append(f"  payload = {json.dumps(payload, ensure_ascii=False)}")
    lines.append("")

    lines.append("[Interpretation]")
    if rec and rec.get("ok"):
        lines.append("A valid corr list was extracted and ao_f12(corr) succeeded.")
        lines.append("Next: identify ao_f12 tensor flattening/order and transform F12 integrals to OBS/CABS+ basis.")
    else:
        lines.append("No complete corr -> ao_f12 path was found automatically.")
        lines.append("Next: inspect the public names in the summary and Psi4 source for FittedSlaterCorrelationFactor.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
