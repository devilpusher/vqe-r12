#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 v2: SAFE F12 interface probe for He.

This version avoids the slow part of the previous probe.  By default it only:

1. verifies ordinary mixed-basis overlap/ERI calls;
2. lists Psi4 core/MintsHelper names related to F12/correlation factors;
3. tries to inspect method signatures;
4. optionally probes Tequila attributes.

It does NOT brute-force call every possible F12 integral signature unless
--deep is explicitly given.

Usage
-----
    python step3_he_f12_integral_probe_v2.py

Optional:
    python step3_he_f12_integral_probe_v2.py --probe-tequila
    python step3_he_f12_integral_probe_v2.py --deep

If --deep is used, several actual F12 calls are attempted and may be slow or
fail depending on the Psi4 build.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step2", default="he_631g_obs_ccpvdz_cabsplus.npz")
    p.add_argument("--obs-basis", default="6-31g")
    p.add_argument("--cabs-basis", default="cc-pvdz")
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--memory", default="2 GB")
    p.add_argument("--nthreads", type=int, default=1)
    p.add_argument("--psi4-output", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--probe-tequila", action="store_true")
    p.add_argument("--deep", action="store_true", help="Actually attempt F12 integral calls; may be slow.")
    return p.parse_args()


def safe_label(s: str) -> str:
    return s.lower().replace("*", "s").replace("+", "p").replace("-", "").replace("_", "")


def asarray_psi4_matrix(x):
    return np.array(np.asarray(x), dtype=float, copy=True)


def arr_stats(x: Any) -> Dict[str, Any]:
    arr = asarray_psi4_matrix(x)
    return {
        "shape": list(arr.shape),
        "size": int(arr.size),
        "norm": float(np.linalg.norm(arr.reshape(-1))) if arr.size else 0.0,
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
    }


def try_call(label: str, func, *args) -> Dict[str, Any]:
    try:
        val = func(*args)
        out = {"label": label, "ok": True}
        out.update(arr_stats(val))
        return out
    except Exception as exc:
        return {"label": label, "ok": False, "error": repr(exc)}


def build_psi4_objects(obs_basis: str, cabs_basis: str, memory: str, nthreads: int, output_file: str):
    try:
        import psi4
    except Exception as exc:
        raise RuntimeError("Cannot import psi4. Activate the Psi4 environment first.") from exc

    psi4.core.clean()
    psi4.set_memory(memory)
    psi4.set_num_threads(nthreads)
    psi4.core.set_output_file(output_file, False)

    mol = psi4.geometry(
        """
        0 1
        He 0.0 0.0 0.0
        symmetry c1
        units bohr
        """
    )

    obs_bs = psi4.core.BasisSet.build(mol, "BASIS", obs_basis)
    cabs_bs = psi4.core.BasisSet.build(mol, "BASIS", cabs_basis)
    mints = psi4.core.MintsHelper(obs_bs)
    return psi4, mol, obs_bs, cabs_bs, mints


def list_methods(obj: Any, keys: Tuple[str, ...]) -> List[str]:
    names = []
    for name in dir(obj):
        low = name.lower()
        if any(k in low for k in keys):
            names.append(name)
    return sorted(set(names))


def signature_or_error(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:
        return f"<signature unavailable: {repr(exc)}>"


def load_step2_metadata(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": path, "exists": Path(path).exists()}
    if not out["exists"]:
        return out
    try:
        data = np.load(path, allow_pickle=True)
        out["keys"] = list(data.keys())
        for k in ["S_block", "C_obs_block", "C_cabs_block", "C_ri_block"]:
            if k in data:
                out[k + "_shape"] = list(np.asarray(data[k]).shape)
        if "metadata_json" in data:
            out["metadata"] = json.loads(str(data["metadata_json"]))
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def ordinary_integral_probe(mints, obs_bs, cabs_bs) -> List[Dict[str, Any]]:
    calls = [
        ("ao_overlap(obs,obs)", mints.ao_overlap, (obs_bs, obs_bs)),
        ("ao_overlap(obs,cabs)", mints.ao_overlap, (obs_bs, cabs_bs)),
        ("ao_overlap(cabs,cabs)", mints.ao_overlap, (cabs_bs, cabs_bs)),
        ("ao_eri(obs,obs,obs,obs)", mints.ao_eri, (obs_bs, obs_bs, obs_bs, obs_bs)),
        ("ao_eri(obs,obs,cabs,cabs)", mints.ao_eri, (obs_bs, obs_bs, cabs_bs, cabs_bs)),
        ("ao_eri(obs,cabs,obs,cabs)", mints.ao_eri, (obs_bs, cabs_bs, obs_bs, cabs_bs)),
    ]
    return [try_call(label, func, *args) for label, func, args in calls]


def constructor_probe(psi4, gamma: float, deep: bool = False) -> List[Dict[str, Any]]:
    core = psi4.core
    candidate_names = [
        "FittedSlaterCorrelationFactor",
        "SlaterCorrelationFactor",
        "GaussianCorrelationFactor",
        "CorrelationFactor",
        "F12CorrelationFactor",
    ]
    records = []
    for name in candidate_names:
        rec: Dict[str, Any] = {"name": name, "exists": hasattr(core, name)}
        if not rec["exists"]:
            records.append(rec)
            continue
        cls = getattr(core, name)
        rec["signature"] = signature_or_error(cls)
        rec["constructors_tried"] = []
        if deep:
            for args in [(gamma,), (), ([1.0], [gamma]), (np.array([1.0]), np.array([gamma]))]:
                try:
                    obj = cls(*args)
                    rec["constructors_tried"].append({"args": repr(args), "ok": True, "repr": repr(obj)})
                except Exception as exc:
                    rec["constructors_tried"].append({"args": repr(args), "ok": False, "error": repr(exc)})
        records.append(rec)
    return records


def f12_method_probe(mints, psi4, obs_bs, cabs_bs, gamma: float, deep: bool = False) -> List[Dict[str, Any]]:
    method_names = list_methods(mints, ("f12",))
    records = []
    for name in method_names:
        func = getattr(mints, name)
        rec: Dict[str, Any] = {
            "method": name,
            "signature": signature_or_error(func),
            "actual_calls": [],
        }
        if deep:
            # Very limited calls only; the old script tried too many signatures.
            # Many Psi4 builds will fail these, which is still useful information.
            factors = []
            for cname in ["FittedSlaterCorrelationFactor", "SlaterCorrelationFactor", "GaussianCorrelationFactor"]:
                if hasattr(psi4.core, cname):
                    cls = getattr(psi4.core, cname)
                    for args in [(gamma,), ()]:
                        try:
                            factors.append((f"{cname}{args}", cls(*args)))
                            break
                        except Exception:
                            pass

            for flabel, fac in factors[:2]:
                for args in [(fac,), (obs_bs, obs_bs, obs_bs, obs_bs, fac)]:
                    rec["actual_calls"].append(try_call(f"{name} {flabel} nargs={len(args)}", func, *args))
        records.append(rec)
    return records


def erf_method_probe(mints) -> List[Dict[str, Any]]:
    records = []
    for name in list_methods(mints, ("erf", "erfc", "omega")):
        func = getattr(mints, name)
        records.append({"method": name, "signature": signature_or_error(func)})
    return records


def tequila_probe(obs_basis: str, cabs_basis: str, gamma: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import tequila as tq
        out["import_ok"] = True
    except Exception as exc:
        out["import_ok"] = False
        out["error"] = repr(exc)
        return out
    try:
        mol = tq.Molecule(geometry="He 0.0 0.0 0.0", basis_set=obs_basis, point_group="c1")
        out["molecule_created"] = True
        out["molecule_class"] = str(type(mol))
        out["has_perturbative_f12_correction"] = hasattr(mol, "perturbative_f12_correction")
        out["f12_like_attrs"] = [
            name for name in dir(mol)
            if "f12" in name.lower() or "r12" in name.lower() or "cabs" in name.lower()
        ]
        if hasattr(mol, "perturbative_f12_correction"):
            out["perturbative_f12_signature"] = signature_or_error(mol.perturbative_f12_correction)
    except Exception as exc:
        out["molecule_created"] = False
        out["molecule_error"] = repr(exc)
    return out


def make_summary(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("Step 3 v2 | SAFE F12 interface probe | He")
    lines.append("=" * 80)

    lines.append("\n[Environment]")
    lines.append(f"Psi4 version     = {report.get('psi4_version')}")
    lines.append(f"OBS basis        = {report.get('obs_basis')}")
    lines.append(f"CABS basis       = {report.get('cabs_basis')}")
    lines.append(f"gamma            = {report.get('gamma')}")
    lines.append(f"nao_obs          = {report.get('nao_obs')}")
    lines.append(f"nao_raw_cabs     = {report.get('nao_raw_cabs')}")
    lines.append(f"deep             = {report.get('deep')}")

    lines.append("\n[Step2]")
    step2 = report.get("step2", {})
    lines.append(f"exists           = {step2.get('exists')}")
    if "metadata" in step2:
        m = step2["metadata"]
        lines.append(f"nobs/ncabs/nri   = {m.get('nobs')} / {m.get('ncabs')} / {m.get('nri')}")
        lines.append(f"orth errors      = obs {m.get('obs_orth_error')}, cabs {m.get('cabs_orth_error')}, cross {m.get('obs_cabs_cross_error')}")

    lines.append("\n[Ordinary mixed-basis integral calls]")
    for rec in report.get("ordinary_integrals", []):
        if rec["ok"]:
            lines.append(f"OK   {rec['label']:<32s} shape={rec['shape']} norm={rec['norm']:.6e}")
        else:
            lines.append(f"FAIL {rec['label']:<32s} {rec['error']}")

    lines.append("\n[Relevant Psi4 core names]")
    for name in report.get("core_relevant_names", []):
        lines.append(f"  {name}")

    lines.append("\n[MintsHelper F12-like methods]")
    if not report.get("f12_methods"):
        lines.append("  none found")
    for rec in report.get("f12_methods", []):
        lines.append(f"  {rec['method']} signature={rec.get('signature')}")
        for call in rec.get("actual_calls", []):
            if call["ok"]:
                lines.append(f"    OK   {call['label']} shape={call['shape']} norm={call['norm']:.6e}")
            else:
                lines.append(f"    FAIL {call['label']} {call['error']}")

    lines.append("\n[Correlation factor constructors]")
    for rec in report.get("factor_constructors", []):
        lines.append(f"  {rec['name']}: exists={rec['exists']} signature={rec.get('signature')}")
        for c in rec.get("constructors_tried", []):
            status = "OK" if c["ok"] else "FAIL"
            lines.append(f"    {status} args={c['args']}" + (f" {c.get('error')}" if not c["ok"] else ""))

    lines.append("\n[ERF/range-separated methods]")
    if not report.get("erf_methods"):
        lines.append("  none found")
    for rec in report.get("erf_methods", []):
        lines.append(f"  {rec['method']} signature={rec.get('signature')}")

    if "tequila" in report:
        tq = report["tequila"]
        lines.append("\n[Tequila]")
        lines.append(f"import_ok        = {tq.get('import_ok')}")
        lines.append(f"molecule_created = {tq.get('molecule_created')}")
        lines.append(f"has f12 corr     = {tq.get('has_perturbative_f12_correction')}")
        lines.append(f"f12-like attrs   = {tq.get('f12_like_attrs')}")
        if "perturbative_f12_signature" in tq:
            lines.append(f"signature        = {tq['perturbative_f12_signature']}")
        if "error" in tq:
            lines.append(f"error            = {tq['error']}")
        if "molecule_error" in tq:
            lines.append(f"molecule_error   = {tq['molecule_error']}")

    f12_call_ok = any(
        any(call.get("ok") for call in rec.get("actual_calls", []))
        for rec in report.get("f12_methods", [])
    )
    lines.append("\n[Interpretation]")
    if f12_call_ok:
        lines.append("At least one F12 call succeeded in --deep mode. Next: identify convention and transform to RI/CABS+ basis.")
    elif report.get("f12_methods"):
        lines.append("F12-like methods exist, but no actual F12 integral call was attempted or succeeded. Use --deep only if needed.")
    else:
        lines.append("No MintsHelper F12 method was found. Likely next route: inspect Tequila's perturbative_f12_correction or use lower-level Libint/Psi4 F12 code.")
    lines.append("Ordinary mixed-basis ERIs should be sufficient to confirm that OBS/CABS plumbing is working.")

    return "\n".join(lines)


def main():
    args = parse_args()

    obs_label = safe_label(args.obs_basis)
    cabs_label = safe_label(args.cabs_basis)
    if args.psi4_output is None:
        args.psi4_output = f"psi4_he_{obs_label}_{cabs_label}_f12_probe_v2.out"
    if args.out_json is None:
        args.out_json = f"he_{obs_label}_{cabs_label}_f12_probe_v2.json"
    if args.summary is None:
        args.summary = f"he_{obs_label}_{cabs_label}_f12_probe_v2_summary.txt"

    psi4, mol, obs_bs, cabs_bs, mints = build_psi4_objects(
        args.obs_basis, args.cabs_basis, args.memory, args.nthreads, args.psi4_output
    )

    report: Dict[str, Any] = {
        "psi4_version": getattr(psi4, "__version__", "unknown"),
        "obs_basis": args.obs_basis,
        "cabs_basis": args.cabs_basis,
        "gamma": args.gamma,
        "nao_obs": int(obs_bs.nbf()),
        "nao_raw_cabs": int(cabs_bs.nbf()),
        "deep": bool(args.deep),
        "step2": load_step2_metadata(args.step2),
    }

    report["ordinary_integrals"] = ordinary_integral_probe(mints, obs_bs, cabs_bs)
    report["core_relevant_names"] = list_methods(psi4.core, ("f12", "correlation", "slater", "gaussian", "erf"))
    report["f12_methods"] = f12_method_probe(mints, psi4, obs_bs, cabs_bs, args.gamma, deep=args.deep)
    report["factor_constructors"] = constructor_probe(psi4, args.gamma, deep=args.deep)
    report["erf_methods"] = erf_method_probe(mints)

    if args.probe_tequila:
        report["tequila"] = tequila_probe(args.obs_basis, args.cabs_basis, args.gamma)

    summary = make_summary(report)
    print(summary)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n")

    print("\n[Saved]")
    print(" ", args.out_json)
    print(" ", args.summary)


if __name__ == "__main__":
    main()
