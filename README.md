# VQE / R12 Helium Prototype Checks

This repository contains a staged set of Python scripts for validating helium
RDM, CABS+, F12/R12 integral, and prototype correction workflows.

## Scripts

- `step1_psi4_he_detci_rdm_check_v2.py` builds the initial Psi4 DETCI/RDM reference.
- `step2_he_cabs_plus_check.py` constructs and checks the CABS+ space.
- `step3_he_f12_integral_probe_v2.py` probes F12 integral availability and metadata.
- `step3b_extract_fittedslater_corr.py` extracts fitted Slater correlation data.
- `step3c_manual_corr_ao_f12_smoke.py` performs an AO F12 smoke test.
- `step4_he_parent_f12_transform_check.py` validates parent-basis F12 RI transforms.
- `step4b_he_parent_obs_fci_rdm_check.py` checks parent observable-basis FCI/RDM data.
- `step5a_he_r12_intermediate_check.py` builds R12 intermediates.
- `step5b_he_r12_prototype_correction.py` evaluates a prototype R12 correction.
- `step5c_he_correction_comparison.py` compares correction results.
- `step6a_fit_slater_corr.py` fits a Slater-type F12 factor by Gaussian expansion.
- `step6b_collect_fit_convergence.py` runs/collects fitted Slater convergence through the prototype pipeline.
- `step6b_scan_slater_pipeline.py` performs exploratory fit-protocol scans.
- `step6c_he_r12_approxc_sp_contraction.py` evaluates the first approximation-C/SP contraction prototype.
- `step6d_formula_projector_audit.py` records the formula map and audits Ansatz-3 projector subtraction.
- `step6e_build_vxbc_intermediates.py` builds explicit V/X/B/C intermediate matrices for approximation-C diagnostics.

## Environment

The scripts require Python 3.10+ and use `numpy`. Steps that call Psi4 require a
working Psi4 installation, which is usually easiest to install with conda or
mamba:

```powershell
conda env create -f environment.yml
conda activate vqecodex
```

For pip-only workflows, install the minimal Python dependency:

```powershell
python -m pip install -r requirements.txt
```

Psi4 is intentionally kept in `environment.yml` because pip availability varies
by platform and Python version.

## Typical Workflow

Run scripts from the repository root so their default input and output file names
line up:

```powershell
python step1_psi4_he_detci_rdm_check_v2.py
python step2_he_cabs_plus_check.py
python step3_he_f12_integral_probe_v2.py
python step4b_he_parent_obs_fci_rdm_check.py
python step5a_he_r12_intermediate_check.py
python step5b_he_r12_prototype_correction.py
python step5c_he_correction_comparison.py
python step6a_fit_slater_corr.py
python step6b_collect_fit_convergence.py
python step6c_he_r12_approxc_sp_contraction.py
python step6d_formula_projector_audit.py
python step6e_build_vxbc_intermediates.py
```

Generated `.npz`, `.out`, summary, and comparison files are ignored by Git.
