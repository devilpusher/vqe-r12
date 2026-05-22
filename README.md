# VQE / R12 Helium Prototype Checks

This repository contains a staged set of Python scripts for validating helium
RDM, CABS+, F12/R12 integral, and prototype correction workflows.

## Scripts

- `r12_common.py` contains shared parent-basis/RDM/tensor utilities.
- `r12_correction.py` contains the production-facing He SF-[2]R12 correction API.
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
- `step6f_he_r12_candidate_energy.py` writes a He-only candidate [2]R12 energy ledger from direct F12 tensors.
- `step6g_audit_approxc_terms.py` audits approximation-C/SP tilde terms, denominators, and prefactor variants.
- `step6h_correlation_factor_sign_probe.py` checks correlation-factor sign conventions.
- `step6i_audit_sp_normalization.py` audits SP prefactors and closed-shell pair normalization.
- `step6j_audit_closed_shell_sp_factors.py` audits He spin-free pair counting and SP factor budgets.
- `step6k_audit_paper_tequila_sf2r12.py` maps the paper/Tequila SF-[2]R12 contractions onto the He tensors.
- `step6l_scan_paper_tequila_convergence.py` scans the audited `paper_tequila_sf2r12` row over fitted-Slater sizes and parent/OBS cases.
- `step6m_he_r12_correction_pipeline.py` runs the formal He parent-basis SF-[2]R12 correction pipeline.
- `step7a_export_ecg_no_data.py` exports local ECG-NO alpha-beta geminal data as Step-compatible spin-free RDMs.
- `step7b_export_ecg_no_orbitals.py` rebuilds ECG-NO `C_obs` and audits PySCF/Psi4 AO basis ordering.

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
python step6m_he_r12_correction_pipeline.py --parent-basis cc-pvdz --nobs 2 --fitN 7
```

The command above is the current clean entry point for the stabilized He
parent-basis route. It fits a local Gaussian expansion of the Slater factor,
runs Step 4b and Step 5a as needed, then emits the selected
`paper_tequila_sf2r12` correction only.

The older staged checks remain available for auditing and regression work:

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
python step6f_he_r12_candidate_energy.py
python step6g_audit_approxc_terms.py
python step6h_correlation_factor_sign_probe.py
python step6i_audit_sp_normalization.py
python step6j_audit_closed_shell_sp_factors.py
python step6k_audit_paper_tequila_sf2r12.py
python step6l_scan_paper_tequila_convergence.py
python step6m_he_r12_correction_pipeline.py
python step7a_export_ecg_no_data.py
python step7b_export_ecg_no_orbitals.py
```

Generated `.npz`, `.out`, summary, and comparison files are ignored by Git.
