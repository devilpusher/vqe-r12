PYTHON ?= python
CONDA_ENV ?= vqecodex

SCRIPTS := \
	step1_psi4_he_detci_rdm_check_v2.py \
	step2_he_cabs_plus_check.py \
	step3_he_f12_integral_probe_v2.py \
	step3b_extract_fittedslater_corr.py \
	step3c_manual_corr_ao_f12_smoke.py \
	step4_he_parent_f12_transform_check.py \
	step4b_he_parent_obs_fci_rdm_check.py \
	step5a_he_r12_intermediate_check.py \
	step5b_he_r12_prototype_correction.py \
	step5c_he_correction_comparison.py \
	step6a_fit_slater_corr.py \
	step6b_collect_fit_convergence.py \
	step6b_scan_slater_pipeline.py \
	step6c_he_r12_approxc_sp_contraction.py \
	step6d_formula_projector_audit.py \
	step6e_build_vxbc_intermediates.py

.PHONY: env update-env check step1 step2 step3 step4b step5a step5b step5c step6a step6b step6c step6d step6e clean

env:
	conda env create -f environment.yml

update-env:
	conda env update -n $(CONDA_ENV) -f environment.yml --prune

check:
	$(PYTHON) -m py_compile $(SCRIPTS)

step1:
	$(PYTHON) step1_psi4_he_detci_rdm_check_v2.py

step2:
	$(PYTHON) step2_he_cabs_plus_check.py

step3:
	$(PYTHON) step3_he_f12_integral_probe_v2.py

step4b:
	$(PYTHON) step4b_he_parent_obs_fci_rdm_check.py

step5a:
	$(PYTHON) step5a_he_r12_intermediate_check.py

step5b:
	$(PYTHON) step5b_he_r12_prototype_correction.py

step5c:
	$(PYTHON) step5c_he_correction_comparison.py

step6a:
	$(PYTHON) step6a_fit_slater_corr.py

step6b:
	$(PYTHON) step6b_collect_fit_convergence.py

step6c:
	$(PYTHON) step6c_he_r12_approxc_sp_contraction.py

step6d:
	$(PYTHON) step6d_formula_projector_audit.py

step6e:
	$(PYTHON) step6e_build_vxbc_intermediates.py

clean:
	rm -f *.npz *.npy *.out *_summary.txt *.csv step6a_slater_fit_N*.json step6b*_fit_convergence.json step6b*_fit_convergence.txt step6b_slater_scan.json step6b_slater_scan.txt
