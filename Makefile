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
	step5c_he_correction_comparison.py

.PHONY: env update-env check step1 step2 step3 step4b step5a step5b step5c clean

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

clean:
	rm -f *.npz *.npy *.out *_summary.txt *.csv

