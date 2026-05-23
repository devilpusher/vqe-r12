PYTHON ?= python
CONDA_ENV ?= vqecodex

SCRIPTS := \
	r12_common.py \
	r12_correction.py \
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
	step6e_build_vxbc_intermediates.py \
	step6f_he_r12_candidate_energy.py \
	step6g_audit_approxc_terms.py \
	step6h_correlation_factor_sign_probe.py \
	step6i_audit_sp_normalization.py \
	step6j_audit_closed_shell_sp_factors.py \
	step6k_audit_paper_tequila_sf2r12.py \
	step6l_scan_paper_tequila_convergence.py \
	step6m_he_r12_correction_pipeline.py \
	step7a_export_ecg_no_data.py \
	step7b_export_ecg_no_orbitals.py \
	step7c_build_ecg_no_step4b_like.py \
	step7d_ecg_no_r12_correction.py \
	step7e_scan_ecg_no_r12_convergence.py \
	step7f_audit_ecg_no_r12_projectors.py \
	step7g_audit_ecg_no_r12_subterms.py \
	step7h_dual_space_projector_prototype.py \
	step7i_residual_weighted_dual_space.py \
	step7j_scan_residual_weights.py \
	step7k_refscale_weighted_dual_core.py \
	step7l_refscale_sensitivity.py \
	step7m_strict_projector_partition_audit.py \
	step7n_fractional_occupation_projector_audit.py \
	step7o_tensor_fractional_projector_audit.py

.PHONY: env update-env check step1 step2 step3 step4b step5a step5b step5c step6a step6b step6c step6d step6e step6f step6g step6h step6i step6j step6k step6l step6m step7a step7b step7c step7d step7e step7f step7g step7h step7i step7j step7k step7l step7m step7n step7o clean

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

step6f:
	$(PYTHON) step6f_he_r12_candidate_energy.py

step6g:
	$(PYTHON) step6g_audit_approxc_terms.py

step6h:
	$(PYTHON) step6h_correlation_factor_sign_probe.py

step6i:
	$(PYTHON) step6i_audit_sp_normalization.py

step6j:
	$(PYTHON) step6j_audit_closed_shell_sp_factors.py

step6k:
	$(PYTHON) step6k_audit_paper_tequila_sf2r12.py

step6l:
	$(PYTHON) step6l_scan_paper_tequila_convergence.py

step6m:
	$(PYTHON) step6m_he_r12_correction_pipeline.py

step7a:
	$(PYTHON) step7a_export_ecg_no_data.py

step7b:
	$(PYTHON) step7b_export_ecg_no_orbitals.py

step7c:
	$(PYTHON) step7c_build_ecg_no_step4b_like.py

step7d:
	$(PYTHON) step7d_ecg_no_r12_correction.py

step7e:
	$(PYTHON) step7e_scan_ecg_no_r12_convergence.py

step7f:
	$(PYTHON) step7f_audit_ecg_no_r12_projectors.py

step7g:
	$(PYTHON) step7g_audit_ecg_no_r12_subterms.py

step7h:
	$(PYTHON) step7h_dual_space_projector_prototype.py

step7i:
	$(PYTHON) step7i_residual_weighted_dual_space.py

step7j:
	$(PYTHON) step7j_scan_residual_weights.py

step7k:
	$(PYTHON) step7k_refscale_weighted_dual_core.py

step7l:
	$(PYTHON) step7l_refscale_sensitivity.py

step7m:
	$(PYTHON) step7m_strict_projector_partition_audit.py

step7n:
	$(PYTHON) step7n_fractional_occupation_projector_audit.py

step7o:
	$(PYTHON) step7o_tensor_fractional_projector_audit.py

clean:
	rm -f *.npz *.npy *.out *_summary.txt *.csv step6a_slater_fit_N*.json step6b*_fit_convergence.json step6b*_fit_convergence.txt step6b_slater_scan.json step6b_slater_scan.txt step6l_*_scan.json step7a_*_export.json step7b_*.json step7c_*.json step7d_*.json step7e_*.json step7f_*.json step7g_*.json step7h_*.json step7i_*.json step7j_*.json step7k_*.json step7l_*.json step7m_*.json step7n_*.json step7o_*.json
