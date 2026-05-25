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
	step7o_tensor_fractional_projector_audit.py \
	step7p_space_aware_fractional_projector.py \
	step8a_export_hem_triplet_data.py \
	step8b_build_hem_triplet_step4b_like.py \
	step8c_hem_triplet_r12_correction.py \
	step8d_audit_same_spin_open_shell_r12.py \
	step8e_audit_hem_same_spin_failure_source.py \
	step8f_hem_pauli_suppressed_geminal_audit.py \
	step8g_scan_hem_suppression_rules.py \
	step8h_generate_hem_triplet_rdm_space.py \
	step8i_scan_hem_open_shell_rule_spaces.py \
	step8j_residual_aware_suppression_audit.py \
	step8k_hem_pair_channel_v_audit.py \
	step8l_channel_resolved_same_spin_candidates.py \
	step8m_internal_qss_v_saturation.py \
	step8n_scan_qs_saturation_rules.py \
	step8o_scan_internal_q_shape_rules.py \
	step8p_hem_same_spin_candidate.py \
	step8q_physical_q_law_audit.py \
	step9a_export_li_ecg_no_rdm_space.py \
	step9b_build_li_step4b_like.py \
	step9c_li_r12_correction.py \
	step9d_scan_li_selected_spaces.py \
	step9e_li_pair_channel_audit.py \
	step10a_export_be_ecg_no_rdm_space.py \
	step10b_build_be_step4b_like.py \
	step10c_be_r12_correction.py \
	step10d_scan_be_selected_spaces.py \
	step10e_be_pair_channel_audit.py \
	step11_cross_system_summary.py

FINAL_SCRIPTS := \
	r12_common.py \
	r12_correction.py \
	step7a_export_ecg_no_data.py \
	step7b_export_ecg_no_orbitals.py \
	step7c_build_ecg_no_step4b_like.py \
	step7d_ecg_no_r12_correction.py \
	step7e_scan_ecg_no_r12_convergence.py \
	step8a_export_hem_triplet_data.py \
	step8b_build_hem_triplet_step4b_like.py \
	step8c_hem_triplet_r12_correction.py \
	step8e_audit_hem_same_spin_failure_source.py \
	step8h_generate_hem_triplet_rdm_space.py \
	step8i_scan_hem_open_shell_rule_spaces.py \
	step8n_scan_qs_saturation_rules.py \
	step8o_scan_internal_q_shape_rules.py \
	step8p_hem_same_spin_candidate.py \
	step8q_physical_q_law_audit.py \
	step9a_export_li_ecg_no_rdm_space.py \
	step9b_build_li_step4b_like.py \
	step9c_li_r12_correction.py \
	step9d_scan_li_selected_spaces.py \
	step9e_li_pair_channel_audit.py \
	step10a_export_be_ecg_no_rdm_space.py \
	step10b_build_be_step4b_like.py \
	step10c_be_r12_correction.py \
	step10d_scan_be_selected_spaces.py \
	step10e_be_pair_channel_audit.py \
	step11_cross_system_summary.py

.PHONY: env update-env check check-final final-summary step1 step2 step3 step4b step5a step5b step5c step6a step6b step6c step6d step6e step6f step6g step6h step6i step6j step6k step6l step6m step7a step7b step7c step7d step7e step7f step7g step7h step7i step7j step7k step7l step7m step7n step7o step7p step8a step8b step8c step8d step8e step8f step8g step8h step8i step8j step8k step8l step8m step8n step8o step8p step8q step9a step9b step9c step9d step9e step10a step10b step10c step10d step10e step11 clean

env:
	conda env create -f environment.yml

update-env:
	conda env update -n $(CONDA_ENV) -f environment.yml --prune

check:
	$(PYTHON) -m py_compile $(SCRIPTS)

check-final:
	$(PYTHON) -m py_compile $(FINAL_SCRIPTS)

final-summary:
	$(PYTHON) step11_cross_system_summary.py --prefix step11_cross_system_summary

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

step7p:
	$(PYTHON) step7p_space_aware_fractional_projector.py

step8a:
	$(PYTHON) step8a_export_hem_triplet_data.py

step8b:
	$(PYTHON) step8b_build_hem_triplet_step4b_like.py --r12-only

step8c:
	$(PYTHON) step8c_hem_triplet_r12_correction.py

step8d:
	$(PYTHON) step8d_audit_same_spin_open_shell_r12.py

step8e:
	$(PYTHON) step8e_audit_hem_same_spin_failure_source.py

step8f:
	$(PYTHON) step8f_hem_pauli_suppressed_geminal_audit.py

step8g:
	$(PYTHON) step8g_scan_hem_suppression_rules.py

step8h:
	$(PYTHON) step8h_generate_hem_triplet_rdm_space.py

step8i:
	$(PYTHON) step8i_scan_hem_open_shell_rule_spaces.py

step8j:
	$(PYTHON) step8j_residual_aware_suppression_audit.py

step8k:
	$(PYTHON) step8k_hem_pair_channel_v_audit.py

step8l:
	$(PYTHON) step8l_channel_resolved_same_spin_candidates.py

step8m:
	$(PYTHON) step8m_internal_qss_v_saturation.py

step8n:
	$(PYTHON) step8n_scan_qs_saturation_rules.py

step8o:
	$(PYTHON) step8o_scan_internal_q_shape_rules.py

step8p:
	$(PYTHON) step8p_hem_same_spin_candidate.py

step8q:
	$(PYTHON) step8q_physical_q_law_audit.py

step9a:
	$(PYTHON) step9a_export_li_ecg_no_rdm_space.py

step9b:
	$(PYTHON) step9b_build_li_step4b_like.py --r12-only

step9c:
	$(PYTHON) step9c_li_r12_correction.py

step9d:
	$(PYTHON) step9d_scan_li_selected_spaces.py

step9e:
	$(PYTHON) step9e_li_pair_channel_audit.py

step10a:
	$(PYTHON) step10a_export_be_ecg_no_rdm_space.py

step10b:
	$(PYTHON) step10b_build_be_step4b_like.py --r12-only

step10c:
	$(PYTHON) step10c_be_r12_correction.py

step10d:
	$(PYTHON) step10d_scan_be_selected_spaces.py

step10e:
	$(PYTHON) step10e_be_pair_channel_audit.py

step11:
	$(PYTHON) step11_cross_system_summary.py

clean:
	rm -f *.npz *.npy *.out *_summary.txt *.csv step6a_slater_fit_N*.json step6b*_fit_convergence.json step6b*_fit_convergence.txt step6b_slater_scan.json step6b_slater_scan.txt step6l_*_scan.json step7a_*_export.json step7b_*.json step7c_*.json step7d_*.json step7e_*.json step7f_*.json step7g_*.json step7h_*.json step7i_*.json step7j_*.json step7k_*.json step7l_*.json step7m_*.json step7n_*.json step7o_*.json step7p_*.json step8a_*.json step8b_*.json step8c_*.json step8d_*.json step8e_*.json step8f_*.json step8g_*.json step8h_*.json step8i_*.json step8j_*.json step8k_*.json step8l_*.json step8m_*.json step8n_*.json step8o_*.json step8p_*.json step8q_*.json step9a_*.json step9b_*.json step9c_*.json step9d_*.json step9e_*.json step10a_*.json step10b_*.json step10c_*.json step10d_*.json step10e_*.json step11_*.json step11_*.md
