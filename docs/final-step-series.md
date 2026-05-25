# Final Step Series for ECG-NO + R12

This file separates the article-facing workflow from the exploratory audit
scripts that remain in the repository for reproducibility.

## Final Program Layer

Use these steps for the current He/HEM/Li/Be small-space ECG-NO + R12 story:

```text
He:
  step7a_export_ecg_no_data.py
  step7b_export_ecg_no_orbitals.py
  step7c_build_ecg_no_step4b_like.py
  step7d_ecg_no_r12_correction.py
  step7e_scan_ecg_no_r12_convergence.py

HEM:
  step8a_export_hem_triplet_data.py
  step8b_build_hem_triplet_step4b_like.py
  step8h_generate_hem_triplet_rdm_space.py
  step8i_scan_hem_open_shell_rule_spaces.py
  step8p_hem_same_spin_candidate.py
  step8q_physical_q_law_audit.py

Li:
  step9a_export_li_ecg_no_rdm_space.py
  step9b_build_li_step4b_like.py
  step9c_li_r12_correction.py
  step9d_scan_li_selected_spaces.py
  step9e_li_pair_channel_audit.py

Be:
  step10a_export_be_ecg_no_rdm_space.py
  step10b_build_be_step4b_like.py
  step10c_be_r12_correction.py
  step10d_scan_be_selected_spaces.py
  step10e_be_pair_channel_audit.py

Cross-system:
  step11_cross_system_summary.py
```

`step11_cross_system_summary.py` is the final no-recompute collector.  It reads
the existing He/HEM/Li/Be JSON/CSV outputs and writes a unified system table,
selected-space table, and notes.

## Validation Commands

Use the full compile check before pushing code:

```bash
make check
```

Use the narrower final-route check when editing only the article-facing route:

```bash
make check-final
```

Regenerate the current cross-system summary without rerunning expensive
calculations:

```bash
make final-summary
```

## Repository Organization Policy

Tracked files should be source code and lightweight documentation only.
Generated scientific artifacts are intentionally ignored: `.npz`, `.npy`,
`.csv`, `*_summary.txt`, and step JSON/markdown output files.

Exploratory scripts from earlier steps are kept because they document how the
current rules were audited.  They should be treated as an audit layer, not as
the article-facing route.  If the repository later needs a stronger cleanup,
move those exploratory scripts into an `audits/` package in a separate PR so the
history remains reviewable and the current result is not mixed with a large
rename-only diff.

## Current Physical Reading

The current result supports a compact cross-system statement:

```text
He   validates the closed-shell opposite-spin cusp/radial recovery limit.
HEM  isolates the Pauli-suppressed same-spin edge case.
Li   is open-shell but still dominated by s-s radial/core recovery.
Be   shows stronger p-shell angular screening on top of large s-s recovery.
```
