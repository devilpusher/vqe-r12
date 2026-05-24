# VQE-R12 Current Handoff

Date: 2026-05-23

This is the compact handoff for continuing the ECG-NO + [2]R12 prototype work.
The active repo is `/home/zy/code/vqe-r12` in WSL.  Use the `vqecodex` conda
environment.

## Basic Commands

```bash
cd ~/code/vqe-r12
source ~/miniconda/etc/profile.d/conda.sh
conda activate vqecodex
make check
```

Useful targets:

```bash
make step7p   # He singlet space-aware fractional projector audit
make step8i   # HEM triplet selected-space scan prerequisites
make step8k   # HEM pair-channel / V contribution audit
make step8l   # HEM channel-resolved candidate scan
make step8n   # HEM q_s_saturation rule scan
```

## Do Not Commit External Programs

External ECG/NO folders and generated scientific outputs must stay local.

- `local_external/`
- `he-sin/`
- `he-meta/`
- Fortran sources, compiled executables, generated `.npz/.csv/.json/.out`

The Fortran program in `he-meta` is not ours to publish.

## Current Git State

There are uncommitted Step 8 files and Makefile/.gitignore changes.  This is
intentional while HEM same-spin rules are still being audited.

Important untracked files:

- `step8a_export_hem_triplet_data.py`
- `step8b_build_hem_triplet_step4b_like.py`
- `step8c_hem_triplet_r12_correction.py`
- `step8d_audit_same_spin_open_shell_r12.py`
- `step8e_audit_hem_same_spin_failure_source.py`
- `step8f_hem_pauli_suppressed_geminal_audit.py`
- `step8g_scan_hem_suppression_rules.py`
- `step8h_generate_hem_triplet_rdm_space.py`
- `step8i_scan_hem_open_shell_rule_spaces.py`
- `step8j_residual_aware_suppression_audit.py`
- `step8k_hem_pair_channel_v_audit.py`
- `step8l_channel_resolved_same_spin_candidates.py`
- `step8m_internal_qss_v_saturation.py`
- `step8n_scan_qs_saturation_rules.py`

## He Singlet Status

The He singlet ECG-NO route is stable enough for the current prototype.

Main idea:

- Build ECG-NO selected OBS space.
- Align PySCF/Psi4 AO order/sign.
- Construct a Step4b-like parent-basis `.npz`.
- Use the `paper_tequila_sf2r12` style singlet correction.
- For larger ECG-NO spaces, use a space-aware/fractional projector instead of a
  naive fixed-core CABS-only correction.

Important scripts:

- `step7a_export_ecg_no_data.py`
- `step7b_export_ecg_no_orbitals.py`
- `step7c_build_ecg_no_step4b_like.py`
- `step7d_ecg_no_r12_correction.py`
- `step7e_scan_ecg_no_r12_convergence.py`
- `step7p_space_aware_fractional_projector.py`

Takeaway:

- Small He ECG-NO spaces receive a useful negative R12 correction.
- The correction does not exceed the bound-state target in the checked cases.
- Large spaces need fractional/space-aware damping to avoid overcorrection.

## HEM Triplet Status

HEM means metastable He triplet, currently treated as `Ms=1`, two alpha
electrons.  It is not a closed-shell alpha-beta problem.

External data comes from `local_external/he-meta`.  The selected-space RDM is
converted to the same spin-free Step4 energy convention:

```python
E = einsum(h, dm1) + 0.5 * einsum(eri, dm2) + Enuc
Tr(dm1) = 2
Tr(dm2) = 2
```

Minimal HEM selected space:

- channels: `s,p`
- picks: `s01 + p01`
- nobs: 8 spatial orbitals, 16 qubits

Energy reconstruction is correct:

```text
E_input_HEM_triplet_FCI = E_OBS_RDM = E_RI_embedded_RDM
```

Parent pair-FCI target gap for the minimal tested parent is tiny:

```text
E_full_parent_triplet - E_OBS ~= -0.015745 mEh
```

This tiny gap makes HEM much harder than the singlet case.

## HEM Formula Lessons

Do not reuse the closed-shell/singlet `paper_tequila_sf2r12` formula directly
for HEM.

Observed:

- Closed-shell formula gives the wrong sign for HEM.
- Same-spin alpha-alpha needs an antisymmetric SP tensor.
- The dominant R12 channel is `s-s`.
- `s-p` is a small cancellation channel.
- `p-p` is negligible in the checked spaces.
- Fixed same-spin prefactors such as `1/8` overcorrect badly.

The useful same-spin rule family is:

```text
lambda_ss = 2 * sqrt(tail_occ) * q_s_saturation
```

where `tail_occ` is the natural-occupation tail after the two dominant
occupied orbitals.

## Latest HEM Result: Step 8n/Extended

`step8n_scan_qs_saturation_rules.py` scans several choices for
`q_s_saturation`.

Original four-space fully internal candidate:

```text
q_s_saturation = sqrt(n_s_radial / n_p_radial)

if n_s_radial > 2 and n_p_radial > 2:
    q_s_saturation *= 2 / n_s_radial
```

This is called `q_count_balance_cross_ss_only`.

Original four-space Step8n result at fitN=5/7/9:

```text
fitN=5   q_count_balance_cross_ss_only mean_abs=0.000519 mEh  max_abs=0.001010 mEh
fitN=7   q_count_balance_cross_ss_only mean_abs=0.000443 mEh  max_abs=0.000797 mEh
fitN=9   q_count_balance_cross_ss_only mean_abs=0.000486 mEh  max_abs=0.000917 mEh
```

The two extra selected spaces added on 2026-05-23 are:

```text
s012345+p01
s01+p012345
```

With those included, `q_count_balance_cross_ss_only` is still fitN-stable but
not selected-space stable enough to promote:

```text
fitN=5   q_count_balance_cross_ss_only mean_abs=0.001042 mEh  max_abs=0.002531 mEh
fitN=7   q_count_balance_cross_ss_only mean_abs=0.000988 mEh  max_abs=0.002526 mEh
fitN=9   q_count_balance_cross_ss_only mean_abs=0.001019 mEh  max_abs=0.002543 mEh
```

Interpretation:

- `q_count_balance_cross_ss_only` remains useful as the clean internal baseline,
  but do not promote it yet.
- The extended-space failures are systematic: `s012345+p01` wants a larger q
  than count balance gives, while `s01+p012345` wants less p-count damping.
- `q_vsat_self_count_ss_only` and `q_vsat_reference_ss_only` remain audit guides
  only because they use V-saturation/reference-like information.

## Step 8o Internal Q-Shape Scan

`step8o_scan_internal_q_shape_rules.py` scans a still-internal ss-only shape:

```text
q = (n_s / 2)^a * max((2 / n_p)^b, p_floor)

if n_s > 2 and n_p > 2:
    q *= (2 / n_s)^c
```

Best cross-fitN candidate from the default grid:

```text
q_shape_a0.80_b0.45_pf0.76_c1.25
```

Equivalent current-grid tie: `b=0.50`, because the p-floor dominates the tested
large-p cases.

Cross-fitN extended-space results:

```text
fitN=5   mean_abs=0.000092 mEh  max_abs=0.000256 mEh
fitN=7   mean_abs=0.000136 mEh  max_abs=0.000327 mEh
fitN=9   mean_abs=0.000097 mEh  max_abs=0.000228 mEh
```

This is a large improvement over `q_count_balance_cross_ss_only`, whose
extended-space max_abs is about `0.0025 mEh`.  Step8o is the current best HEM
internal candidate, but it should still be tested on at least one mixed larger
space before promotion.

One mixed larger-space test was run at fitN=5/7/9:

```text
s012345+p0123
```

The RDM and bridge checks passed.  Including this case, the previous best
`q_shape_a0.80_b0.45_pf0.76_c1.25` remains good but is not the worst-case best.
Across fitN=5/7/9:

```text
q_shape_a0.80_b0.35_pf0.76_c1.00  worst_max=0.000327 mEh  mean_of_mean=0.000202 mEh
q_shape_a0.80_b0.45_pf0.76_c1.00  worst_max=0.000334 mEh  mean_of_mean=0.000162 mEh
q_shape_a0.80_b0.45_pf0.76_c1.25  worst_max=0.000418 mEh  mean_of_mean=0.000152 mEh
```

The mixed-space signal suggests the cross damping exponent should probably move
from `c=1.25` to `c=1.00` before promotion.  The choice between `b=0.35` and
`b=0.45` depends on whether the production criterion prioritizes worst-case
residual (`b=0.35`) or average residual (`b=0.45`).

## Step 8p Named HEM Candidate

`step8p_hem_same_spin_candidate.py` freezes the conservative worst-case rule as
the current named HEM candidate:

```text
model = hem_q_shape_worst_case_ss_only

q = (n_s / 2)^0.80 * max((2 / n_p)^0.35, 0.76)

if n_s > 2 and n_p > 2:
    q *= (2 / n_s)^1.00

lambda_ss = 2 * sqrt(tail_occ) * q
```

Step8p results across the current seven-space set, including `s012345+p0123`:

```text
fitN=5   mean_abs=0.000182 mEh  max_abs=0.000322 mEh
fitN=7   mean_abs=0.000234 mEh  max_abs=0.000327 mEh
fitN=9   mean_abs=0.000191 mEh  max_abs=0.000316 mEh
```

This is the conservative candidate to promote next if no larger-space stress
test is requested.  It remains HEM-specific, not a universal R12 formula.

## Step 8q Physical Reading

`step8q_physical_q_law_audit.py` decomposes the Step8p candidate into three
interpretable internal factors:

```text
q = s_boost * p_screen * cross_cancel

s_boost      = (n_s / 2)^0.80
p_screen     = max((2 / n_p)^0.35, 0.76)
cross_cancel = 1                    if n_s <= 2 or n_p <= 2
             = 2 / n_s              if n_s > 2 and n_p > 2
```

Physical interpretation:

- `s_boost`: adding s radial functions improves same-spin short-range
  cusp/contact resolution, but sublinearly because of normalization and Pauli
  suppression.
- `p_screen`: adding p radial functions opens exchange-hole/angular relaxation
  that screens the s-s correction, but this screening saturates near `0.76`.
- `cross_cancel`: when both s and p spaces are enlarged, the s-p cancellation
  observed in Step8l/8n removes the direct s boost.  With `c=1.00`, the mixed
  branch becomes weakly decreasing in `n_s` rather than explosively growing.

Step8q confirms this picture is fitN-stable.  Unit same-spin V coupling weakens
strongly as selected s/p space grows, while `tail2` and pair residual weight are
nearly constant; the q rule is therefore compensating selected-space projection
of the same-spin cusp, not changing the amount of residual correlation itself.

## Step 8 Stress Test: s012345+p012345

The largest mixed selected-space stress test was run at fitN=7:

```text
s012345+p012345
nobs = 24 spatial orbitals, 48 qubits
```

RDM and bridge checks passed:

```text
E_input_HEM_triplet_FCI = E_OBS_RDM = E_RI_embedded_RDM
Tr(dm1) = 2
Tr(dm2) = 2
```

The parent pair-FCI target gap is tiny:

```text
OBS -> full parent gap = -0.000459 mEh
```

Step8p including this eighth space:

```text
hem_q_shape_worst_case_ss_only  mean_abs=0.000233 mEh  max_abs=0.000327 mEh
q_count_balance_cross_ss_only   mean_abs=0.000838 mEh  max_abs=0.002526 mEh
```

The new stress case itself is not the limiting residual:

```text
s012345+p012345  q=0.6101  dE=-0.000231 mEh  target=-0.000459 mEh  residual=+0.000228 mEh
```

This supports the Step8q physical reading: p-screening saturates and mixed
cross-cancellation keeps the enlarged s/p space from over-amplifying the
same-spin correction.

Per-space oracle saturation factors from Step8l were:

```text
s01+p01       q ~= 0.992
s0123+p01     q ~= 1.644
s01+p0123     q ~= 0.763
s0123+p0123   q ~= 0.535
```

This shows why a simple monotone damping rule is insufficient: expanding `s`
strengthens the same-spin s-s correction, expanding `p` damps it, and expanding
both causes extra cancellation.

## Suggested Next Step

Continue HEM validation before promoting a formula:

1. Derive a revised internal q rule that handles the 4-to-6 radial extrapolation.
2. Decide the promotion criterion: worst-case suggests
   `q_shape_a0.80_b0.35_pf0.76_c1.00`; mean residual suggests
   `q_shape_a0.80_b0.45_pf0.76_c1.00`.
3. If proceeding conservatively, use Step8p's
   `hem_q_shape_worst_case_ss_only` as the named HEM same-spin candidate.
4. Use Step8q's three-factor reading as the physical explanation if promoting
   Step8p.
5. Keep `q_vsat_self_count_ss_only` only as an audit comparison.
6. Do not move to Li/Be or triplet production until HEM Step8 consistency is
   settled.

## Verification Already Run

```bash
python step8i_scan_hem_open_shell_rule_spaces.py --fitN 5 --memory "4 GB" --nthreads 1 --prefix step8i_hem_triplet_open_shell_space_scan_extended_fitN5
python step8i_scan_hem_open_shell_rule_spaces.py --fitN 7 --memory "4 GB" --nthreads 1 --prefix step8i_hem_triplet_open_shell_space_scan_extended_fitN7
python step8i_scan_hem_open_shell_rule_spaces.py --fitN 9 --memory "4 GB" --nthreads 1 --prefix step8i_hem_triplet_open_shell_space_scan_extended_fitN9
python step8n_scan_qs_saturation_rules.py --fitN 5 --prefix step8n_hem_qs_saturation_rules_extended_fitN5
python step8n_scan_qs_saturation_rules.py --fitN 7 --prefix step8n_hem_qs_saturation_rules_extended_fitN7
python step8n_scan_qs_saturation_rules.py --fitN 9 --prefix step8n_hem_qs_saturation_rules_extended_fitN9
python step8o_scan_internal_q_shape_rules.py --fitN 5 --prefix step8o_hem_internal_q_shape_rules_fitN5
python step8o_scan_internal_q_shape_rules.py --fitN 7 --prefix step8o_hem_internal_q_shape_rules_fitN7
python step8o_scan_internal_q_shape_rules.py --fitN 9 --prefix step8o_hem_internal_q_shape_rules_fitN9
python step8i_scan_hem_open_shell_rule_spaces.py --fitN 7 --memory "4 GB" --nthreads 1 --prefix step8i_hem_triplet_open_shell_space_scan_mixed_fitN7
python step8o_scan_internal_q_shape_rules.py --fitN 7 --prefix step8o_hem_internal_q_shape_rules_mixed_fitN7
python step8i_scan_hem_open_shell_rule_spaces.py --fitN 5 --memory "4 GB" --nthreads 1 --prefix step8i_hem_triplet_open_shell_space_scan_mixed_fitN5
python step8i_scan_hem_open_shell_rule_spaces.py --fitN 9 --memory "4 GB" --nthreads 1 --prefix step8i_hem_triplet_open_shell_space_scan_mixed_fitN9
python step8o_scan_internal_q_shape_rules.py --fitN 5 --prefix step8o_hem_internal_q_shape_rules_mixed_fitN5
python step8o_scan_internal_q_shape_rules.py --fitN 9 --prefix step8o_hem_internal_q_shape_rules_mixed_fitN9
python step8p_hem_same_spin_candidate.py --fitN 5 --prefix step8p_hem_same_spin_candidate_fitN5
python step8p_hem_same_spin_candidate.py --fitN 7 --prefix step8p_hem_same_spin_candidate_fitN7
python step8p_hem_same_spin_candidate.py --fitN 9 --prefix step8p_hem_same_spin_candidate_fitN9
python step8q_physical_q_law_audit.py --fitN 5 --prefix step8q_hem_physical_q_law_audit_fitN5
python step8q_physical_q_law_audit.py --fitN 7 --prefix step8q_hem_physical_q_law_audit_fitN7
python step8q_physical_q_law_audit.py --fitN 9 --prefix step8q_hem_physical_q_law_audit_fitN9
python step8i_scan_hem_open_shell_rule_spaces.py --fitN 7 --memory "4 GB" --nthreads 1 --prefix step8i_hem_triplet_open_shell_space_scan_stress_fitN7
python step8p_hem_same_spin_candidate.py --fitN 7 --prefix step8p_hem_same_spin_candidate_stress_fitN7
python step8q_physical_q_law_audit.py --fitN 7 --inp step8p_hem_same_spin_candidate_stress_fitN7.json --prefix step8q_hem_physical_q_law_audit_stress_fitN7
make check
```
