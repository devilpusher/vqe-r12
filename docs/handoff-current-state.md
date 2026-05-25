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

## Li Small-Space Step9 Status

The Li ground-state doublet `s01+p0` ECG-NO validation has now been added as
Step9a/Step9b.

Step9a exports the selected-space Li FCI RDMs directly from the local Li
ECG-NO construction in `/mnt/d/vqecodex/lino`:

```text
selection       = s[0,1] + p[0]
nobs            = 5 spatial orbitals = 10 qubits
E_FCI           = -7.45432964359971 Eh
E_RDM           = -7.45432964359972 Eh
Delta(RDM-FCI)  = -1.776e-15 Eh
Tr(dm1)         = 3.000000000000
Tr(dm2)         = 6.000000000000
Tr(D_pair)      = 3.000000000000
natural occ     = [1.9967821057, 0.9999844012, 0.0010778310, 0.0010778310, 0.0010778310]
```

The spin-free spatial RDM convention is therefore internally consistent for
the three-electron Li case: `Tr(dm2)=N(N-1)=6`, while the spin-orbital unordered
pair-space trace is `N(N-1)/2=3`.

Step9b builds the Step4b-like Psi4/CABS+/RI bridge for the same Li RDM:

```text
nobs/ncabs/nri                  = 5 / 59 / 64
Max|S_pyscf-S_psi4|             = 1.332e-15
Max|C_obs^T S_pyscf C_obs-I|    = 1.110e-16
Max|C_obs^T S_psi4 C_obs-I|     = 8.882e-16
E_input_Li_FCI                  = -7.45432964359971 Eh
E_OBS_RDM                       = -7.45432964359972 Eh
E_RI_embedded_RDM               = -7.45432964359972 Eh
Delta(OBS_RDM-input)            = -4.441e-15 Eh
Delta(RI_RDM-input)             = -3.553e-15 Eh
```

This confirms that Li `s01+p0` is now ready for a subsequent small-space R12
candidate/correction step using the Step9b bridge file:

```text
step9b_li_sp_s01_p0_fitN7_step4b_like.npz
```

Step9c applies the first Li small-space SF-[2]R12 correction audit to this
bridge:

```text
DeltaE_R12        = -1.534515799953e-02 Eh  (-15.345158 mEh)
E_OBS_plus_R12    = -7.46967480159924 Eh
V                 = -2.155821177883e-02 Eh
B                 =  4.285916124540e-03 Eh
X                 =  2.850929727860e-03 Eh
Delta             = -9.237920731023e-04 Eh
```

This is a validated contraction/audit result, not yet a promoted Li/Be physical
law.  The diagnostics pass the Li three-electron conventions:

```text
Delta OBS-RDM minus FCI = -4.441e-15 Eh
Delta RI-RDM minus FCI  = -3.553e-15 Eh
Tr(dm1), Tr(dm2)        = 3.000000000000, 6.000000000000
```

Step9d scans four Li selected spaces at `fitN=7` against the local Li ECG
reference `-7.47806002667149 Eh`:

```text
case           nobs      E_obs              dR12/mEh     E_obs+R12          recovery   residual/mEh
sp_s01_p0         5   -7.454329643600     -15.345158   -7.469674801599    0.646646     8.385225
sp_s012_p0        6   -7.468625421019      -3.436386   -7.472061807283    0.364232     5.998219
sp_s01_p01        8   -7.455919693371     -13.742070   -7.469661763378    0.620680     8.398263
sp_s012_p01       9   -7.470147641235      -2.474123   -7.472621764326    0.312690     5.438262
```

Component scan, in mEh:

```text
case                 V          B          X      Delta
sp_s01_p0      -21.558212   4.285916   2.850930  -0.923792
sp_s012_p0      -5.067714   1.367265   0.733136  -0.469074
sp_s01_p01     -19.328159   3.451605   3.001846  -0.867362
sp_s012_p01     -3.860409   1.082390   0.708909  -0.405013
```

Main Step9d reading: adding the third `s` radial NO is the dominant Li
selected-space improvement and reduces the R12 correction magnitude by roughly
a factor of four to six.  Adding the second `p` radial NO gives a much smaller
OBS improvement and only moderately screens the correction.  This supports a
first physical interpretation that the Li small-space correction is mostly
radial/core-valence cusp recovery, with p-space acting as angular screening.

Step9e confirms this interpretation by decomposing the spin-free R12 V
numerator and the spin-orbital pair populations:

```text
case            V_total/mEh    V_s-s       V_s-p      V_p-p      ab_frac   ss_pair_frac
sp_s01_p0        -21.558212  -25.368107   3.992819  -0.182924   0.666667    0.998383
sp_s012_p0        -5.067714   -6.284523   1.394790  -0.177981   0.666667    0.998419
sp_s01_p01       -19.328159  -23.359204   4.167205  -0.136160   0.666667    0.998040
sp_s012_p01       -3.860409   -4.783450   1.051250  -0.128209   0.666667    0.998087
```

The spin-orbital pair population is overwhelmingly s-s (`~0.998`) with the
expected Li doublet spin-pair partition (`ab_frac=2/3`).  The R12 V attraction is
also dominated by s-s.  Adding the third `s` radial NO strongly reduces this
s-s attraction, while s-p is positive and therefore screens the attraction.
This gives a compact physical picture for Li: the current R12 correction mostly
recovers missing short-range radial/core correlation, and p-space acts as an
angular screening channel rather than as the primary source of attraction.

## Be Small-Space Step10 Status

The Be ground-state singlet work now starts from the local early `exnot13.f90`
NO data in `/mnt/d/vqecodex/be2`.  The Fortran calculation was not rerun; the
existing files were used as requested.

The user-specified `be_no_multi.py` validation was run first:

```text
selection                       = s[0,1,2] + p[0]
nobs                            = 6 spatial orbitals = 12 qubits
Max|C^T S C - I|                = 4.441e-16
Be ECG-NO FCI energy            = -14.619416561534816 Eh
RHF energy in this GTO basis    = -14.572976726383626 Eh
10-MO RHF-space FCI             = -14.574417339189374 Eh
```

Step10a reproduces this ECG-NO selected-space energy and validates the
four-electron RDM convention:

```text
E_FCI           = -14.61941656153482 Eh
E_RDM           = -14.61941656153482 Eh
Delta(RDM-FCI)  =  7.105e-15 Eh
Tr(dm1)         = 4.000000000000
Tr(dm2)         = 12.000000000000
Tr(D_pair)      = 6.000000000000
natural occ     = [1.9997039778, 1.8074721795, 0.0637700190, 0.0637700190, 0.0637700190, 0.0015137856]
```

Step10b builds an s,p parent/CABS bridge for the same Be RDM:

```text
nobs/ncabs/nri                  = 6 / 58 / 64
E_input_Be_FCI                  = -14.61941656153482 Eh
E_OBS_RDM                       = -14.61941656153482 Eh
E_RI_embedded_RDM               = -14.61941656153481 Eh
Delta(RI_RDM-input)             =  1.066e-14 Eh
```

Step10c first Be R12 correction audit:

```text
DeltaE_R12        = -4.065982041863e-02 Eh  (-40.659820 mEh)
E_OBS_plus_R12    = -14.66007638195346 Eh
V                 = -7.500628048618e-02 Eh
B                 =  2.176096066236e-02 Eh
X                 =  1.493187702372e-02 Eh
Delta             = -2.346377618530e-03 Eh
```

Against the early Be ECG reference from `enerx.dat`,
`-14.6672938007836 Eh`, the selected-space gap is about `-47.877239 mEh`.
The first R12 correction recovers about `84.92%` of that gap and leaves
approximately `7.217419 mEh` residual:

```text
OBS -> ECG reference gap    = -47.877239 mEh
R12 correction              = -40.659820 mEh
R12 residual to reference   =   7.217419 mEh
```

This is a stronger effect than Li, as expected for a compact Be closed-shell
small space with missing short-range radial/core correlation.  Because the Be
NO data are explicitly early/incomplete, these results should be treated as a
first stability and interpretability audit rather than a final Be benchmark.

Step10d scans the requested Be selected spaces at `fitN=7`:

```text
case             nobs      E_obs               dR12/mEh      E_obs+R12           recovery   residual/mEh
sp_s012_p0          6   -14.619416561535     -40.659820    -14.660076381953    0.849252     7.217419
sp_s0123_p0         7   -14.632406642677     -26.358356    -14.658764999142    0.755532     8.528802
sp_s012_p01         9   -14.642047415424     -20.255386    -14.662302801427    0.802308     4.990999
sp_s0123_p01       10   -14.654774790248      -6.745181    -14.661519970827    0.538795     5.773830
```

Component scan, in mEh:

```text
case                  V          B          X       Delta
sp_s012_p0      -75.006280  21.760961  14.931877  -2.346378
sp_s0123_p0     -54.944110  17.691221  12.276174  -1.381641
sp_s012_p01     -29.241876   7.295663   4.189619  -2.498792
sp_s0123_p01    -10.175169   3.401208   1.492073  -1.463293
```

Main Step10d reading: Be shows a much larger raw R12 correction than Li in the
compact `s012+p0` space, recovering about `85%` of the early ECG reference gap.
The second p radial shell is important for Be: `s012+p01` gives the best
residual among these four cases (`~4.99 mEh`).  Adding the fourth s shell
improves OBS substantially but also reduces the R12 correction, giving a
slightly larger residual in this early-data scan.  This suggests Be has stronger
angular/shell-coupling sensitivity than Li, so the next Be interpretation step
should decompose V by s-s/s-p/p-p channels as in Li Step9e.

Step10e decomposes Be's spin-free R12 V numerator and spin-orbital pair
populations:

```text
case             V_total/mEh    V_s-s       V_s-p      V_p-p      ab_frac   ss_pair_frac
sp_s012_p0        -75.006280  -74.928532   1.089961  -1.167709   0.666667    0.920288
sp_s0123_p0       -54.944110  -54.869622   1.014931  -1.089418   0.666667    0.921427
sp_s012_p01       -29.241876  -31.751580   3.127521  -0.617817   0.666667    0.920387
sp_s0123_p01      -10.175169  -11.705955   2.080498  -0.549712   0.666667    0.921454
```

Be is still strongly s-s dominated in the V attraction, but less purely so than
Li: the s-s pair fraction is about `0.92` rather than Li's `~0.998`.  The
closed-shell spin-pair partition is correct (`ab_frac=2/3`, with aa/bb each
about 1/6 in the detailed spin CSV).  Adding `p01` strongly reduces the s-s V
attraction and the s-p term is positive, so p space acts as angular screening.
The Be interpretation is therefore: large closed-shell s-s radial/core recovery,
with materially stronger p-shell screening than Li.

## Step11 Cross-System Summary

`step11_cross_system_summary.py` is now the no-recompute cross-system collector.
It only reads existing He/HEM/Li/Be JSON/CSV outputs and writes a unified JSON,
system CSV, selected-space CSV, markdown notes, and terminal-style summary.  It
does not rerun FCI, Psi4, R12 contractions, or the external Li/Be Fortran
workflows.

Current representative rows from
`python step11_cross_system_summary.py --prefix step11_cross_system_summary`:

```text
system  representative       dR12/mEh    recovery   residual/mEh
He      sp_s012_p0          -2.011204    0.589431      1.400907
HEM     stress scan          mean residual 0.000233, max residual 0.000327
Li      sp_s012_p01         -2.474123    0.312690      5.438262
Be      sp_s012_p01        -20.255386    0.802308      4.990999
```

The condensed physical statement is: compact ECG-NO selected spaces can be
augmented by a channel-resolved R12 correction with a consistent interpretation
across systems.  He is the closed-shell two-electron opposite-spin cusp sanity
check; HEM isolates the Pauli-suppressed same-spin edge case; Li keeps an
open-shell but still mainly s-s radial/core recovery character; Be shows the
same s-s recovery with stronger p-shell angular screening.

## Suggested Next Step

For the cross-system article goal, continue from the validated small spaces:

1. Use Step11 as the single source for the current cross-system table and
   article-facing wording.
2. Keep HEM conservative: use Step8p's
   `hem_q_shape_worst_case_ss_only` as the named HEM same-spin candidate.
3. Use Step8q's three-factor reading as the physical explanation if promoting
   Step8p.
4. If final quantitative claims are needed, run fitN=5/7/9 stability for the
   representative Li/Be spaces; keep Be labeled as early exnot13 data until the
   optimized Be NO run is available.

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
python step9a_export_li_ecg_no_rdm_space.py --li-dir /mnt/d/vqecodex/lino --s-pick 0,1 --p-pick 0
python step9b_build_li_step4b_like.py --li-dir /mnt/d/vqecodex/lino --rdm-inp step9a_li_sp_s01_p0_rdm_export.npz --s-pick 0,1 --p-pick 0 --fitN 7 --memory 4 --nthreads 1 --r12-only
python step9c_li_r12_correction.py --inp step9b_li_sp_s01_p0_fitN7_step4b_like.npz --prefix step9c_li_sp_s01_p0_fitN7
python step9d_scan_li_selected_spaces.py --li-dir /mnt/d/vqecodex/lino --fitN 7 --memory 4 --nthreads 1 --prefix step9d_li_selected_space_scan_fitN7
python step9e_li_pair_channel_audit.py --fitN 7 --prefix step9e_li_pair_channel_audit_fitN7
cd /mnt/d/vqecodex/be2 && python be_no_multi.py
cd /home/zy/code/vqe-r12
python step10a_export_be_ecg_no_rdm_space.py --be-dir /mnt/d/vqecodex/be2 --s-pick 0,1,2 --p-pick 0
python step10b_build_be_step4b_like.py --be-dir /mnt/d/vqecodex/be2 --rdm-inp step10a_be_sp_s012_p0_rdm_export.npz --s-pick 0,1,2 --p-pick 0 --fitN 7 --memory 4 --nthreads 1 --r12-only
python step10c_be_r12_correction.py --inp step10b_be_sp_s012_p0_fitN7_step4b_like.npz --prefix step10c_be_sp_s012_p0_fitN7
python step10d_scan_be_selected_spaces.py --be-dir /mnt/d/vqecodex/be2 --fitN 7 --memory 4 --nthreads 1 --prefix step10d_be_selected_space_scan_fitN7
python step10e_be_pair_channel_audit.py --fitN 7 --prefix step10e_be_pair_channel_audit_fitN7
python step11_cross_system_summary.py --prefix step11_cross_system_summary
make check
```
