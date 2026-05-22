# ECG-NO Local Workflow Notes

This project can consume ECG natural-orbital data, but the current ECG-NO
generator source is an external local asset. Do not commit `he-sin/`,
`exnot16.f90`, compiled executables, module files, or generated NO/RDM outputs.

## Local Staging

The current local source folder is:

```bash
/mnt/d/vqecodex/he-sin
```

For repo-side testing, copy it into the ignored local workspace:

```bash
cd ~/code/vqe-r12
mkdir -p local_external
rm -rf local_external/he-sin
cp -a /mnt/d/vqecodex/he-sin local_external/he-sin
```

`local_external/` and `he-sin/` are ignored by Git.

## Compile And Run The ECG-NO Generator

The Fortran program reads the optimized He ECG basis and uses its internal GTO
settings to generate channel-resolved NO coefficients and occupations:

```bash
cd ~/code/vqe-r12/local_external/he-sin
ifx exnot16.f90 \
  -O3 -xCORE-AVX2 \
  -qmkl=sequential \
  -ipo \
  -fp-model=fast=2 \
  -qopt-zmm-usage=high \
  -static-intel \
  -o 1.exe

./1.exe
```

Expected generated files include:

```text
no.dat    nop.dat    nod.dat    nof.dat
occ_s.dat occ_p.dat occ_d.dat occ_f.dat
```

The NO files use Fortran column-major ordering. The first token is `NS`; the
remaining values reshape as `(NS, ncols)` with `order="F"`. See
`he_2rdm_compare.py::read_no_fortran_cols`.

In the current He test run, the generator reported:

```text
max|X^T S X - I| for s,p,d,f blocks: about 1e-14 to 1e-15
sum(occ_s)                         = 1.9917
sum(occ_p radial)                  = 0.0026192
sum(occ_d radial)                  = 0.000066376
sum(occ_f radial)                  = 0.0000057665
sum_s + 3 sum_p + 5 sum_d + 7 sum_f = 1.99992364987102
```

## Check ECG-NO FCI/RDM Diagnostics

`he_2rdm_compare.py` requires PySCF:

```bash
cd ~/code/vqe-r12
conda activate vqecodex
python -m pip install pyscf
```

Then run the diagnostic script from the local `he-sin` directory:

```bash
cd ~/code/vqe-r12/local_external/he-sin
conda activate vqecodex
python he_2rdm_compare.py
```

Current output files:

```text
he_2rdm_compare_summary.csv
he_2rdm_compare_pair_weights.csv
he_2rdm_compare_top_pairs.csv
he_2rdm_compare_matrices.npz
```

The current `he_2rdm_compare_matrices.npz` keys are:

```text
methods
labels
pair_coeff_ab
gamma_spatial
cumulant_ab
metrics_json
ao_channels
```

Current compact He ground-state diagnostics for the selected 14-orbital
ECG-NO subspace:

```text
ECG-NO-FCI energy       = -2.9017962843565535 Ha
trace_gamma             = 2.0
trace_gamma2            = 3.935808026434711
pair symmetry error     = 8.709972850394938e-17
dominant pair weights   = s-s 0.9958613451, p-p 0.0040001390, d-d 0.0001385160
leading geminal weight  = 0.9919334806
```

## Planned R12 Handoff

The R12 side now has a clean formal He correction entry point:

```bash
python step6m_he_r12_correction_pipeline.py --parent-basis cc-pvdz --nobs 2 --fitN 7
```

The ECG-NO handoff should be a later Step 7. The planned replacement is:

1. Use ECG-NO orbital coefficients as `C_obs`.
2. Use ECG-NO-FCI/VQE spin-free `dm1` and `dm2`.
3. Keep the existing parent-basis/CABS route and direct F12 tensor generation.
4. Reuse `r12_correction.compute_he_sf2r12_correction` after a Step-7 data
   adapter produces Step-5a-like arrays.

Do not route F12 production through mixed-basis Psi4 calls.
