# Step 6 Formula Map: He Parent-Basis [2]R12 Prototype

This note maps the paper/Psi4 MP2-F12 symbols to the current He prototype
objects.  It is a convention checkpoint before turning Step 6c diagnostics into
a final He-only [2]R12 candidate.

## Reference And Spaces

| Symbol / concept | Current object | Notes |
|---|---|---|
| OBS \(P\) | first `nobs` RI orbitals | In Step 4b, OBS is the first `nobs` RHF MOs in the parent basis. |
| CABS / RI complement | RI orbitals `nobs:nri` | Constructed as parent-basis CABS+ complement. |
| RI space | `C_ri = [C_obs, C_cabs]` | All F12 tensors are generated in one parent AO basis and transformed to RI. |
| reference state \(|\Phi_\mathrm{ref}\rangle\) | `Cab_obs`, `dm1_obs`, `dm2_obs` | For current He tests this is OBS pair-FCI, not ECG-NO/VQE yet. |
| spin-free 1-RDM \(\gamma^p_q\) | `dm1_obs`, `dm1_ri` | `Tr(dm1)=2` is required. |
| spin-free 2-RDM \(\gamma^{pq}_{rs}\) | `dm2_obs`, `dm2_ri` | `Tr(dm2)=2` is required with current convention. |

## Operators And Integrals

| Symbol / concept | Current object | Notes |
|---|---|---|
| one-electron Hamiltonian | `h_ri` | Includes kinetic + nuclear attraction. |
| Coulomb tensor \(g\) | `eri_ri` | Chemist ordering in code: `eri[p,q,r,s] = (p q | r s)`. |
| generalized Fock \(F\) | `F_ri` | Built from spin-free `dm1_ri` in Step 5a. |
| correlation factor \(f_{12}\) | `f12_ri` | Psi4 corr order is `(Gaussian exponent, coefficient)`. |
| \(f_{12}^2\) | `f12sq_ri` | Direct Psi4 tensor; do not replace with finite RI matrix closure `f @ f`. |
| \(f_{12} g_{12}\) | `f12g12_ri` | Direct Psi4 tensor; do not replace with finite RI matrix closure `g @ f`. |
| \([f_{12}, [T, f_{12}]]\) | `f12dc_ri` | Direct Psi4 double-commutator tensor. |

## SP Ansatz

The SP ansatz from Ten-no fixes amplitudes:

```text
d[p,q,r,s] = 3/8 delta[p,r] delta[q,s] + 1/8 delta[p,s] delta[q,r]
```

Current object:

```text
D_sp
Cab_sp = einsum("pqrs,pq->rs", D_sp, Cab_obs)
A_sp_Q = Q f12 Cab_sp
```

## Projectors

The Step 6d audit follows the SR-F12 Ansatz-3 projector pattern used in the
Psi4 MP2-F12 theory documentation:

```text
Q12 = 1 - |a'j><a'j| - |ib'><ib'| - |rs><rs|
```

Current He labels:

| Label | Meaning |
|---|---|
| `rs_obs` | both pair indices in OBS |
| `a_prime_j` | first index in CABS, second index occupied OBS |
| `i_b_prime` | first index occupied OBS, second index in CABS |
| `q_ansatz3` | complement after removing the above three blocks |

For He/cc-pVDZ/nobs=2/nocc=1/nri=5:

```text
pair dim = 25
dim(rs_obs) = 4
dim(a_prime_j) = 3
dim(i_b_prime) = 3
dim(q_ansatz3) = 15
```

## Explicit V/X/B/C Intermediate Plan

Step 6e should build intermediate tensors/scalars explicitly and label their
status:

| Intermediate | Current ingredients | Status |
|---|---|---|
| `V_direct` | direct `f12g12_ri` contractions | Available. |
| `X_direct` | direct `f12sq_ri` contractions | Available. |
| `B_dc_direct` | direct `f12dc_ri` contractions | Available. |
| `B_fock_model` | `F(1)+F(2)` contractions | Diagnostic model; P-Q coupling is zero in current He case. |
| `C_orbital_denominator` | occupied/virtual orbital-energy denominators | Not final; must be audited against [2]R12 approximation-C convention. |
| `tilde V`, `tilde B` | `V/B` corrected by `C` terms | To implement as candidate diagnostics, not final energy. |

## Do Not Conflate

- `f12g12_ri` is not equal to finite RI matrix closure `eri_ri @ f12_ri`.
- `f12sq_ri` is not equal to finite RI matrix closure `f12_ri @ f12_ri`.
- full Q-pair solve is an external-pair diagnostic, not [2]R12.
- fixed-amplitude rows are diagnostics only.
- ECG-NO is Step 7, after He/Psi4 OBS is stable.

