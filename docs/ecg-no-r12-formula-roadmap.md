# ECG-NO + [2]R12 Formula Roadmap

This note records the formula-development branch point after Step 7m.

## Baseline Branch

The current strict-projector audit baseline is preserved at:

```text
codex/step7m-strict-projector-baseline
```

Use this branch if the occupation-weighted projector route needs to be abandoned
or compared against the pre-Step7n state.

## Route 1: Natural-Occupation-Weighted Projector

This is the active main route.

Motivation:

- Full active CABS-only projector is formally conservative but quenches the R12
  correction as low-occupation ECG-NO orbitals enter the OBS space.
- Hard active/passive partitioning restores negative short-range corrections in
  smaller spaces but overcorrects in larger ECG-NO spaces.
- A fractional NO projector should interpolate between these limits using only
  the RDM natural occupations, not an external refscale.

Audit target:

```text
Step7n: fractional occupation projector audit
```

Initial models:

- `linear_occ`: active weight proportional to NO occupation.
- `sqrt_occ`: softer occupation damping.
- `square_occ`: stronger damping of low-occupation NOs.

Acceptance signs:

- 12q retains a meaningful negative R12 correction.
- 18q produces an intermediate correction without exceeding the ECG14 reference.
- 28q is damped enough to avoid the fixed-core overcorrection seen in Step7m.
- fitN=5/7/9 remains stable.

Cross-system requirements:

- The projector definition must not be tuned only to He singlet.
- The same `q(n)` rule must be applicable to He singlet, He triplet, Li ground
  state, and Be ground state.
- The rule must handle different electron counts and spin sectors.  In
  spin-free form, the natural occupations should be interpreted relative to the
  maximum spin-summed occupancy of a spatial orbital.  For closed-shell
  singlets this maximum is 2; for open-shell or high-spin domains the effective
  singly occupied orbitals must not be treated as low-occupation virtuals.
- A viable projector should preserve the cusp-correction role of R12:
  low-occupation correlation NOs may enter the complementary space
  fractionally, while physically occupied or singly occupied valence orbitals
  remain protected from passive-space overcorrection.

Route-1 formula preference:

```text
P_core/state = state-aware occupied or significant natural-orbital domain
Q_frac       = CABS + fractional low-occupation NO complement
q_p          = q(n_p; state, spin, occupancy class)
```

For He singlet the first tensor-level audit is Step7o:

```text
q_p = 0 for fixed core OBS
q_p = model(n_p) for non-core ECG-NO OBS
q_p = 1 for CABS
```

Before this becomes a final formula, the fixed-core rule must be replaced by a
state-aware occupied-domain rule that generalizes to triplet He, Li, and Be.

Next route-1 audit:

```text
Step7p: state-aware occupation-domain projector design
```

This should report, before any new R12 tensor contraction:

- spin multiplicity and electron count;
- spatial NO occupations and occupancy classes;
- which orbitals are protected occupied/singly occupied domains;
- which orbitals are fractional complementary NOs;
- whether the classification is invariant under adding diffuse/correlation NOs.

## Route 2: RDM Cumulant / Multireference Projector

Fallback if Route 1 still overcorrects or lacks a clean interpretation.

Task:

```text
Step7o: cumulant decomposition audit
```

Questions:

- Does projector quenching mainly come from the disconnected part of `dm2`, or
  from the cumulant?
- Should approximation-C/SP subtraction act differently on independent-particle
  and correlated pair-density pieces?
- Can a cumulant-aware contraction remove the need for empirical damping?

## Route 3: Pair-Domain / Pair-Natural-Geminal Projector

Fallback if one-particle NO projectors are inherently insufficient.

Task:

```text
Step7p: pair-natural-geminal projector diagnostic
```

Questions:

- Are the dominant pair-density eigenmodes already localized in the compact
  `s012+p0` domain?
- Is the low-occupation NO cancellation a one-particle projection artifact?
- Can a pair-domain projector define the R12 complementary space more naturally
  than an orbital-domain projector?
