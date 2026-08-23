# Workstream G — Environmental & Health Physics Modeling

Round-7 Workstream G applies the tree-free N-body engine to four humanity-scale
environmental and health-physics challenges. Each challenge maps a real-world
diffusion/transport/attenuation problem onto a screened-Coulomb (Yukawa) or
Gaussian kernel evaluated by the verified `core/yukawa3d_fmm.py` /
`core/gaussian2d_fgt.py` engines via `RadialTaylorFMM.evaluate_targets`.

## Challenges

| Task | Challenge | Kernel | Engine | Status |
| --- | --- | --- | --- | --- |
| T-G1 | Radiotherapy dose distribution | 3D double-Gaussian (pencil-beam) | `Gaussian3DFGT` × 2 widths + analytic anchors | Done |
| T-G3 | Groundwater contaminant plume | 3D Yukawa (full advection-diffusion factorization) | `Yukawa3DFMM` + analytic anchor tests | Done |
| T-G2 | Electrolyte screening (battery) | 3D Yukawa (Debye-Hückel) | `Yukawa3DFMM` + net-charged tail fit + SL moment | Done |
| T-G4 | Airborne pollutant exposure | 3D Yukawa (atmospheric dispersion) | `Yukawa3DFMM` + first-order image sources + well-mixed anchor | Done |

## Physical model

Each challenge maps to a radial kernel K(r) evaluated at target points
(receivers / measurement grid) from source points (emitters / ions / beamlets):

- **T-G1 Radiotherapy**: K(r) = a·exp(-r²/2s₁²) + b·exp(-r²/2s₂²) (isotropic
  double-Gaussian), where s₁, s₂ are the pencil-beam lateral spread widths.
  Sources = beam interaction points; targets = dose grid voxels. The
  `SuperpositionDoseEngine` evaluates two `Gaussian3DFGT` passes (one per
  width) and combines them. Verified by erf anchor, linearity, and convergence
  tests.
- **T-G3 Groundwater**: K(r) = C₀ · exp(-r/L) / r (3D Yukawa), where L is the
  advection-diffusion length scale. The full advection-diffusion-decay equation
  is factorized via the substitution c(x) = exp(v·x / 2D)·u(x), collapsing the
  advection term into a rescaled screened-Helmoltz equation with
  κ² = λ/D + |v|²/(4D²). Sources = contaminant release points; targets =
  monitoring wells. Verified by pure-diffusion and advected analytic anchor
  tests.
- **T-G4 Airborne**: K(r) = Q · exp(-r/λ) / r (3D Yukawa), where λ is the
  atmospheric mixing length. Sources = emission stacks; targets = population
  receptors. Room-scale diagnostics add first-order image sources (7× total:
  original + 6 wall reflections) for no-flux wall BCs, and an eigenfunction
  expansion for the well-mixed anchor test.
- **T-G2 Electrolyte**: K(r) = q_i q_j · exp(-κr) / (εr) (Debye-Hückel), where
  κ = 0.329·√I is the Debye screening parameter. Sources = ions; targets =
  electrode surface grid. Diagnostics: net-charged Debye tail fit (asserts
  κ_fit within 2% of theory) and Stillinger-Lovett second-moment report
  (diagnostic, not a gate).

## Verification

Each challenge includes `test_*` functions that cross-validate the FMM
evaluation against a direct O(N²) reference and/or analytic anchors, with
rel-L2 / percent-error acceptance thresholds. Run all via:

```
python -X utf8 -m tests.environmental_modeling.test_environmental_suite
```

## Honest scope & remaining approximations

- These are **physics-similarity models**: the Yukawa/Gaussian kernels capture
  the dominant attenuation/dispersion physics but are NOT full CFD or Monte
  Carlo radiation transport codes. They give O(N) screening-level estimates,
  not regulatory-grade dose calculations.
- **Open boundaries (T-G2 electrolyte)**: the Debye-Hückel kernel uses
  free-space (open) boundaries — no periodicity, no electrode surface charge
  redistribution. The tail-fit and SL diagnostics validate the kernel physics,
  not a bounded-cell Poisson solver.
- **First-order images (T-G4 airborne)**: the image-source method includes
  only the 6 wall reflections (first order). No corner or edge images
  (reflections of reflections) are included, so the no-flux wall condition is
  approximate. The eigenfunction expansion is exact for a rectangular room
  with Neumann BCs but requires mode truncation.
- **Isotropic kernels**: all kernels are radially symmetric (isotropic). Real
  atmospheric dispersion has directional wind advection (folded into the
  effective λ in T-G4) and real radiotherapy has anisotropic depth-dose
  curves (approximated by the double-Gaussian lateral kernel in T-G1).
- The `evaluate_targets` API (sources ≠ targets) is the Workstream-G enabler;
  it was added to `RadialTaylorFMM` in this round.
