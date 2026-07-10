# Isotropic CHR Discretization Design

## Goal

Remove Cartesian grid-direction bias from the effective 2D graphite stage-2 to
stage-1 CHR model while preserving conservation, differentiability, fixed-step
execution, and the existing inversion interface.

## Confirmed Failure

The current `masked_laplacian` exchanges material only across four axial cell
faces. Its leading truncation error is direction dependent. The circular
boundary also uses the count of exposed square faces as its reaction weight.
Together these choices favor horizontal and vertical propagation and produce
the cross-shaped fronts seen in the montage. The measured transition scores
0.497 maximum angular RMS against a 0.05 gate.

## Considered Approaches

1. Increase resolution without changing the operator. This reduces but does not
   remove the directional error and multiplies runtime, so it is rejected.
2. Move immediately to an unstructured finite-element mesh. This gives the best
   geometric boundary representation but would replace the differentiable JAX
   solver and invalidate much more of the pipeline than necessary.
3. Use a conservative nine-point Cartesian flux operator and analytic circular
   arc weights. This removes the leading directional truncation error, preserves
   pairwise flux cancellation and JAX differentiation, and directly addresses
   both observed sources. This is the selected approach.

## Spatial Operator

The new graph-flux Laplacian connects each active cell to four axial and four
diagonal neighbours. Each undirected pair contributes equal and opposite flux.
Axial conductance is `2/3` and diagonal conductance is `1/6`, with all terms
scaled by `dx**2`. On a full grid this is the standard isotropic nine-point
Laplacian:

```text
1  4  1
4 -20  4   / (6 dx^2)
1  4  1
```

Only active-active pairs contribute, which enforces discrete no-flux behavior
at the particle boundary. Pairwise cancellation makes the operator conservative
and symmetric. The same operator is used for concentration gradients,
chemical-potential diffusion, stabilization, and the implicit biharmonic term.

## Circular Boundary Reaction

Reactive cells remain the active cells adjacent to the exterior. Instead of
counting exposed square faces, each reactive cell receives the Voronoi arc of
the analytic circle closest to its polar angle. The weights sum to exactly
`2*pi*radius`, rotate by 90 degrees with the grid, remain zero in the interior,
and distribute a uniform surface reaction independently of the staircase face
orientation.

The active cell area remains binary in this repair. Fractional cut-cell volumes
are deferred because they would change the inner product and require a
volume-weighted symmetric linear solve. If refinement still fails, that is the
next justified numerical upgrade.

## Validation

Tests are added before implementation. They require:

- constants are annihilated;
- pairwise fluxes sum to zero and the operator is symmetric;
- the nine-point stencil has lower axial-versus-diagonal Fourier-symbol error
  than the current five-point stencil;
- circular boundary weights are nonnegative, interior-zero, 90-degree
  equivariant, and sum to the analytic circumference;
- a controlled zero-noise radial transition passes angular isotropy;
- a rotated non-radial initial condition produces the rotated trajectory;
- mass balance, relaxation, full cycle, and grid refinement still pass.

The controlled radial test is the claim-bearing isotropy gate. A random
perturbation may legitimately break radial symmetry, so its angular RMS is
reported as morphology rather than used alone as a discretization verdict.

## Downstream Gate

No synthetic cohort, inversion result, or backend authorization from the old
operator is reusable. After the local solver gates pass, the transition is
rerun and rendered with fixed concentration limits. The execution fingerprint
is regenerated, then canonical CPU and GPU probes must pass before any cohort
is rebuilt.

## Controlled-Run Revision

The selected Cartesian repair was implemented and falsified before downstream
generation. Controlled angular RMS remained `0.492`, `0.452`, and `0.422` at
48, 64, and 96 pixels. The diffuse interface is only `0.015` model-length units
wide: `0.72`, `0.96`, and `1.44` cells on those grids. The binary boundary also
retains an approximately ten-percent fourfold source-density harmonic.

Therefore the nine-point Cartesian solver remains a diagnostic backend, not the
claim-bearing circular reference. The first claim-bearing model will be a
volume-weighted radial finite-volume CHR solver with at least four cells across
the 10-90 interface. Its 1D radial state will be rasterized into a 2D circular
movie by a fixed differentiable interpolation. This exactly matches the current
homogeneous circular-particle assumptions and provides an isotropic reference
oracle. It deliberately cannot represent non-radial heterogeneity; a full 2D
polar or agglomerated cut-cell backend is a later, separately gated extension.
