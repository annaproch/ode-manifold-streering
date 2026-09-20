# Analysis of the first pilot run (`mps-smoke-v1`) and a code review

*Written 2026-09-20 by an independent review pass. Every number below was read from the committed artifacts or recomputed from them; nothing here was produced by rerunning the pilot.*

| | |
|---|---|
| Run | `pilot-runs/mps-smoke-v1`, committed in `dbc6e52` |
| Code | `fa8ba11` (`pilot-runs/mps-smoke-v1/code-commit.txt`) |
| Model | `meta-llama/Llama-3.1-8B`, revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, fp16 on MPS, batch 1, no chat template |
| Config | smoke profile: `hue_step` 2.0, one saturation/lightness slice, layers {18, 27}, k ∈ {8, 16}, **one seed**, strengths [0, 1] |
| Stages completed | preflight → data → readout-check → collect → calibrate-noop → fit → diagnostics → patch → select |
| Stages **not** run | `validate`, `select` (operating points), `validate-paths`, `test`, `report` |

**Headline:** the causal gate passed, and an 8-dimensional subspace at layer 27 is causally sufficient for this task. The steering comparison that the project actually rests on has not been run yet.

## 1. Gates that passed

### Readout

| Answer set | Chromatic accuracy | Control accuracy | Mean answer mass |
|---|---|---|---|
| 6-way | 1.00 | 0.35 (gray 0.50, brown 0.19) | 0.923 |
| **8-way (used)** | **1.00** | **1.00** (all eight classes at 1.00) | **0.900** |

The 6-way controls failing while the 8-way controls are perfect is a useful internal check: adding `gray`/`brown` was necessary, not cosmetic. Base Llama-3.1-8B does this task cleanly at the first answer token, so the readout is not a limiting factor.

### No-op calibration

Max absolute probability error **0.0** over 48 checks (2 layers × 24 prompts × batch sizes {1}), tolerance 0.02, in fp16 on MPS. Re-injecting a cached activation through the patch hook is exact, so later "did the intervention do anything?" questions are not confounded by the plumbing.

### Zero and small strength on real activations

From `diagnostics/small-strength.json` (1,500 audit rows, validation only, no model forwards):

- **T = 0 is exactly the identity** for every method and every stimulus kind (displacement 0.0000), including gray, brown and unrelated.
- Displacement stays inside the `(1 + correction_ratio)·T` budget: 0.1085 (C-end) and 0.1032 (E2) at T = 0.1; 0.0112 and 0.0103 at T = 0.01.
- **The eligibility gate rejects every off-concept input**: colour stimuli eligible 1.00; gray, brown and unrelated 0.00, all with `reason = rejected_start`.

For contrast, the ungated R0 diagnostic in the synthetic preflight moves a grey start by 3.00 and an unrelated start by 4.05. That is the failure mode the gate exists to prevent — retraction *creating* the concept on inputs that had none — and on real activations it is closed.

### Probe

Analytic gradient versus finite differences: max error **1.4e-12**, radial component 6.9e-17 (the gradient is orthogonal to x, as the normalised sketch requires).

## 2. The substantive result: the causal gate

26 validation prompts whose unsteered answer is not red. Identity patch: success 0.00, p(red) 0.013, other-mass 0.068.

| Layer, k | Full donor patch | Subspace-only patch | Gate |
|---|---|---|---|
| 18, k=8 | 0.46 success | 0.50 | fail |
| 18, k=16 | 0.50 | 0.50 | fail |
| **27, k=8 (selected)** | **1.00**, p(red) 0.596 | **1.00**, p(red) 0.611, other 0.126 | pass |
| 27, k=16 | 1.00, p(red) 0.619 | 1.00, p(red) 0.624, other 0.137 | pass |

Two readings:

1. **Moving 8 PCA coordinates does everything a full activation replacement does** at layer 27. The intervention space the method depends on is real and small. Selection took k = 8 by the predeclared rule (smallest adequate dimension, then strongest effect).
2. **Layer 18 fails for the full patch too**, so the answer is not yet formed at that site. This is a property of the layer, not a bug. `experiment-plan.md` §5.2 currently says a failed full patch means debugging the hook/readout; it should say to run a layer scan first and only suspect the hook if full patches fail *after* the answer is represented.

### The oracle ceiling (recomputed from the collection)

| Condition | p(red) | other-mass |
|---|---|---|
| Natural prompts the model answers red (n = 18) | 0.664 | 0.123 |
| Subspace donor patch, layer 27 k=8 (oracle) | 0.611 | 0.126 |
| Unsteered source prompts | 0.013 | 0.068 |
| All primary stimuli, top-class probability (n = 180) | 0.786 | 0.075 |

The oracle lands just below genuinely-red behaviour and matches its other-mass. **This is the ceiling for any steering method at this site**, and it is narrow: every comparison between B, C, C-end, E2, F and S has to fit between the unsteered baseline and roughly natural quality.

The patch outputs are already saved in `patches/layer27-k8.npz`, so the oracle's endpoint deviation can be computed in the same Bhattacharyya metric the methods are judged by, at zero extra compute. Doing that before the steering sweep fixes the headroom in advance.

## 3. Diagnostics worth carrying forward

**The probe pathology reproduces on real activations** (`fits/layer27-k8-seed0`):

| Metric | Train (n = 92) | Validation (n = 28) |
|---|---|---|
| ROC-AUC | 0.994 | 1.000 |
| Posterior accuracy | 0.891 | 0.929 |
| Balanced accuracy | 0.500 | 0.500 |
| Fraction predicted positive at posterior 0.5 | **0.00** | **0.00** |
| `log_dre > 0`: recall / false-positive rate | 1.00 / 0.256 | 1.00 / 0.269 |

Ranking is essentially perfect while the posterior never crosses 0.5, so `predict()` is all-negative and balanced accuracy is exactly chance — the same artefact found in the toys, now confirmed on Llama activations with a 1:9 class balance. The pipeline reports this correctly rather than hiding it. The corrected threshold is not a sharp red detector either (26% of negatives pass), which matters only for target-set definitions, not for the gradient direction.

**Geometry** (layer 27, k = 8): PCA variance ratios [0.414, 0.365, 0.094, 0.071] — the first two components carry 78%, consistent with a hue circle, but the next two are not negligible. KDE bandwidth 0.795 against a typical projected neighbour distance of 0.798. Gate coverage 0.929 on colour validation, 0.000 on controls. Strength scales: centroid distance 19.2, activation norm 34.8.

## 4. What this run does not show

- **The experiment itself has not run.** No steering evaluation, no operating points, no report.
- **Smoke-sized fits.** 92 training activations with `neighbors: 20` means each "local" tangent uses 22% of the dataset; the local geometry is barely local. The full config (`hue_step` 0.25 → roughly 650 training points) fixes this. Do not read geometric conclusions out of these fits.
- **One seed.** The report criterion requires consistent direction across ≥3 seeds, so this profile cannot produce a publishable comparison by construction.
- **Upstream parity was not checked** (`preflight/report.json` → `upstream: null`). `pilot/preflight.py::upstream_check` compares the reimplemented probe against the real ODESteer kernel (features and a 10-step flow endpoint) given identical hashes and weights. A pinned checkout already exists at `toy/third_party/odesteer` after `toy/setup.sh`, so this is one `--odesteer-src` flag away. It is what certifies that condition B is actually ODESteer.

## 5. Implication for the project's value

An 8-dimensional **linear** subspace already reaches near-natural output quality at layer 27. That validates the architecture, but it also compresses the room in which "estimated data geometry beats plain subspace restriction" could show a gain — condition C may already sit close to the ceiling. The C vs C-end vs E2 comparison is now the whole experiment, and the margin available to it is visibly small.

This does not make the run a negative result. It makes the next run cheap and decisive: if every method lands near the oracle, the answer arrives immediately, and "repeated manifold following does not beat one endpoint projection" is itself a reportable finding.

## 6. Recommended next steps

1. Compute the oracle's endpoint deviation from `patches/layer27-k8.npz` and record it as the predeclared ceiling.
2. Run `preflight --odesteer-src toy/third_party/odesteer` to certify the B baseline.
3. Move to the full colour config with ≥3 seeds and the full strength grid; keep layer 27, k = 8.
4. Before interpreting E2 versus F, sweep the tangent dimension (`tangent_dim` is currently fixed at 1, which prescribes that the concept is a curve — see §7).
5. Check that integrator checkpoints agree with endpoints from the strength sweep, which is what licenses reading path metrics as "dialling strength".

## 7. Code review

Reviewed `pilot/` (2,198 lines) by reading, and `toy/` with an independent review agent. No correctness bug was found that would invalidate results. Verified correct by inspection: the probe's sketch adjoint and normalisation Jacobian match ODESteer's analytic VJP; the correction budget is capped at `ratio·T` independent of step and retraction counts, and vanishes as T → 0; left padding with explicit `position_ids`; gray+grey token aggregation with an overlap check and an unrenormalised `other` bin; PCA/bandwidth/probe fitted on train only with gate thresholds from validation; the paired bootstrap resamples prompts with seeds averaged inside and applies a Bonferroni correction; donors are chosen nearest in the preserved residual.

Findings, in rough priority order:

| # | Where | Issue |
|---|---|---|
| 1 | `toy/geometries.py:83` | The curve is embedded with a zero third coordinate but 3 PCs are kept, so PC3 is noise of sd 0.385 — larger than the hard-coded bandwidth 0.25. The retraction partly chases noise and `max_gap` is inflated about 3×. **Affects the numbers quoted in plan §9.4**; the qualitative conclusions likely survive. Fix: keep 2 components and rerun. |
| 2 | `toy/run_all.py` | Writes results unconditionally, so a quick smaller run silently overwrites the committed 6-replicate results the plan cites. Write to a configuration-derived path or refuse to overwrite. |
| 3 | `pilot/steering.py:70, :98` | `caa` is recomputed from the full training matrix on every call, and `probe.grad(x)` is computed before the method dispatch — so condition A and E3 pay for a gradient they never use. Precompute `caa` in `Geometry`; move the gradient into the branches that need it. |
| 4 | `pilot/steering.py` (`unit`), `pilot/geometry.py:127` | Below `tiny_gradient`, or with an unreachable graph score, the step silently becomes zero while `reason` still reads `ok`. A stalled run is then indistinguishable from a completed one in the per-row records. Give each a distinct `reason`. |
| 5 | `toy/ring_toy.py:223`, `:70` | `clf_pos` and `train_acc` are artefacts of the posterior threshold (`train_acc` 0.8330 versus a majority rate of 0.8340). Report AUC or the corrected threshold instead. |
| 6 | `toy/ring_toy.py:221` | `swing` subtracts raw `atan2` degrees with no wrapping; latent today, but it would corrupt exactly the 180° case the plan highlights. |
| 7 | `toy/setup.sh:13` | Dependency install is guarded on the venv directory rather than on success, so a failed download leaves a venv that reports "ready" while imports fail. |
| 8 | `pilot/preflight.py:108` | `"passed": True` is hardcoded; failures raise instead, but a constant field named `passed` invites a misleading grep. |
| 9 | `pilot/tests` | The upstream parity check is not exercised by the suite; it could run automatically when `toy/third_party/odesteer` exists. |
| 10 | `toy/ring_toy.py:119` | The `Cond` docstring calls D "density only" although the code adds the classifier direction; the plan describes it correctly. |

## 8. Design gaps in the pilot (not code defects)

- **The "ODE" half is untested.** `steps` is global in the config, so there is no single-step control. The obvious comparator is upstream's own `StepODESteer`: one step of the same total length. Without it, an E2 win says "constrained beats unconstrained", not "multi-step integration earns its cost". Adding it needs per-method step counts, because `steering` settings are part of the artifact fingerprint and two configs cannot be compared inside one report.
- **The intrinsic dimension is prescribed.** `tangent_dim: 1` is a fixed scalar with no sweep. On the primary fixed-saturation slice the concept really is a 1-D curve, so this is correct — but by using ground-truth knowledge of the geometry, which weakens "estimated, not prescribed". Report local eigenvalue spectra and sweep m ∈ {1, 2}.
- **Colours can only show parity, never value.** The payoff of not prescribing the curve matters where the curve cannot be prescribed; on a hue circle with six labelled classes the spline (condition F) is trivially available. The colour pilot's role is falsification — it can kill the idea cheaply, it cannot establish the payoff. Plan §5.4 should say so.
- **"Unwanted change" is narrow.** It currently means the answer-token distribution, other-mass, and off-target KL on 12 unrelated prompts. No generation happens anywhere in the pilot, so fluency and content preservation are untested. More unrelated controls are one forward pass each.
