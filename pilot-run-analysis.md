# First pilot run: analysis, pre-run checks, and why the colour comparison should not be run

*Written 2026-09-20 by an independent review pass. Every number was read from committed artifacts or produced by the scripts in [`checks/`](checks/); nothing here comes from rerunning the pilot itself.*

| | |
|---|---|
| Run | `pilot-runs/mps-smoke-v1`, committed in `dbc6e52` |
| Code | `fa8ba11` (`pilot-runs/mps-smoke-v1/code-commit.txt`) |
| Model | `meta-llama/Llama-3.1-8B`, revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, fp16 on MPS |
| Config | smoke profile: `hue_step` 2.0, one saturation/lightness slice, layers {18, 27}, k ∈ {8, 16}, one seed |
| Stages completed | preflight → data → readout-check → collect → calibrate-noop → fit → diagnostics → patch → select |
| Stages not run | `validate`, `select` (operating points), `validate-paths`, `test`, `report` |

## Summary

The pipeline works and the causal gate passed convincingly: an **8-dimensional subspace at layer 27 is causally sufficient** — patching only those coordinates from a real red donor matches a full activation replacement (both 1.00 success), and lands just below genuinely-red behaviour.

Three cheap pre-run checks then established that **the planned steering comparison would not measure anything**, for a reason the plan did not anticipate:

1. **64% of ODESteer's gradient energy lies outside that subspace, and moving along it is behaviourally inert.** Displacing an activation 20 units along the gradient's off-subspace component (larger than the oracle's 16.8) changes the answer in **0 of 16** cases — less than a random direction does.
2. **The component E2 additionally discards is productive**, not noise: it reaches p(red) = 0.54 and flips 69% of answers.
3. **The "path through grey" prediction is falsified.** p(gray) never exceeds 0.006 in any direction at any distance.

So there is no off-manifold damage for the manifold constraint to prevent, and no third state for it to avoid. The endpoint and path hypotheses are both untestable at this site. Total cost of learning this: about two minutes of GPU time.

**What survives:** the concept subspace is causally sufficient and everything outside it is inert, which argues for *projection* (condition C) rather than manifold-following (E2) — a smaller claim than the project set out to make, but one this data already supports.

**Recommendation:** do not run the dense colour sweep. Run one generation-fragility test instead (§5).

## 1. Gates that passed

### Readout

| Answer set | Chromatic accuracy | Control accuracy | Mean answer mass |
|---|---|---|---|
| 6-way | 1.00 | 0.35 (gray 0.50, brown 0.19) | 0.923 |
| **8-way (used)** | **1.00** | **1.00** (all eight classes) | **0.900** |

The 6-way controls failing while the 8-way controls are perfect confirms that adding `gray`/`brown` was necessary rather than cosmetic. Base Llama-3.1-8B does this task cleanly at the first answer token.

### No-op calibration

Max absolute probability error **0.0** across 48 checks at batch size 1. Re-injecting a cached activation through the patch hook is exact, so nothing downstream is confounded by the plumbing.

At batch size 16 the same check gives **3.45e-03** — the fp16 reduction order depends on batch size. Well inside the 0.02 tolerance, but it confirms `calibrate-noop` must be re-run at whatever batch size a real run uses.

### Zero and small strength on real activations

From `diagnostics/small-strength.json` (1,500 audit rows, validation only):

- **T = 0 is exactly the identity** for every method and stimulus kind, including gray, brown and unrelated.
- Displacement stays inside the `(1 + correction_ratio)·T` budget: 0.1085 (C-end) and 0.1032 (E2) at T = 0.1.
- **The eligibility gate rejects every off-concept input**: colour eligible 1.00; gray, brown and unrelated 0.00.

For contrast, the ungated R0 diagnostic in the synthetic preflight moves a grey start by 3.00 and an unrelated start by 4.05. That failure mode — retraction *creating* the concept on inputs that had none — is closed on real activations.

### Probe

Analytic gradient versus finite differences: max error **1.4e-12**, radial component 6.9e-17.

## 2. The causal gate

26 validation prompts whose unsteered answer is not red. Identity patch: 0.00 success, p(red) 0.013, other 0.068.

| Layer, k | Full donor patch | Subspace-only patch | Gate |
|---|---|---|---|
| 18, k=8 | 0.46 success | 0.50 | fail |
| 18, k=16 | 0.50 | 0.50 | fail |
| **27, k=8 (selected)** | **1.00**, p(red) 0.596 | **1.00**, p(red) 0.611, other 0.126 | pass |
| 27, k=16 | 1.00, p(red) 0.619 | 1.00, p(red) 0.624, other 0.137 | pass |

Layer 18 fails for the *full* patch too, so the answer is not yet formed there. That is a layer property, not a bug: `experiment-plan.md` §5.2 should say to run a layer scan first and only suspect the hook if full patches fail *after* the answer is represented.

## 3. Pre-run checks

Three checks, none of which existed before this review, in [`checks/`](checks/). All are read-only.

### 3.1 Oracle ceiling — `checks/oracle_ceiling.py` (no model)

Scores the donor patches already saved by the `patch` stage in the same metric the arms would be judged by.

| Condition | success | p(red) | other | endpoint BC |
|---|---|---|---|---|
| unsteered | 0.000 | 0.013 | 0.068 | **1.116** |
| full donor patch | 1.000 | 0.596 | 0.122 | 0.0080 |
| **subspace patch (oracle)** | 1.000 | 0.611 | 0.126 | **0.0067** |
| natural red (floor) | — | 0.664 | 0.131 | 0.000 |

The metric has a wide dynamic range (1.12 → 0.007), and the oracle sits essentially on the floor. Also surfaced: the reference split holds only **3** natural red outputs, barely above `minimum_reference_points: 2`. Any real run needs a much larger reference split.

### 3.2 Arm separation — `checks/arm_separation.py` (no model)

Runs the real `steer()` on collected activations; reports where each arm's endpoint lands. At T = 1 (19.2 raw units; oracle displacement 17.6), 25 eligible validation starts:

| Method | Displacement | Distance to oracle |
|---|---|---|
| A (CAA) | 19.22 | 12.42 |
| B (ODESteer) | 18.97 | 15.60 |
| C (projected) | 19.19 | 11.65 |
| C-end | 17.87 | **7.43** |
| E2 | 15.19 | 10.76 |
| F (spline) | 14.11 | 9.77 |
| S (spherical) | 18.96 | 15.48 |
| R0 (retraction only) | 0.63 | 17.59 |

Pairwise separations: **E2 vs C-end 12.36**, E2 vs C 15.49, E2 vs F 3.30, C vs C-end 4.80, B vs C 12.22, **B vs S 0.54**, **A vs B 12.36**.

Readings:

- The arms genuinely differ — E2 and C-end separate by 70% of the oracle distance — so the comparison is not dead on arrival for lack of contrast.
- **B and S are the same intervention** (0.03 apart at T = 0.25, 0.54 at T = 1, 1.26 at T = 1.5). ODESteer's gradient is already tangent to the sphere, so GeoSteer-style norm preservation is vacuous here.
- **A and B differ by 12.4**, so unlike the ring toy the probe is not a disguised mean-difference vector.
- The ranking flips with strength — at T = 1.5, F (6.86) and E2 (9.06) are closest to the oracle, C-end (10.22) further. Comparisons must be at matched success, never matched T.

### 3.3 Premise check — `checks/premise.py` (needs the model, ~400 forwards)

Displaces 16 real activations along five directions and decodes each. At **d = 20** (oracle distance 16.8):

| direction | p(start) | p(target) | p(gray) | other | argmax changed |
|---|---|---|---|---|---|
| on_manifold (toward donor, inside U) | 0.046 | 0.662 | 0.000 | 0.128 | **0.94** |
| **grad_off_U** (gradient outside U) | 0.765 | 0.039 | 0.003 | 0.144 | **0.00** |
| grad_in_U_off_ring | 0.274 | 0.540 | 0.000 | 0.126 | 0.69 |
| random_off_U | 0.734 | 0.043 | 0.003 | 0.122 | 0.06 |
| random_in_U_off_ring | 0.560 | 0.123 | 0.006 | 0.131 | 0.38 |

At **d = 10**: on_manifold 0.422 / 0.315 / 0.31 changed; grad_off_U 0.817 / 0.043 / **0.00** changed; grad_in_U_off_ring 0.683 / 0.116 / 0.00 changed.

Three results:

1. **Movement outside the subspace is inert.** `grad_off_U` carries 64% of the probe gradient's energy — it is exactly what B keeps and what C and E2 discard — and 20 units along it changes nothing: p(target) stays at its baseline 0.039–0.045, and the argmax never changes. It is *less* disruptive than a random direction of the same length.
2. **What E2 additionally discards is productive.** `grad_in_U_off_ring` — the in-subspace component with the local tangent removed, which E2's projection throws away — reaches p(red) 0.540 and flips 69% of answers, most of what the full move achieves. E2 is discarding signal, not noise, which is consistent with its smaller displacement per unit T in §3.2.
3. **No grey anywhere.** p(gray) never exceeds 0.006 in any direction at any distance. Intermediate states are a blend of the start and target colours (at d = 10: start 0.42, target 0.32, everything else 0.16), never a third colour.

**Caveat on labelling.** `on_manifold` is the straight line inside the subspace toward the donor's coordinates — a chord through the disc, not a path along the hue ring. So this run does not separate "along the ring" from "across the interior". What it establishes unambiguously is the part that matters: inside the subspace movement drives the answer, outside it movement does nothing.

## 4. What this means

**The endpoint hypothesis has nothing to measure at this site.** It was framed as "at matched success, E2 causes less unwanted change than simpler controls". The unwanted change was supposed to come from leaving the data manifold — but the off-subspace excursion is behaviourally inert, and other-mass at d = 20 is the same in every direction (0.122–0.144) and matches natural red prompts (0.123–0.131). At matched success, every arm will look alike.

**The path hypothesis loses its colour prediction.** Crossing the interior of the hue disc does not produce grey, so there is no "sensible versus nonsensical intermediate state" contrast to measure. The model interpolates between the start and target colours whichever way it is moved.

**The premise behind manifold-constrained steering is not supported here.** The idea rests on off-manifold states being *bad*. In this model, at this site, they are *ignored*. A constraint that prevents something harmless cannot pay for itself.

**What the data does support** is smaller and cleaner: the concept subspace is causally sufficient, and the majority of ODESteer's gradient lies outside it doing nothing. "Project the steering direction onto the concept subspace and discard the rest" is condition C, not E2 — a real and cheap improvement to ODESteer, already supported by the numbers above.

## 5. Recommendation

**Do not run the dense colour sweep.** It would spend hours to produce a tie on the endpoint metric and no grey on the path metric. The pre-run gates did their job.

**Run one experiment instead: generation fragility.** Take the d = 20 `grad_off_U` states and matched `on_manifold` states, generate 30–40 tokens from each, and compare perplexity under the unsteered model plus repetition and coherence.

- If off-subspace movement degrades generation while leaving the answer token untouched, the manifold story is alive — the colour task was simply too insensitive a readout, and the project should move to a generation-based measurement.
- If generation is unaffected too, off-subspace steering is genuinely inert in this model and the premise itself is wrong.

Either way that is roughly 30 minutes of work and a couple of minutes of compute, and it determines whether the project continues.

**If it continues,** the testbed should be a concept whose behaviour is *not* a near-linear readout of a small subspace, and the readout should be one that can register damage — which points at the deferred phases (emotion, open-ended generation) rather than at more colour work.

## 6. Code review

Reviewed `pilot/` (2,198 lines) by reading, and `toy/` with an independent review agent. No correctness bug was found that would invalidate results.

Verified correct by inspection: the probe's sketch adjoint and normalisation Jacobian match ODESteer's analytic VJP; the correction budget is capped at `ratio·T` independent of step and retraction counts and vanishes as T → 0; left padding with explicit `position_ids`; gray+grey aggregation with an overlap check and an unrenormalised `other` bin; PCA, bandwidth and probe fitted on train only with gate thresholds from validation; the paired bootstrap resamples prompts with seeds averaged inside, with Bonferroni correction; donors chosen nearest in the preserved residual.

| # | Where | Issue |
|---|---|---|
| 1 | `toy/geometries.py:83` | The curve is embedded with a zero third coordinate but 3 PCs are kept, so PC3 is noise of sd 0.385 — larger than the hard-coded bandwidth 0.25. The retraction partly chases noise and `max_gap` is inflated ~3×. **Affects the numbers in plan §9.4**; qualitative conclusions likely survive. Fix: keep 2 components, rerun |
| 2 | `toy/run_all.py` | Writes results unconditionally, so a smaller run silently overwrites the committed results the plan cites |
| 3 | `pilot/steering.py:70, :98` | `caa` is recomputed from the full training matrix every call, and `probe.grad(x)` runs before the method dispatch — A and E3 pay for a gradient they never use |
| 4 | `pilot/steering.py` (`unit`), `pilot/geometry.py:127` | Below `tiny_gradient`, or with an unreachable graph score, the step silently becomes zero while `reason` still reads `ok` |
| 5 | `toy/ring_toy.py:223`, `:70` | `clf_pos` and `train_acc` are artefacts of the posterior threshold (`train_acc` 0.8330 vs a majority rate of 0.8340) |
| 6 | `toy/ring_toy.py:221` | `swing` subtracts raw `atan2` degrees with no wrapping; latent, but it would corrupt the 180° case |
| 7 | `toy/setup.sh:13` | Dependency install is guarded on the venv directory rather than on success |
| 8 | `pilot/preflight.py:108` | `"passed": True` is hardcoded; failures raise instead, but the constant field invites a misleading grep |
| 9 | `pilot/tests` | The upstream ODESteer parity check is not exercised by the suite; it could run when `toy/third_party/odesteer` exists. It was also skipped in this run (`preflight: upstream: null`) |
| 10 | `pilot/model.py:52` | `torch_dtype` is deprecated in current transformers; switch to `dtype` |

## 7. Design gaps in the pilot (not code defects)

- **The "ODE" half is untested.** `steps` is global in the config, so there is no single-step control. Without one, an E2 win would say "constrained beats unconstrained", not "multi-step integration earns its cost".
- **The intrinsic dimension is prescribed.** `tangent_dim: 1` is a fixed scalar — correct for a hue curve, but by using ground-truth knowledge of the geometry, which weakens "estimated, not prescribed".
- **Colours can only falsify, never demonstrate.** The payoff of not prescribing the curve matters where the curve cannot be prescribed; on a hue circle with six labelled classes the spline is trivially available. The pilot's role was falsification — and that is exactly what it delivered.
- **"Unwanted change" was narrow.** It meant the answer-token distribution, other-mass, and off-target KL on 12 unrelated prompts. No generation happens anywhere in the pilot, which is precisely the gap §5 proposes to close.
