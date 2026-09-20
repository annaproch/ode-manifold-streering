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

Three cheap pre-run checks then established three things the plan did not anticipate:

1. **64% of ODESteer's gradient energy lies outside that subspace, and moving along it is behaviourally inert.** Displacing an activation 20 units along the gradient's off-subspace component (larger than the oracle's 16.8) changes the answer in **0 of 16** cases — less than a random direction does.
2. **The component E2 additionally discards is productive**, not noise: it reaches p(red) = 0.54 and flips 69% of answers.
3. **The "path through grey" prediction is falsified.** p(gray) never exceeds 0.006 in any direction at any distance.

So the component the constraint removes is inert, the component it *also* removes is useful, and there is no third state to avoid. The path hypothesis loses its colour prediction outright. Total cost of learning this: about two minutes of GPU time.

**Correction (after review).** An earlier version of this document concluded that the endpoint hypothesis was untestable here. That is **not established**: the check reported other-mass, while the endpoint metric is `nearest_bc` over the full nine-bin vector. At d = 20 the two in-subspace directions have almost identical other-mass (0.128 vs 0.126) but very different distributions — `on_manifold` leaves 0.046 on the start colour, `grad_in_U_off_ring` leaves **0.274**, and natural red outputs do not look like the latter. That residue is an unnaturalness signature the endpoint metric can see and the printed column cannot. Resolving it needs a re-run of `checks/premise.py` that computes `nearest_bc` against the natural reference, compared at matched behaviour rather than matched distance (§5).

**What survives:** the concept subspace is causally sufficient and everything outside it is inert, which argues for *projection* (condition C) rather than manifold-following (E2) — a smaller claim than the project set out to make, but one this data already supports.

**Recommendation:** still do not run the dense colour sweep — but for a narrower reason than before (§5). Run two cheap tests first: the corrected endpoint measurement, and a generation-fragility test.

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

**The endpoint hypothesis is narrowed, not dead.** It was framed as "at matched success, E2 causes less unwanted change than simpler controls". For the *off-subspace* excursion the answer is clear: that component is behaviourally inert, so the B-vs-C contrast is bounded at roughly 0.016 of other-mass and zero success — there is nothing there to win.

For the *in-subspace, off-ring* excursion — the component E2's tangent projection removes — the question is open, and this check could not answer it because it printed other-mass rather than the endpoint metric. The two directions differ sharply in shape (residual mass on the start colour 0.046 vs 0.274) while matching on other-mass, so `nearest_bc` may well discriminate where the printed column cannot. Two further cautions apply to any such comparison: the per-row SD of other-mass is 0.014–0.017, giving a standard error near 0.0035 at n = 16, against a batch-16 plumbing error of 3.45e-03; and `on_manifold`'s other-mass is non-monotonic in distance (0.074, 0.082, 0.102, 0.159, 0.128), so comparing at matched distance rather than matched behaviour flatters whichever arm has already committed to the target.

**The path hypothesis loses its colour prediction.** Crossing the interior of the hue disc does not produce grey, so there is no "sensible versus nonsensical intermediate state" contrast to measure. The model interpolates between the start and target colours whichever way it is moved.

**The premise splits in two, and only one half is settled.** Manifold-constrained steering rests on off-manifold states being *bad*. "Off-manifold" turns out to mean two different things here. Outside the concept subspace, states are *ignored* — a constraint that prevents something harmless cannot pay for itself, and that disposes of the B-vs-C contrast. Inside the subspace but off the hue ring, states are emphatically not ignored: they flip 69% of answers. Whether they are also *worse* is the open question, and it is the only place left where E2's tangent projection could earn anything.

**What the data does support** is smaller and cleaner: the concept subspace is causally sufficient, and the majority of ODESteer's gradient lies outside it doing nothing. "Project the steering direction onto the concept subspace and discard the rest" is condition C, not E2 — a real and cheap improvement to ODESteer, already supported by the numbers above.

## 5. Recommendation

**Do not run the dense colour sweep yet**, and probably not at all. The path half is settled — no grey, so nothing to measure. The endpoint half needs one more minute of compute before it can be judged, and even the optimistic outcome most likely favours the simpler arm: the premise check showed that what E2 discards is productive, so E2's only route to a win is a more natural endpoint — which is precisely what **C-end** delivers by keeping the productive signal and snapping onto the manifold once. C-end was already closest to the oracle at T = 1 (7.43 versus E2's 10.76). "C-end matches E2, prefer the simpler method" is an outcome the plan predeclared.

**Test 1 — the corrected endpoint measurement (~1 minute).** Re-run `checks/premise.py` saving the per-state probabilities, compute `nearest_bc` against the same natural target reference the `evaluation` stage uses, and compare directions at matched behaviour (equal fraction of answers flipped) rather than matched distance.

A direct method comparison is now prepared in `checks/endpoint_comparison.py`. Run on the machine holding the pinned model:

```sh
python checks/endpoint_comparison.py --run pilot-runs/mps-smoke-v1 --batch-size 4
```

It uses 16 eligible non-target validation starts sampled across hue order, C/C-end/E2, and strength multipliers 0.5/1/1.5/2: 208 decoded prompt states including identity checks. It saves complete nine-bin endpoint probabilities, endpoint BC distances, success curves and a Markdown report under `pilot-runs/mps-smoke-v1/checks/endpoint-comparison/`. It checks success comparability at the smallest sampled strengths reaching 80%; unmatched success is explicitly flagged. Only three independent red-reference outputs are available, so this is exploratory and cannot establish superiority. Existing output directories are preserved; choose a new `--output` for a repeat. The script's dry run and mocked output checks passed locally, but **the decoded comparison has not run**: the available host has 8 GB RAM, no accelerator and no cached model weights. No timing estimate for this new comparison has been measured.

**Test 2 — generation fragility (~30 minutes).** Take the d = 20 `grad_off_U` states and matched `on_manifold` states, generate 30–40 tokens from each, and compare perplexity under the unsteered model plus repetition and coherence.

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

## 8. Update: decoded endpoint results (2026-09-20)

**This update supersedes the earlier recommendation and statements that the decoded comparison has not run.** Results committed in `c1a27ec` provide a positive exploratory signal for **C-end**, not for continuous manifold-constrained steering E2. The earlier sections remain as the record of the pre-comparison analysis; their strongest negative conclusions should not be treated as established findings.

Sources: [decoded comparison report](pilot-runs/mps-smoke-v1/checks/endpoint-comparison/report.md), [summary](pilot-runs/mps-smoke-v1/checks/endpoint-comparison/summary.json), and the per-method endpoint probability files in the same directory.

### Setup and what changed

The experiment still uses the colour task: Llama-3.1-8B, layer 27, an eight-dimensional concept subspace, one fitted seed, and 16 eligible non-target validation starts sampled across hue order. It decoded 208 prompt states: identity plus C, C-end and E2 at four strength multipliers. These are first-answer probability distributions, not generated text. The no-op error was 0.00176, below the configured 0.02 tolerance.

- **C:** ODE steering projected into the concept subspace, without manifold correction.
- **C-end:** the same steering followed by three bounded manifold-correction iterations at the endpoint.
- **E2:** tangent-projected steering with bounded manifold correction throughout the ten integration steps.

C-end and E2 have the same maximum total correction budget, 0.25T. The endpoint metric is Bhattacharyya distance to the nearest natural red-reference output; lower is better. Only three target references are available.

### Main results

| Method | Strength multiplier | Red success | Mean endpoint BC |
|---|---:|---:|---:|
| C | 1.5 | 16/16 | 0.02166 |
| C-end | 1.5 | 16/16 | 0.01255 |
| E2 | 1.5 | 8/16 | 0.19891 |
| C | 2 | 16/16 | 0.03613 |
| C-end | 2 | 16/16 | 0.02477 |
| E2 | 2 | 12/16 | 0.07893 |

At strength 1.5, **C-end reduces mean endpoint BC by 42.1% versus C at equal observed success**, improving 15 of 16 paired prompts. At strength 2, both again reach 100% success and C-end reduces mean BC by 31.4%, improving all 16 prompts. At 1.5, their mean red probabilities are almost identical (0.797), while other-mass falls from 0.151 to 0.130 with C-end. These are descriptive comparisons chosen after inspecting validation results.

E2 reaches at most 75% success over the tested grid. It therefore has no operating point matching C/C-end's 100% success here; its endpoint quality cannot be ranked fairly against those points. This is evidence against the current E2 configuration, not proof that all strengths or schedules would fail.

The report's `success_comparable: false` concerns its automatic selection of the smallest strength reaching 80% success: C reaches 13/16 at T=1, C-end 14/16 at T=1, and E2 never reaches that threshold. It does not invalidate the separate exploratory C-versus-C-end comparisons at T=1.5 and T=2, where observed success is equal.

### What we have and have not validated

We now have evidence that **endpoint manifold correction can improve the decoded target distribution beyond concept-subspace ODE steering alone**. That is the promising contribution to pursue. These results do not establish that continuous manifold constraints help, that multi-step ODE integration is necessary, or that the method is superior to all existing steering methods.

The result is provisional: one seed, 16 validation starts, three target references, no held-out confirmation and no bootstrap claim. An exploratory leave-one-reference-out check keeps the C-end improvement at T=1.5 positive, but changes its magnitude to roughly 20–42%; reference coverage matters. This sensitivity check is not independent replication.

The earlier premise check also needs a narrower interpretation. Not producing grey weakens that particular prediction; it does not settle intermediate-path naturalness. Tested off-subspace directions being mostly inert for the first answer does not establish that all such directions are harmless for generation, or prove C beats B. The donor-directed straight chord labelled `on_manifold` is not a demonstrated manifold path. Colours can demonstrate a local endpoint benefit, though they cannot by themselves establish generality or a novel advantage on unknown geometry.

### Next experiment and novelty

Prioritise **C-end versus C**. Freeze the operating-point selection rule and endpoint metric before examining held-out outcomes; collect more independent target references; then confirm on more held-out prompts and fitted seeds. Keep equal-success comparisons explicit. A later single-step control is needed to establish whether ODE integration contributes beyond the correction itself.

Do not prioritise a broad E2 sweep on the present evidence. Any E2 follow-up should target a specific reason continuous correction might help, with a matched-success comparison. Generation and preservation of unrelated behaviour remain separate, untested questions.

The candidate novelty is the measured value of a **locally data-estimated manifold correction beyond subspace projection**, particularly the simpler endpoint-only variant. "ODE + manifold" alone is not an established novelty claim; positioning against prior work still requires verification. The present result justifies a focused confirmation experiment, not a superiority claim.
