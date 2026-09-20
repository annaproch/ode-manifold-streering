# What is novel here, measured against Wurgaft et al. (Manifold Steering)

*Written 2026-09-20. Compares this project's findings against [arXiv 2605.05115](https://arxiv.org/html/2605.05115), the closest prior work. Their facts were read from the paper's HTML; ours come from `pilot-runs/mps-smoke-v1` and the scripts in [`checks/`](checks/). Every number below is traceable to one of those.*

## 0. The overlap is large — start there

| | Wurgaft et al. | This project |
|---|---|---|
| Model / site | Llama-3.1-8B, layer 28 | Llama-3.1-8B, layer 27 |
| Concept space | 64-D PCA | 8-D PCA (chosen by causal patching) |
| Off-subspace residual | held fixed, "combined with the prompt's unchanged off-subspace residual" | held fixed |
| Path | cubic splines through class centroids | classifier gradient flow, optionally constrained |
| Evaluation | output distributions vs a behaviour manifold | output distributions vs natural target outputs |

Shared ideas: a low-dimensional concept subspace from PCA, residual preservation, curved versus straight paths, a single-site intervention, and output-distribution metrics. Density-weighted geometry is also theirs — their Definition 1 already contains `G_E(h) = (α e^{−E(h)} + β)^{−1} I` with `E ∝ −log p`.

Anything claimed as novel has to survive that list.

## 1. They measure paths; we measured endpoints — and the ordering reverses

Their headline is cumulative path energy over 50 waypoints: spline steering is **2.8× more natural** than linear (p < 0.001), with isometry r = 0.99 vs 0.89 and pullback R² 0.77 vs 0.42.

But the transformer never consumes the path. The hook integrates internally and returns one activation; downstream behaviour depends only on that endpoint. So a path result does not imply an endpoint result, and we measured the endpoint at matched success — a comparison their setup does not make.

The ordering reverses. Success by strength multiplier (16 starts, one seed):

| Arm | T = 1 | T = 1.5 | T = 2 |
|---|---|---|---|
| C (projected, unconstrained within the subspace) | 0.812 | 1.000 | 1.000 |
| C-end (C + one endpoint snap) | 0.875 | 1.000 | 1.000 |
| **E2 (stays on the manifold)** | **0.375** | **0.500** | **0.750** |

The on-manifold arm is the one that cannot reach the target. Among the arms that do, the winner at matched success is the *cheapest* correction — one snap at the end:

| T | Endpoint BC, C-end vs C | Paired 95% CI | Prompts favouring C-end |
|---|---|---|---|
| 1.5 | 0.0125 vs 0.0217 (−42%) | [−0.0119, −0.0063] | 15/16 |
| 2 | 0.0248 vs 0.0361 (−31%) | [−0.0126, −0.0100] | 16/16 |

**Claim:** path-naturalness gains do not transfer to endpoint control, and iterated manifold-following costs success. This is a boundary condition on their result, not a contradiction of it.

## 2. Off-subspace movement is behaviourally inert

Displacing real activations 20 units along the classifier gradient's off-subspace component — further than the distance that flips answers when applied inside the subspace — changed **0 of 16** answers, with p(target) flat at 0.039 against a 0.045 baseline. A *random* orthogonal direction of the same length changed 6%. The structured excursion is less disruptive than noise.

Their framework cannot surface this: their energy metric lives in behaviour space, so an off-manifold activation with a natural-looking output does not register as anything. Measuring it directly in activation space, with a causal test, is new here.

The same check found the complementary fact: movement *inside* the subspace but off the data ring is emphatically not inert — it reaches p(red) 0.540 and flips 69% of answers. The model cares where you are inside the plane; it does not care that you are in the plane.

## 3. Overshooting beats real activations — the finding I would keep

Activations the model genuinely answers red with sit at ring radius ≈ 10 and give **p(red) = 0.664**. Steering to radius **26.8** — three times beyond anything that occurs naturally — gives **p(red) = 0.797**.

| Endpoint (T = 1.5) | Radius in PC1–2 | Distance off the subspace | Distance to nearest real activation |
|---|---|---|---|
| a real activation | 10.4 | 1.7 | 0.8 |
| C | **26.8** | 1.7 | 15.7 |
| C-end | 20.9 | 1.7 | 8.6 |
| E2 | 13.8 | 1.7 | 2.1 |
| *the data ring* | *9.9 ± 1.7* | | |

Note the middle column: every arm has the same off-subspace distance as the unsteered activation, because they all move only inside the subspace and preserve the residual. Subspace membership discriminates nothing. What differs is position *within* the subspace — and the data ring occupies only a thin part of it.

**So the most confident "red" state in the model is one that never occurs.** Staying on the data is not merely unnecessary here, it is a ceiling. An interpolation between two known endpoints cannot discover this, because it never leaves the data by construction.

This also explains §1 mechanically: the readout is roughly monotone along the concept direction and does not stop at the data's edge, so an arm that overshoots has a higher ceiling than one that stays home. The endpoint snap then recovers distribution shape without giving back the confidence gained (p(red) unchanged at 0.797; other-mass 0.151 → 0.130 against 0.123 for natural red).

## 4. What the ODE contributed

**Real.** The target is a classifier *region* rather than a pair of class centroids, so no endpoint or ordering has to be specified — a genuine difference from a spline. And the learned direction is not a disguised mean-difference vector: CAA and ODESteer endpoints separate by **12.36** raw units at T = 1, against an oracle displacement of 17.6.

**Costly.** 64% of that gradient's energy points outside the concept subspace, where §2 shows it does nothing. To first order the in-subspace progress per unit strength is `sqrt(1 − 0.638²) ≈ 0.77` of the projected arm's, so ODESteer needs roughly **1.3× the strength** for the same effect, with no quality gain. "Project the gradient onto the concept subspace and discard the rest" is a free efficiency win, and it lives exactly in the gap between the two papers: Wurgaft has no classifier flow, ODESteer has no subspace.

**Untested.** The adaptive multi-step part — the actual ODE — was never isolated. `steps` is a single global setting (10) applied to every arm, with no single-step control, so we cannot say whether integration beats one step of the same total length. Given how nearly straight the field is inside the subspace, my expectation is that it adds nothing, but that is a prediction.

## 5. Support gating has no analogue in their work

Their method interpolates between two known endpoints, so "should this activation be steered at all?" never arises. Here it does, because ODESteer applies its update to every token.

The gate — a proximity test calibrated on validation data — accepted **92.9%** of colour validation activations and **0%** of grey, brown and unrelated ones, and a 1,500-row audit confirmed exact identity at zero strength on real activations. Its *value* is untested; what it fixes is a failure mode we created and then closed, where an unbudgeted correction pulled unrelated activations onto the concept manifold, creating the concept rather than steering it.

Conditional activation steering exists in a prompt-level, semantic form, so the literature check matters before claiming this as new. Support-based, token-level gating appears not to be covered.

## 6. What they have that we do not

The behaviour-manifold formalism and the Hellinger construction; isometry between activation and behaviour geometry; the pullback experiment recovering activation curvature from behaviour; several concepts (weekdays, months, letters, ages); vision models; and a proper statistical treatment. Our evidence is one concept, one site, one seed.

## 7. Honest weighting

Sections 1–3 form a coherent result: **geometry helps for path interpretability, not for endpoint control, because the readout stays monotone past the edge of the data.** That is a short negative or boundary paper on top of theirs, if it replicates.

Section 4 yields one practical recommendation for ODESteer specifically — project the gradient — supported by a direct measurement.

Everything else is either theirs already or untested.

**What this rests on, stated plainly:** 16–26 prompts, one fitted seed, a smoke-sized fit (92 training activations with 20 neighbours), a 3-element reference set for the endpoint metric, validation rather than held-out test data, and a single answer token at a single site. The whole picture also depends on off-manifold being free — and a single answer token is structurally unable to express "this activation is unnatural". A generation-based readout is what would decide whether §3 is a property of the model or an artefact of the measurement.

Before any of this is claimed: more reference points, three seeds, held-out test prompts, a single-step control for §4, and the generation test for §3.
