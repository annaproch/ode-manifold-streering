# Review and next experiment — 2026-09-19

**Decision: pursue one bounded real-model pilot. Do not commit to the six-phase programme yet.** The toys justify testing the mechanism, but do not establish an advantage for language-model generation. The strongest remaining question is whether a data-estimated concept constraint improves the success/side-effect tradeoff beyond subspace restriction, norm preservation, or a cheap endpoint correction.

Reviewed `experiment-plan.md`, the toy implementations and committed results, and the primary papers linked below. Ran an additional seed-0 classifier diagnostic for the ring, horseshoe and Y. No LLM experiment was run.

## What would make this worth pursuing

A useful result would be: with the same classifier and activation data, local concept geometry gives better endpoint behavior at matched target success, with manageable overhead and some generalization across prompts. Controllable intermediate behavior across steering strengths is also a legitimate independent research objective. Predeclare separate endpoint and path hypotheses; path naturalness is primary for the latter, as in [Wurgaft et al., Appendix A.7](https://arxiv.org/html/2605.05115), rather than treating it only as a fallback after endpoint results are known.

The current hook computes all integration steps before returning the final activation to the model. The downstream transformer does not consume the internal trajectory. Two methods returning exactly the same activation in the same model/cache context must give the same downstream result. A beautiful path is therefore a diagnostic, not sufficient evidence of better generation. Add **projection-only steering followed by a single endpoint retraction** as a crucial cheap baseline.

Novelty is narrower than the plan suggests. In addition to its existing references, [GeoSteer, submitted September 9](https://arxiv.org/html/2609.10658), uses a polynomial-count-sketch probe, tangent-projected gradients and geodesic updates on the activation sphere. It does not estimate the concept's empirical manifold, so it does not settle this proposal, but it is a close comparator. The distinctive claim must concern the benefit of the estimated data geometry. “No training” should become “no additional neural geometry model”; the classifier is trained and the geometry is fitted.

## Issues that change the experiment

1. **A concept manifold need not preserve the properties you care about.** If the colour data include changes in saturation and lightness, a valid on-manifold path can change either. Grey can be a perfectly natural colour activation. First test a conditional hue curve at fixed saturation/lightness; use other slices as controls. A dense input hue sweep also does not guarantee a dense smooth activation curve: the model may collapse inputs into colour-name clusters. Verify local continuity and causal usefulness.

2. **Classifier accuracy is misleading in the toys.** Rebuilding seed 0 gives:

   | Geometry | Reported training accuracy | Predicted-positive fraction |
   |---|---:|---:|
   | Ring | 0.8330 | 0 |
   | Horseshoe | 0.9053 | 0 |
   | Y | 0.9163 | 0 |

   **Correction following follow-up review:** these are all-negative posterior predictions at the default 0.5 threshold, not failures of the density-ratio classifier. ODESteer's `log_dre` adds `log((1-pi)/pi)` to the logit, where pi is the training positive fraction. Consequently `log_dre > 0` is equivalent to posterior probability > pi, not > 0.5. In a fresh seed-0 ring check, pi = 0.167; the corrected criterion accepts every training positive and every E2 endpoint from the 150°, T = 20 condition. It also accepts 33.0% of training negatives, so it is not an exact red-region classifier.

   The ranking is strong: training ROC-AUC is 0.9897 for horseshoe and 0.9982 for Y; a noisy-coordinate proxy for the ring labels gives 0.9870. Prior correction is an additive constant and does not change the gradient direction. Report held-out ranking, class balance, calibration and the score along the measured data path. E3 can use labeled positives, a validated density-ratio threshold, or a predeclared high-score subset; labels are not required. The user's check reports unchanged E3 success with highest-scoring stored points; this review has not independently rerun that comparison. E3 changes both the objective and the geometry relative to B, so its success does not isolate the constraint's contribution.

3. **The residual probe is not a causal conflict test.** E2 leaves `(I-UUᵀ)x` unchanged, so a fixed probe on that residual necessarily gives the same answer before and after steering. Decodable residual hue indicates redundant information, not necessarily harmful conflict. Test causal sufficiency by replacing the concept coordinates with those of real target donors while preserving the source residual; compare with full donor replacement as a diagnostic upper control. Sweep k only if this test motivates it.

4. **The colour concentration metric is not physical saturation.** A 50/50 red/green prediction can have low circular concentration without representing grey; brown can be chromatic. Use explicit gray/brown probabilities separately, retain an `other` bin for probability outside the answer tokens, and compare to natural held-out output distributions near real colour boundaries. Do not classify ordinary boundary uncertainty as damage.

5. **The geometry does not give the claimed guarantees.** E2's score need not be Euclidean closeness to a target; the actual restricted score landscape determines whether it stalls. E3's discrete shortest-path values decrease correctly along graph routes, but a smoothed local direction followed by finite retractions need not preserve that property. A Y junction has no unique smooth one-dimensional tangent. Delay geography until measured activations support a branch interpretation; a semantic hierarchy alone does not establish one.

6. **Retraction is an intervention even with zero steering strength.** The current loop applies its normal corrections independently of T. A fresh seed-0 check with 40 iterations and r = 3 gave median displacement 0.063 R for noisy ring starts and 0.311 R for noisy grey-centre starts at T = 0. This confirms the mechanism, but does not reproduce the user's 0.76 R under their unspecified start conditions. Pulling arbitrary token activations toward a colour manifold can manufacture a concept rather than preserve it. This is a major obstacle to steering every generated token.

   Add a diagnostic retraction-only control with zero directional step and the same correction count. The deployed steering operation must return x exactly at T = 0; keep that separate from the diagnostic control. Also sweep small positive T: an early return alone can hide a discontinuity. Require displacement to approach zero as T approaches zero, and bound correction size. Fit a near-manifold eligibility rule on validation activations before interventions, report eligibility coverage and rejected cases, and compare all methods on the same eligible set as well as the full set. Proximity in projected coordinates is insufficient evidence of membership for unrelated prompts or generation positions; validate those separately before claiming per-token applicability.

## Run next: a small colour experiment with a causal gate

Use one pinned Llama-3.1-8B checkpoint and one exact prompt/token convention. Record base versus instruct explicitly. Fit at the last prompt token and evaluate the first answer token only for this pilot.

**First job: collect activations and test whether the intervention space works.**

- Generate approximately 2,000–3,000 distinct hex stimuli covering a dense hue sweep and a few saturation/lightness slices, plus gray/brown controls. Choose one chromatic slice as the primary hue curve. Split by hex identity before fitting anything; keep near-boundary examples. Reserve a small second prompt template as a later generalization check.
- Validate the answer-token aggregation and raw answer mass. Start with the existing six-colour readout and validate the extended readout independently. Cache a small set of candidate layers around the cited colour and late-answer sites; record their exact zero-based hook indices.
- Fit PCA on training data only. At k = 8, 16, 64, patch real target-donor coordinates into source prompts, preserving their residuals. Compare with full donor patching. Examine whether neighboring hue inputs remain neighbors across data splits and whether the patched concept actually changes the output.
- Select one layer and one k on validation data. If even full donor patching fails, debug the hook/readout. If full patching works but subspace patching consistently fails, the current E2 architecture has failed its causal gate; do not spend time on tangent and retraction sweeps.

**Second job, only after that gate passes: compare a small set of methods.**

| Condition | Question it answers |
|---|---|
| Unsteered and CAA | Basic reference levels |
| Retraction only, zero directional step | Does the correction itself create the apparent benefit or damage? |
| B: original ODESteer | Does the proposed change help its intended baseline? |
| C: unit-normalized projected ODESteer | Is subspace restriction sufficient? |
| C plus endpoint-only retraction | Is repeated manifold following necessary? |
| E2: tangent projection plus normal correction | Main hypothesis |
| Spline reference | Does a known curved path work in the same intervention space? |
| Norm-preserving spherical flow using the same probe | Does norm control explain the gain? |

Use the same stored activations, target labels and source prompts where applicable; acknowledge the spline's extra ordering information. Start with red as target and 90°, 120°, 150° source bands; keep 180° separate. Use a short validation strength grid, one default neighborhood and retraction setting, and 10 integration steps for both iterative methods. Check 40 steps at selected operating points only if discretization could explain the result. Save all intermediate states, but decode a common fixed number of checkpoints for path comparisons.

Choose operating strengths on validation data to approach a predeclared target-success level, such as 80%, then freeze them for test. Report the whole success/quality frontier as well: an operating point that misses the threshold is not matched success. Use at least three fitted seeds for the confirmatory pilot and paired, prompt-level uncertainty with seed variation shown separately.

Primary outputs for the generation hypothesis: target success/probability, endpoint distance to independently held-out natural target-output distributions, gray/brown/other mass, and runtime. For the separately predeclared path-control hypothesis, output naturalness along a common set of decoded checkpoints is primary at matched target success. Also report distance to independent reference activations, measured path length, zero/small-T displacement and gate coverage. This establishes a controlled colour mechanism; it does not establish general fluency preservation. General generation and same-context content preservation are the next stage only after a convincing pilot.

**Continue rule:** before examining outcome data, predeclare practically useful improvements for each hypothesis and the multiple-comparison procedure if both are tested. For the generation hypothesis, an example is at least 20% lower endpoint output deviation at comparable success, with paired uncertainty supporting a gain over C and endpoint-only correction. The 20% is an illustrative project decision, not an evidence-based scientific standard or an already fixed threshold. Define a separate path-control criterion if pursuing that objective. Require stable direction across fitted seeds and confirm on another target hue before expanding.

**Stop or redirect:** C matches E2; endpoint correction matches E2; only training-density metrics improve; or improvements disappear under a well-tuned norm-preserving control. If only intermediate paths improve, the evidence supports a path-control/interpretability project and does not yet support better generation. If E2 stalls despite a causally useful manifold and nonmonotone score, add E3 as a specific follow-up rather than starting an unlimited variant search.

Defer the full weekday reproduction, geography, emotion, SAE extraction and alignment benchmarks. Use a tiny weekday smoke test only if needed to debug hooks. More synthetic shape sweeps have lower information value than this real activation test.

## Corrections needed before implementing the full plan

- **Prior-work dimensions and controls:** [Wurgaft et al., §2.1 and Appendix A](https://arxiv.org/html/2605.05115) say **3D** visualization and **64-dimensional** steering, not 33D. Their reported linear intervention replaces the full residual while their manifold intervention preserves the complement of PCA. A new comparison should also hold the intervention subspace fixed. Their output distribution retains an `other` bin; their energy uses 50 sampled waypoints. Reproducing the quoted ratio requires matching that protocol, not merely running CAA at another layer.
- **Emotion evidence:** [Sun et al., §§3–4](https://arxiv.org/html/2604.03147) fit VA directions using emotion-label ratings and evaluate word projections against the 44,728-word NRC-VAD lexicon. Their generated text is evaluated with VAD-BERT and VADER. A circle of emotion steering vectors is not evidence that individual generation activations occupy a hollow ring, and neutral emotion need not be unnatural. The proposed lexicon-angle success metric needs independent validation.
- **RMS details:** [Riemannian-Manifold Steering, §8](https://arxiv.org/html/2605.24942v2) explicitly reports weekdays, months, letters and ages. Its path-naturalness advantage holds on the smaller two output spaces, not all four. Avoid the plan's blanket natural-path claim.
- **Numerical accounting:** Euler steps perpendicular to x increase squared norm by the squared step length; they do not exactly rotate at fixed norm. E2's retractions also mean T is a proposal budget, not its actual traveled distance. Log net step lengths, correction lengths, norm changes and latency rather than asserting matched T equals matched distance.
- **Code/documentation mismatches:** `toy/geometries.py` uses 60 steps, versus 40 for the ring. Its `max_gap` is measured in projected coordinates, not full activation space. `graph_dir` returns an unwhitened cross-covariance, not a general least-squares regression slope; after a one-dimensional projection and normalization the direction can coincide, but this does not extend automatically to higher-dimensional manifolds. The local-PCA normal correction is an SCMS-style approximation; it should not inherit density-ridge guarantees without checking the assumptions.

The best immediate compute expenditure is the dense colour activation collection plus donor-patching gate. It tests the main unresolved assumption before investing in a steering system whose search space may not causally control the output.
