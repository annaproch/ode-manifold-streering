# Experiment plan: ODE steering constrained to a concept manifold

*Status: consolidated 2026-09-19; implementation added 2026-09-20. This is the single current plan, incorporating the design reviews and follow-up corrections. No LLM experiment has been run. Committed CPU toy results are in §9; additional seed-0 review diagnostics are identified separately. The new CPU/CUDA/MPS pilot code and safeguards have been written in [`pilot/`](pilot/README.md), but have not been executed or tested. The user explicitly requested stopping before execution.*

**Decision: pursue one bounded real-model pilot.** Run CPU controls and a 50–100-prompt colour/readout and donor-patching check first. Expand to a dense colour pilot only if the intervention space causally controls the answer. The broader roadmap is conditional (§6); the immediate work is the gated colour pilot.

## 1. Idea and intended contribution

ODESteer edits a hidden activation using a classifier gradient. Proposal E2 restricts that gradient to a concept subspace, projects it onto a locally estimated tangent, and applies a normal correction toward the observed data. E3 additionally replaces the classifier score with a graph-distance score, but is a follow-up variant rather than the first pilot's main method.

The research question is whether **estimating the concept's actual data geometry improves control beyond subspace restriction, endpoint correction and norm preservation**. The classifier is trained and the geometry is fitted; the potential economy is no additional neural geometry model.

Curved steering and density-based geometry are prior work. [GeoSteer](https://arxiv.org/html/2609.10658), submitted September 9, already combines a polynomial-count-sketch probe, spherical tangent projection and norm-preserving geodesic updates. It constrains the activation sphere rather than estimating a concept-specific data manifold. That distinction must earn its value experimentally; tangent projection alone is not the novelty claim. See §8.

## 2. Two separate hypotheses

Predeclare both hypotheses and their analysis before looking at pilot outcomes:

1. **Endpoint hypothesis:** at matched target success, E2 produces final output distributions closer to natural target outputs, with less unwanted change than simpler controls. The colour pilot tests this restricted claim; general fluency and content preservation require later generation experiments.
2. **Path-control hypothesis:** at matched target success, interventions at intermediate points produce more natural concept transitions. This is an independent interpretability/control objective, with path naturalness as a primary metric, following [Wurgaft et al., Appendix A.7](https://arxiv.org/html/2605.05115).

The hook computes its integration steps before returning the final activation. The downstream transformer does not consume the internal integration trajectory. Identical final activations under the same model/cache context imply identical downstream behavior. Thus a path improvement can support hypothesis 2 without establishing hypothesis 1; it is a valid result rather than an after-the-fact consolation outcome.

**Scope:** E2 depends on the score landscape restricted to the estimated manifold, not simply on Euclidean closeness to a target. In the horseshoe toy it stalls; E3 resolves that particular failure (§9.4). Neither has a general reachability guarantee. Near-antipodal starts may have a small tangent gradient and a slow/noisy start; keep 180° separate from the main 90°/120°/150° tests rather than assuming every concept has the toy's symmetry.

**Conditional geometry:** preserving membership in a colour manifold does not necessarily preserve saturation or lightness. Grey can be a valid on-manifold colour. Start with a hue curve at fixed saturation/lightness, and use other slices as controls. A dense hue sweep in the inputs does not prove that the model has a continuous activation curve.

## 3. Background: ODESteer and classifier interpretation

Source: [ODESteer code](https://github.com/ZhaoHongjue/odesteer), pinned by the toy setup to `8a3c481`; `src/odesteer/steer/_ode_steer.py`, `utils/kernels/`, `lm/_huggingface_lm.py`, and `confs/steer/ODESteer.yaml`.

Offline, the frozen model supplies positive and negative last-token layer activations. Each is L2-normalized and transformed by a fixed random degree-2 polynomial count sketch (8000 features, gamma = 0.1, coef0 = 1). Logistic regression (`max_iter=1000`, default C = 1) is the trained component.

The normalized gradient flow is integrated by Euler steps:

```text
dx/dt = grad h(x) / ||grad h(x)||
10 steps, step size T/10
```

The original hook edits the last prompt token during prefill and each newly generated token thereafter. **This pilot edits only the last prompt token and evaluates the first answer token.** Fitting at that site does not establish applicability to subsequent generation positions.

Important implementation distinctions:

- The raw classifier logit estimates posterior log-odds. `log_dre(x)` adds `log((1-pi)/pi)`, with pi the training positive fraction, to estimate the log density ratio. `predict()` instead tests posterior probability > 0.5. Therefore `log_dre > 0` is equivalent to posterior probability > pi, not > 0.5.
- All-negative `predict()` results in the seed-0 toys do not mean that the model learned nothing. Corrected ring positives and E2 endpoints pass `log_dre > 0`; training ranking is strong (§9.6). The prior term is constant, so it does not change gradient direction. The code also scales its gradient by a positive prior factor, which cancels in normalized directions.
- Success is evaluated using toy ground truth or LLM output, independently of the steering score. Report held-out ranking, calibration, class balance and the restricted score landscape; classifier training accuracy alone is insufficient.
- Because h uses normalized x, its gradient is orthogonal to x. The continuous flow preserves norm, but a finite Euler step increases squared norm by the squared step length. It is not an exact rotation.
- T is the sum of proposed unit-gradient step lengths for B, up to numerical normalization. It is **not** the actual path length of E2 after its normal corrections. Measure actual movement, correction lengths, endpoint displacement, norm change and runtime.
- The yaml default is T = 1; the paper's Llama-3.1-8B settings are around 4–6. Do not transplant toy T values to the LLM. Code layer index 13 probably corresponds to the paper's "layer 14" (inferred; the paper does not state its indexing); record exact zero-based hook indices for every pilot configuration.

## 4. Method and required controls

### 4.1 Fitted components

Fit only on training activations from the exact prompt template and token position being intervened on. Keep validation/test hex identities separate before fitting.

| Component | Construction |
|---|---|
| Mean μ and subspace U | PCA of training activations; choose dimension by causal donor-patching validation |
| Local geometry | Stored projected points z = Uᵀ(x − μ), kNN local PCA, KDE bandwidth chosen on training/validation data |
| Target score h | ODESteer's polynomial sketch and logistic regression |
| Eligibility rule | Pre-intervention proximity/support test calibrated on validation data, including gray and unrelated controls |
| E3 score, deferred | Shortest-path distance to a specified positive set on a kNN graph |

Only the concept coordinates move in E2. The residual is preserved. A probe on that residual necessarily gives the same answer before and after intervention; it measures redundancy, not causal conflict. Test causal sufficiency using target-donor coordinate patches (§5.2).

### 4.2 Current toy mechanism and pilot implementation requirements

The following is the existing toy mechanism, retained to explain the archived results. It is **not yet an acceptable zero-strength pilot implementation**:

```text
z = Uᵀ(x − μ)
a = normalize(P_T(z) Uᵀ grad h(x))
z = z + (T/n) a
repeat r times:
    z = z + (I − P_T(z)) bw² grad_z log p(z)
x_new = x + U(z − Uᵀ(x − μ))
```

P_T is the projector onto the top m local PCs of the k nearest stored points. The normal correction is SCMS-style; local-PCA normals are an approximation rather than a basis for claiming exact density-ridge guarantees. Plain mean shift pulls along the manifold toward density bumps and stalls in the ring toy as r grows; keep it as a deferred ablation (§9.3).

**The zero-strength problem:** the correction has no T factor and still changes x at T = 0. Seed-0 diagnostics moved noisy ring starts by median 0.063 R; grey-centre starts are bimodal, with 57–65% captured onto the ring (§9.6). Projecting an arbitrary activation onto a colour manifold can create the concept instead of preserving it.

Before using E2 in the pilot:

- Make the deployed operation exactly the identity at T = 0. Separately retain a **retraction-only diagnostic control**, with zero directional step and the same correction count, to expose the correction's own effect.
- Check a sequence of small positive T values. An early return at zero alone is insufficient: total displacement must approach zero as T approaches zero. Choose and document a correction budget that vanishes with strength; the precise rule is an implementation decision still to be tested.
- Freeze a near-manifold eligibility rule on validation data, applied to the original activation before any correction. Reject unsupported starts rather than pulling them onto the fitted concept. Monitor departures during integration and define a bounded fallback; do not silently repair arbitrary excursions.
- Report eligibility coverage and rejected cases. Compare methods on the same eligible set and also report full-set results. A/B/S run ungated, including gray, brown and unrelated starts; geometry methods use identity on rejected inputs. R0 remains an ungated diagnostic. Report unrelated-prompt full-vocabulary endpoint KL and displacement, not merely gate acceptance.
- Validate prompt/token support as well as projected proximity. An unrelated activation can project near the curve while having an incompatible residual. No claim about steering every generated token is warranted yet.

With U = I and all geometry operations/gating disabled, the baseline integration must recover ODESteer's normalized Euler vector field at the same step count. The new implementation is standalone rather than subclassing upstream `BaseODESteer`, avoiding an install-time dependency on the unlicensed upstream repository. It includes an optional numerical parity check using a user's existing upstream checkout with shared sketch hashes/weights. Corrections are explicit integration operations, not only a replacement vector field. Safeguards may change toy results; identify the implementation version rather than reusing archived numbers as validation of a new algorithm.

The untested implementation clips each E2 correction to `correction_ratio * (T/steps) / retractions`, with provisional default ratio 0.25, and freezes the last accepted state on a support exit. Thus the correction budget vanishes with T; R0 deliberately retains unbounded strength-independent corrections as a diagnostic. These are explicit design choices to inspect in the preflight, not evidence of validated behavior. See [`pilot/README.md`](pilot/README.md) for profiles, stage commands, and artifact semantics.

### 4.3 E3 and optional variants

E3 assigns each stored point a negative shortest-path distance to the target set and estimates a local direction of increasing score. Target sets may use labels, a validated density-ratio threshold, or a predeclared high-score subset; labeled positives are not required. A raw `predict()` threshold would be empty in the seed-0 toys. High-score selection must specify its threshold or count and be validated independently.

Discrete graph routes have monotone shortest-path scores; local smoothing, tangent projection and finite corrections do not automatically preserve that property. The current `graph_dir` computes cross-covariance rather than a general least-squares slope. After one-dimensional tangent projection and normalization directions can coincide; a higher-dimensional implementation needs an explicit estimator.

Keep E3 for a measured score/topology failure after E2's causal gate passes. Deferred ablations are additive density guidance (E1), plain mean shift (E2-ms), step halving (E2-h), and residual motion (E2-full). Additive terms need separate normalization because classifier and density-gradient scales differ. E2-full would use the change in a kernel regression of full x on z; adopt it only after causal patching demonstrates the need, not merely because a residual probe decodes the source.

## 5. Active pilot: colours

### 5.1 Readout and data

Use one pinned Llama-3.1-8B checkpoint, recording base versus instruct and exact token formatting. Start from causalab's task:

```text
Question: Which color name best describes the hex code {hex}? Choose one of: red, orange, yellow, green, blue, purple.
Answer:
```

The existing fixture was validated on Qwen3-4B; validate the chosen Llama checkpoint rather than assuming transfer. First run 50–100 readout prompts. **Hardware:** the user has 128 GB unified memory. The implementation provides CPU float32, Apple Silicon MPS float16 and CUDA bf16 profiles. Llama-3.1-8B weights alone require approximately 32 GB in float32 or 16 GB in 16-bit precision, plus runtime overhead. Memory capacity does not make the full CPU evaluation practical; use CPU for numerical preflight/small checks and an accelerator for the expanded pilot. Quantizing would change activations and require a distinct experiment. Validate the six-way readout and an extended eight-way version with gray and brown separately. Require extended chromatic accuracy at least 90% of six-way accuracy, correct gray/brown controls, useful absolute performance and adequate answer-token probability. Freeze acceptance criteria before evaluating the full pilot.

Leading-space answer words, including gray/grey/brown, were checked as single tokens in the Llama-3.1 tokenizer. Recheck the exact checkpoint/context and aggregate valid spelling/case variants. Keep an **other** bin for all remaining vocabulary probability; do not renormalize away invalid answers.

For the first red-target classifier, label training sweep stimuli positive when the unsteered model answers red, and negative otherwise, so the target matches the measured readout. Report the resulting hue range, class balance, and behavior on neighboring orange/purple stimuli. Use the same target labels for classifier-based controls and keep target-success evaluation on held-out model outputs.

Sample the **primary curve** densely: HSL saturation 0.8, lightness 0.5, hue step 0.25°, deduplicated after rounding to 8-bit hex. The earlier 2,000–3,000-stimulus aspiration must not be met by counting unused context slices or duplicated hexes: RGB quantization limits distinct colours on this fixed curve. Record actual primary counts before collecting activations. Other saturation/lightness slices use a coarse 6° step for plots; evaluate them only as separate conditional experiments. Split by hex identity into 45% train, 10% independent reference, 22.5% validation and 22.5% test. Reserve a second template for transfer, with the same hex-level split; repeated templates are not independent colours.

The full pilot requires at least 20 primary starts in each 90°/120°/150° band and 10 in the separate 180° band in both validation and test, after deduplication/subsampling. These minima are coverage safeguards, not proof of statistical power. Collection checks raw counts, selection checks eligible main-band counts per seed, and confirmatory reports require sufficient eligible test counts. Require at least eight independent model-labeled target-reference outputs; inspect reference-density sensitivity. The reduced CPU smoke profile is explicitly exploratory. Do not infer feasibility or success from an unexecuted default configuration.

The bundled 600 causalab stimuli form six narrow hue clusters (about ±6° around class centres), so they cannot supply tangents through the gaps. Use them for readout checks and held-out evaluation after deduplication against the generated data. Main starts 90°, 120° and 150° toward red need generated stimuli; 150° gives a stronger toy contrast than 120°. Keep 180° separate. Verify the model's unsteered answers and local activation continuity at all these starts.

[sae-manifold](https://github.com/goodfire-ai/sae-manifold) uses a different colour template and layer 19; its colour geometry includes hue, saturation and lightness rather than a universal hollow ring. Use it to motivate candidate sites, not to reuse a mismatched manifold. The `burkelibbey/colors` dataset has hex codes and generated descriptions, not clean unique colour names. Plot PCs against all colour coordinates and include grey controls before interpreting geometry.

### 5.2 Causal gate: patch real donor activations first

Cache a few candidate layers around the previously studied colour and late-answer sites; record exact zero-based indices. Fit PCA on training data only. Compare k = 8, 16, 64 using validation data, with a smaller exploratory k only if warranted.

For source x_s and a real target donor x_t at the same site, apply:

```text
subspace donor patch: x_s + U Uᵀ(x_t − x_s)
full donor patch:     x_t
```

Prefer donors from the same saturation/lightness slice. The full patch is a diagnostic upper control, not a fair low-side-effect steering competitor. Check changes in answer probabilities, output validity and whether neighboring hues remain neighbors across splits.

If full donor patching fails, debug the hook/readout. If full patching works but subspace patching consistently fails across the candidate configurations, the current E2 architecture has failed its causal gate; do not launch tangent/retraction sweeps. Select one layer and k on validation data only after this test passes. Decodable hue or a pleasing PCA plot is insufficient.

### 5.3 Conditions, metrics and analysis

Run a small comparison first; hold classifier/data, source prompts, intervention site and eligibility set fixed where applicable. Acknowledge spline ordering information and any method-specific fitting costs.

| Condition | Purpose |
|---|---|
| Unsteered | Original behavior and exact no-op reference |
| A: CAA | Linear mean-difference reference |
| R0: retraction only, zero directional step | Measure effects of correction independent of target guidance; diagnostic exception to deployed no-op rule |
| B: ODESteer | Intended unconstrained baseline |
| C: unit-normalized projected ODESteer | Is subspace restriction sufficient? Normalize after projection |
| C-end: C followed by endpoint-only normal correction | Is repeated manifold following necessary? |
| E2: tangent projection plus bounded normal correction | Main proposed method, with §4.2 safeguards |
| F: spline path in the same intervention subspace | Does a known curved path work with matched residual handling? |
| S: norm-preserving spherical flow using the same probe | Diagnostic for finite-step norm drift; not a full GeoSteer reproduction or an independent novelty test |

For C-end use the same normal-correction primitive, with its budget chosen on validation data, and report correction cost. For F, hold the orthogonal residual fixed as for C/E2. Wurgaft's published full-space linear versus PCA-manifold comparison uses different residual handling; reproducing its number and isolating geometry are distinct comparisons.

The normalized-input classifier already has a gradient tangent to the activation sphere. Consequently B and S can coincide closely at small step sizes; S mainly removes finite Euler norm growth. Retain it as a numerical diagnostic, exclude it from the confirmatory superiority family, and do not treat B≈S as evidence against or a reproduction of GeoSteer. A claim against that method still requires its actual objective, schedule and evaluation protocol.

Start with one default neighborhood/retraction configuration and a short validation strength grid. Calibrate raw T before outcomes: use multipliers [0, 0.25, 0.5, 1, 2, 4] times the training median of target-to-other chromatic centroid distances at the selected layer. Median training activation norm is a predeclared alternative. Save the scale, multiplier and raw T. Use 10 integration steps initially; test 40 at selected points only if discretization could explain a difference. Record m, neighbors, bandwidth, correction budget and r. Save every solver state; decode endpoints for strength selection, then the common checkpoint grid at frozen strengths for validation/test path comparisons. Pad support-exit trajectories with their frozen state; C-end correction replaces the terminal state without adding a time tick. A wider r sweep is a later sensitivity check.

Before steering, calibrate the identity patch on real validation activations using singleton and repeated-prompt batches versus mixed collection batches, at the intended precision/device. The configurable default maximum probability difference is 0.02; inspect bf16 behavior before trusting this threshold and never widen it automatically. Estimate decode counts before loading weights. Reuse identical prompt/layer/state outputs and enforce an explicit run budget; the CPU profiles intentionally cap evaluation at 1,000 prompt-state decodes.

| Metric | Role |
|---|---|
| Target argmax and target probability | Success, independent of h |
| Endpoint output deviation from held-out natural target distributions | Primary quality metric for the endpoint hypothesis; fix the reference and distance before testing |
| Path output deviation from an independently fitted natural behavior reference | Primary for the path-control hypothesis; use Bhattacharyya/Hellinger geometry and common checkpoint count/parameterization |
| Gray, brown and other probability | Separate output diagnostics at endpoint and along path |
| Circular concentration ρ | Secondary uncertainty/concentration diagnostic, not physical saturation |
| Distance to independent reference activations, in full space and concept coordinates | Support diagnostic, normalized to natural held-out distances; training KDE density alone is insufficient |
| Actual path length, correction lengths, endpoint displacement and norm change | Separate geometry from numerical budget |
| Zero/small-T displacement, eligibility coverage and rejection behavior | Required intervention validity checks |
| Wall time, per-token overhead and fitting cost | Practical tradeoff |
| Residual hue probe | Optional redundancy diagnostic; causal gate remains donor patching |

Define ρ = |Σ_c p_c exp(iθ_c)| using chromatic options at centres red 0°, orange 30°, yellow 58°, green 120°, blue 235°, purple 285°, with p retaining gray/brown/other mass. Uneven centres and ordinary boundary uncertainty affect ρ; a mixture of chromatic predictions can lower it without representing grey, and brown can be chromatic. Compare to natural boundary examples rather than labeling every decrease as damage.

For E_BC-style path evaluation, define the natural reference and checkpoint rule before testing. The published weekday protocol sums distances at 50 waypoints; a raw integral over method-specific T is not a comparable naturalness score. The untested pilot implementation uses nearest independent reference output distributions, with target-only references for endpoints, and common normalized solver-step checkpoints. This finite-sample support metric is not an exact reproduction of the published behavior-spline E_BC. Report that distinction and check reference-density/parameterization sensitivity before a stronger comparison.

Choose operating strengths on validation data to approach a predeclared success level (80% is a candidate, not yet frozen), then freeze them for test. Report success/quality frontiers too: a method that misses the threshold has no matched-success operating point. Report nominal T, measured length and endpoint displacement separately; equal T does not mean equal distance for E2. Use paired prompt comparisons, at least three fitted seeds for the confirmatory pilot, and prompt-level bootstrap intervals with seed variation reported separately; nested start samples are not independent fitted worlds.

Before outcome inspection, define a practically useful improvement and the analysis for both hypotheses, including multiple comparisons if both are tested. **20% lower endpoint deviation is only an illustrative project threshold, not a fixed or evidence-based criterion.** Define a separate path-control threshold if that objective is pursued. Require consistent direction across seeds and confirm another target hue before expanding.

### 5.4 Reading outcomes and continuation

| Observation | Decision |
|---|---|
| E2 improves endpoint quality at comparable success over B, C and C-end | Continue the endpoint hypothesis; confirm another target and later generation task |
| E2 improves path quality alone | Continue the predeclared path-control objective if its criterion is met; no generation-quality claim |
| C or C-end matches E2 on the relevant metric | Prefer the simpler method; repeated manifold following has not earned its cost |
| B and S nearly coincide | Expected for tangent gradients and small Euler steps; S supplies little independent evidence |
| B and S differ materially | Inspect finite-step norm drift and integration sensitivity before attributing gains to concept geometry |
| Only training-density metrics improve | Insufficient evidence of useful control |
| R0 causes much of the apparent effect | Attribute it to correction, not target guidance; examine off-concept damage |
| Full patches work but subspace patches fail | Reconsider the representation before tuning steering |
| E2 stalls with useful causal geometry and a measured score obstruction | Add E3 as a targeted follow-up; do not assume a guarantee |
| Gating rejects most intended inputs or misses unrelated activations | The applicability claim fails; avoid blanket per-token deployment |

## 6. Execution order and deferred extensions

### 6.1 Run now

1. **CPU preflight:** implement/check no-op and small-strength behavior, R0 and eligibility diagnostics. Existing toys already justify a real-model check; avoid another broad shape sweep. Verify that geometry-disabled code reproduces the baseline and preserve archived results separately from revised algorithms.
2. **Smoke test on the pinned Llama-3.1-8B checkpoint:** 50–100 readout prompts followed by a small activation collection, no-op calibration and donor-patching check. The 128 GB host has room for the stated weight sizes plus overhead; CPU latency still needs measurement. Do not substitute a smaller checkpoint unless it separately passes the readout gate.
3. **Expanded colour pilot after the gate passes:** check deduplicated primary coverage and decode budgets before collection. Run the §5.3 comparison on one checkpoint/site. PCA, classifiers, graphs and analyses are CPU-friendly; use MPS/CUDA for the full model-forward workload. Select strengths from validation endpoints, freeze them, then decode selected validation paths and the confirmatory test. Scoped code hashes are provenance, not enforced cache keys: report/comment changes must not trigger recollection. Use a new output directory for substantive algorithm changes.
4. **Continue only against the predeclared hypothesis:** confirm another target hue and a held-out prompt template before broader tasks. Report failures as well as successful gates.

### 6.2 Conditional follow-ups, not current milestones

- **Weekdays:** optional tiny hook/readout smoke test; defer full reproduction. The causalab 49-prompt demo and Wurgaft's published path protocol differ in sites and interventions. Reproduce exact settings before demanding its crossing counts or 2.8× energy ratio. Seven answer clusters have gaps that disadvantage local tangents. A weekday result alone does not settle continuous-manifold steering.
- **Geometry-driven E3:** introduce only after diagnosing the restricted score or graph need. A semantic country/continent hierarchy does not establish a branched activation manifold; measure support and topology first. A branch junction is not a smooth one-dimensional manifold.
- **Emotion or another practical behavior:** first show an appropriate manifold at the actual generation sites and validate eligibility. [Sun et al.](https://arxiv.org/html/2604.03147) derive VA directions from emotion-label ratings and study projected emotion steering vectors. The 44,728-word NRC-VAD lexicon is a word-projection validation resource; generated text is scored with VAD-BERT and VADER. A circle of steering vectors is not proof that individual token activations form a hollow ring. Neutral states can be natural, and lexical-average angles/minimum radius are not prevalidated success/naturalness metrics. Choose start/target pairs from measured geometry, and validate text-level metrics before preregistering them. Measure content preservation and unrelated task performance; refusal changes can be part of the intended affect mechanism.
- **SAE-discovered geometry:** possible route to less hand-specified concept subspaces, using sae-manifold clustering or related tools, after the basic method demonstrates value. The classifier and geometry still require fitting; do not claim training-free operation.
- **TruthfulQA/RealToxicityPrompts and honesty:** conditional on a defined, causally useful intervention space and token-support rule. There is no established manifold here in this plan. Determine dataset/service access and evaluation needs when this phase is justified; linear geometry may make E2 unnecessary.

Colour success establishes feasibility. A stronger methods paper would need an advantage over existing geometric controls and generalization to another concept or task; path-control and generation papers can have different valid endpoints.

## 7. Risks and open decisions

- **Support and conditioning:** a pooled data manifold can permit exactly the saturation/lightness changes the application wants to forbid. Fixed-template, fixed-token and conditional-slice fitting are part of the initial scope.
- **Retraction and eligibility:** support membership does not follow from projected distance alone; zero/small-T behavior and gate coverage are go/no-go checks (§4.2), especially before any every-token claim.
- **Causal sufficiency versus redundancy:** residual decodability alone is not interference. Donor patches distinguish an unusable subspace from redundant coding. Higher k can improve control while making local density/tangent estimation less reliable.
- **Score/topology:** classifier gradients can stall; discrete graph scores can shortcut, disconnect or produce poor continuous directions. Normalizing a tiny tangent gradient can amplify noise. Define numerical thresholds and fallback behavior on validation data.
- **Finite steps and cost:** correction lengths and step count confound nominal T. Fixed steps can hover at the target; record the effect, and apply any stopping/step-control comparison fairly to B as well as E2.
- **Readout and evaluation:** colour classification uncertainty is not desaturation. Preserve other-token mass, validate boundary examples and keep evaluation reference data independent of method fitting. Separate selection and test data.
- **Extrapolation:** support constraints intentionally limit access to unobserved regions; improved on-support control is not a promise of novel concept combinations.
- **Scale:** kNN/KDE can become unreliable at large k or sparse sample counts. A learned geometry model is a possible redesign, not an automatic extension of a successful low-dimensional pilot.
- **Provisional implementation defaults:** a correction budget vanishing with T, distance-based gate and identity/freeze fallbacks, neighborhood/m, layer candidates and nearest-reference output estimator are now specified in the untested code. Inspect them in preflight and validation; record any changes in a new run directory. Checkpoint identity and practical thresholds still need to be fixed before confirmatory evaluation. Default practical-effect thresholds are null, so default runs are exploratory.

## 8. Related work and novelty

| Work | Verified relevance / role |
|---|---|
| [ODESteer](https://arxiv.org/abs/2602.17560) | Original classifier-guided flow and main baseline |
| [Wurgaft et al., Manifold Steering](https://arxiv.org/html/2605.05115) | Spline paths and density-weighted geometry; Llama-3.1-8B layer 28, 64-D PCA steering and 3D visualization. Published path evaluation uses 50 waypoints and retains other-token mass. Its linear/full-residual versus manifold/PCA-residual handling needs a matched-subspace control here |
| [GeoSteer](https://arxiv.org/html/2609.10658) | Method §3.2 verified: polynomial-count-sketch probe, spherical tangent gradients and exact geodesic norm preservation. Direct control for whether data-estimated geometry adds value; benchmark results have not been independently reproduced |
| [Riemannian-Manifold Steering](https://arxiv.org/html/2605.24942v2) | Output-Hellinger pullback geometry with a learned encoder, no per-prompt labels. Reports weekdays, months, letters and ages; path-naturalness gains hold on the smaller two output spaces, not all four. Potential later reference if implementation is available |
| [Curveball](https://arxiv.org/abs/2603.09313) | Polynomial-kernel-PCA steering and geometric-distortion analysis; nonlinear steering is established prior work |
| [Engels et al.](https://arxiv.org/abs/2405.14860) | Circular weekday/month features and causal interventions |
| [Sun et al.](https://arxiv.org/html/2604.03147) | VA/emotion evidence; distinguish fitted directions, class vectors and token activation support (§6.2) |

Additional leads, not relied on for quantitative claims in the pilot: [FishBack](https://arxiv.org/abs/2605.17231), [dual steering](https://arxiv.org/html/2602.15293v2), [FLAS](https://arxiv.org/abs/2605.05892), [UniSteer](https://arxiv.org/abs/2605.30076), [non-surjective steering](https://arxiv.org/abs/2604.09839), and [steerability prediction](https://arxiv.org/abs/2606.11599). Recheck methods and quantitative claims before comparing. Diffusion analogies include [MPGD](https://arxiv.org/abs/2311.16424), [DPS](https://arxiv.org/abs/2209.14687), [MCG](https://arxiv.org/abs/2206.00941), and the [SCMS reference](https://jmlr.org/papers/v12/ozertem11a.html); their precise applicability is not a theorem supplied by this plan.

E3 resembles classic graph-geodesic/Isomap ideas and existing geodesic steering. It has no established novelty claim here. Search this literature specifically before writing up an E3 contribution. Neither estimating an unknown shape nor avoiding a fixed output-token schema guarantees an advantage over a suitably adapted baseline.

## 9. Synthetic evidence and source checks

The original sweeps below are from committed [`toy/ring_toy.py`](toy/ring_toy.py), [`toy/run_all.py`](toy/run_all.py), and [`toy/geometries.py`](toy/geometries.py). Full results: [`tables.md`](toy/results/tables.md) and [`geometries.md`](toy/results/geometries.md), with raw rows alongside them. `toy/setup.sh` fetches ODESteer at `8a3c481`; it is not vendored because the repository has no license file.

Sections 9.1, 9.3 and 9.4 summarize **three fitted worlds × two sets of 32 noisy starts**. Ring runs use 40 Euler steps; horseshoe/Y runs use 60; original ODESteer uses 10. These are mechanism checks, not LLM evidence, and use the unguarded correction in §4.2. The separate seed-0 review diagnostics in §9.6 were run interactively and are not regenerated by the committed sweep scripts.

### 9.1 Ring toy: initial mechanism checks

Setup: a hue ring of radius R = 3 plus a lightness axis in a 3-D subspace of a 256-D space, mean offset of norm 20, isotropic noise 0.3, 3000 points; positive = hue within ±30° of red. Classifier = ODESteer's `NormedPolyClassifier` (8000 features, gamma 0.1, coef0 1); posterior-threshold training accuracy 0.831–0.836 across worlds (interpretation corrected in §9.6). KDE bandwidth by 5-fold CV: 0.228.

Success (final hue within ±30° of red) and lowest radius ÷ R, T = 8 unless stated:

These archived conditions include D (the full-space classifier direction plus a normalized density term in U) and E1 (the corresponding additive combination restricted to U). The ⟂ variant removes the density term's component along the classifier direction. They are not additional required arms of the first real-model pilot.

| Condition | From 120°: success · radius | From 180°: success · radius | Notes |
|---|---|---|---|
| A: CAA (linear) | 1.00 · 0.87 | 1.00 · 0.09 | ends off the data (log p −100 from 120°) |
| B: ODESteer | 1.00 · 0.88 | 1.00 · 0.09 | ends off the data (log p −85); ≈ A and C in this toy |
| C: projection only | 1.00 · 0.88 | 1.00 · 0.09 | same as B |
| D: density only, λ = 1 | 0.00 (0.80 at T = 10) · 0.94 | 0.00 · 0.90 | drifts out of the subspace (up to +3.4 off-plane) |
| E1, λ = 0.3 | 1.00 · 0.94 | 1.00 · 0.19 | crosses the centre from 180°; ends off the data |
| E1, λ = 1 | 0.29 (0.99 at T = 10) · 0.94 | 0.00 · 0.81 | on the data from 120°; stuck from 180° (hue 94–124°) |
| E1, λ = 3 | 0.00 · 0.96 | 0.00 · 0.95 | stuck near the start at every T (hue 119–163°) |
| E1 ⟂ (density term at right angles to ∇h), λ = 1 / 3 / 10 | 1.00 / 0.03 / 0.00 | 0.93 / 0.00 / 0.00 · radius 0.34 / 0.51 / 0.75 | either crosses or stalls |
| E2 first version (plain mean shift, r = 3), T = 10 | 0.97 · 0.95 | 0.20 · 0.93 | on the data; slow from 180° |

The archived `clf_pos` columns use the posterior threshold 0.5, without the class-prior correction in `log_dre`. Their positive/negative contrast is not evidence that the density-ratio score rejects real red states; see §9.6. Use geometric ground truth for toy success and the model output for LLM success.

### 9.2 Current interpretation of the toy evidence

The additive-density and plain-mean-shift failures motivate the tangent/normal split. The toy antipodal condition is a slow start, not a universal stall. E3 resolves the tested horseshoe failure but changes the score as well as the constrained path. No toy tests language-model output quality, causal sufficiency of a real concept subspace, or support across generated token positions.

The follow-up review corrected posterior-versus-density-ratio interpretation, exposed nonzero zero-strength retraction, separated path and endpoint hypotheses, added C-end/R0/spherical controls, and replaced the broad immediate roadmap with the gated colour pilot. These are incorporated in the active sections above rather than left as competing recommendations.

### 9.3 Ring toy: fixed-step and retraction sweeps

Grid: start ∈ {60, 90, 120, 150, 180}°, T ∈ {10, 20, 40}, r = 3; separate r sweep r ∈ {1, 3, 10, 30}. Success / lowest radius ÷ R:

| Condition | from 120°, T = 10 / 20 / 40 | from 150° | from 180°, T = 10 / 20 / 40 |
|---|---|---|---|
| B: ODESteer | 1.00 / 1.00 / 1.00 · 0.88 | 1.00 · 0.51 | 1.00 / 1.00 / 1.00 · **0.09–0.13** (through the centre) |
| E2-ms: plain mean shift | 0.97 / 1.00 / 1.00 · 0.95–0.97 | 0.96–1.00 · 0.95–0.97 | 0.20 / 0.98 / 0.99 · 0.93–0.94 |
| **E2: SCMS, fixed step** | 0.92 / 1.00 / 1.00 · 0.94–0.95 | 0.87 / 1.00 / 1.00 · 0.94 | 0.64 / 0.93 / 1.00 · 0.91–0.92 |
| E2 + stop on reversal (not adopted) | 0.84 / 0.93 / 0.97 | 0.78 / 0.89 / 0.94 | 0.23 / 0.39 / 0.71 |
| E2-h: halve on reversal | 0.84 / 0.94 / 1.00 | 0.75 / 0.90 / 0.99 | 0.22 / 0.53 / 0.86 |

- **Retraction:** success at r = 1 / 3 / 10 / 30 (120°, T = 20): E2-ms 1.00 / 1.00 / 0.36 ± 0.49 / **0.00**; E2 (SCMS) 1.00 / 1.00 / 1.00 / 0.99. From 180°, T = 20: E2-ms 0.93 / 0.98 / 0.42 / 0.00; E2 0.94 / 0.93 / 0.98 / 1.00.
- **Step length:** with fixed steps, E2's final swing is ≈ one step length (3.7° / 8.4° / 17° at T = 10 / 20 / 40); B's is 0.1–2.4° because it settles at the classifier's off-data peak.
- **Antipodal starts:** E2 reaches red from 180° while staying on the ring (radius ≈ 0.91), with somewhat more T than from 120°; ODESteer passes through the centre (radius 0.09).
- **Radius understates ODESteer's departure:** from 120° at T = 20 (single world) B's in-plane radius stays ≈ 0.9 R but it moves 3.6 units out of the subspace (log p ≈ −1500). Hence the distance-to-data metric in §5.3.
- Not testable in the ring toy: causal subspace sufficiency (its hue lives only in U by construction), the readout, and generation quality; these are handled by checks on real data (§5.2, §5.3, §6, §7).

### 9.4 Non-circular geometries: horseshoe and Y-branch

Setup ([`toy/geometries.py`](toy/geometries.py)): 1-D curves in a 3-D subspace of a 256-D space, same noise, offset and classifier recipe (posterior-threshold train accuracy 0.90–0.92, which is the majority-class baseline; see §9.6). **Horseshoe:** an arc of radius R from −150° to +150°; start at one tip, target at the other; the straight path crosses a gap of ≈ R, the on-data path is ≈ 5.2 R long and first moves away from the target. **Y:** three arms of length 2R at 0°, 120°, 240°; start at one tip, target at another; the on-data path goes through the junction. Success = within 0.5 R of the target tip; max gap = largest distance in the fitted PCA coordinates from the path to the nearest stored point ÷ R (≈ 0.06 means close in that projection, not a full-space manifold guarantee).

| Geometry | Method | Success at T = 2 / 4 / 6 / 10 / 20 / 40 | Max gap ÷ R |
|---|---|---|---|
| horseshoe | B: ODESteer | 0 at every T | 0.5–5.7 (leaves the data) |
| horseshoe | E2 | 0 at every T (0.01 at T = 40); progress ≤ 3% of the path | 0.06 |
| horseshoe | E3: graph score | 0 / 0 / 0 / 0 / 0 / **0.98** (progress 0.63 at T = 20) | 0.06 |
| Y | B: ODESteer | 0 at every T | 0.3–5.0 |
| Y | E2 | 0 / 0 / 0 / 0 / 0.12 / **0.94** | 0.06 |
| Y | E3: graph score | 0 / 0 / 0 / 0 / 0.58 / **1.00** | 0.06 |

Reading: the fitted score and tangent estimates make E2 stall on this horseshoe; replacing the score with E3 resolves that failure in these runs. This does not establish that a classifier score is always Euclidean closeness, or that E3 succeeds on arbitrary geometry. Both manifold methods use proposal budgets above the geometric route length. Tangent noise, normal corrections and score behavior can all affect progress. A Y junction lacks a unique one-dimensional tangent. These results motivate specific follow-ups, not a general branch-steering claim.

### 9.5 Source checks and remaining uncertainty

The local ODESteer implementation supports the classifier/prior and hook account in §3. Prior review checked causalab's colour template, clustered stimulus coverage, weekday demo and Llama tokenization; these still need validation on the selected checkpoint and exact prompt.

The paper review corrected Wurgaft's dimensions to 3D visualization and 64-D steering, identified the different residual handling in its published baselines, and checked its other-token bin and waypoint protocol. RMS tasks and restricted naturalness gains, Sun's lexicon-versus-generation evaluation distinction, and GeoSteer's §3.2 method were checked directly in the papers linked above. Code availability and comparative performance of all these methods have not been independently established here. No broad novelty-completeness claim is made.

### 9.6 Follow-up CPU diagnostics: classifier and zero-strength correction

These are additional **seed-0 diagnostics**, not six-replicate sweep estimates.

| Geometry | Posterior-threshold training accuracy | Fraction predicted positive at 0.5 |
|---|---:|---:|
| Ring | 0.8330 | 0 |
| Horseshoe | 0.9053 | 0 |
| Y | 0.9163 | 0 |

These values are majority-class accuracy under `predict()`, not evidence of no learned discrimination. Training ROC-AUC is 0.9897 for horseshoe and 0.9982 for Y; a noisy-coordinate proxy for ring labels gave 0.9870 and must not be reported as exact-label ring AUC.

An exact-positive-set check on the seed-0 ring gave pi = 0.167. `log_dre > 0` accepted every training positive and every E2 endpoint in a 150° start, T = 20, 40-step, r = 3 run. It also accepted 33.0% of training negatives, so a corrected score threshold is not an exact red-region classifier. Selecting the highest-scoring stored points gave unchanged E3 success (numbers below). Labels, score thresholds and high-score subsets remain alternatives to validate (§4.3).

With the unmodified E2 normal corrections, 40 iterations and r = 3, a seed-0 check gave:

| Start distribution | Directional T | Median full-space displacement / R |
|---|---:|---:|
| Noisy ring near 150° | 0 | 0.0633 |
| Noisy grey centre | 0 | 0.3113 (32 starts) |

The grey starts used the fitted world's true centre μ plus isotropic noise 0.3 in all dimensions. **This outcome is bimodal, so a median is not a stable summary.** With 256 starts from the same distribution and three noise seeds (0, 1000, 7): 57–65% end on the ring (end radius > 0.8 R); displacement percentiles 10th/50th/90th ≈ 0.05–0.07 / 0.74–0.79 / 0.90 R; medians of 32-start batches range 0.24–0.83 R. The 0.31 R above and the earlier 0.76 R are draws from this one distribution. Starts exactly at the in-plane centre (an unstable fixed point) or within ≈ 0.02 R of it barely move. Report the capture fraction, not a median. There is no claim that the current code already satisfies the §4.2 checks.

**Default ODESteer is near-linear in these toys, but the cut-across result is robust.** At default settings the classifier gradient has cosine ≈ 0.98 with the CAA direction at all tested starts, which is why A ≈ B ≈ C in §9.1. Refitting with C = 10³ and gamma = 10 (training balanced accuracy 1.00, held-out AUC 0.98, cosine with CAA ≈ 0.65) still gives lowest radius 0.09 R from 180°, 0.46 R from 150° and 0.81 R from 120° (seed 0, 64 starts, T = 10, 40 steps). So B's crossing is not an artefact of the underfit default classifier.

**E3 with classifier-derived positives (seed 0):** taking the top-k stored points by `log_dre` (k = number of true positives; precision 0.81–0.95) gives E3 success equal to true labels: horseshoe 0.98 vs 0.98 at T = 40; Y 0.89 vs 0.91 at T = 20.

These seed-0 diagnostics were run interactively and are not yet regenerated by a committed script.

## 10. References and code

- ODESteer (ICLR 2026): [paper](https://arxiv.org/abs/2602.17560) · [code](https://github.com/ZhaoHongjue/odesteer) · [OpenReview](https://openreview.net/forum?id=CFewUmgIIL)
- Goodfire Neural Geometry series: [overview](https://www.goodfire.com/research/neural-geometry)
  - Manifold steering: [paper 2605.05115](https://arxiv.org/abs/2605.05115) · [blog](https://www.goodfire.com/research/manifold-steering)
  - SAEs and manifolds: [paper 2604.28119](https://arxiv.org/abs/2604.28119) · [code](https://github.com/goodfire-ai/sae-manifold)
  - Block-sparse featurizers: [paper 2606.25234](https://arxiv.org/abs/2606.25234) · [code](https://github.com/goodfire-ai/block-sparse-featurizer)
  - Arithmetic in the Wild: [paper 2605.01148](https://arxiv.org/abs/2605.01148) · [code](https://github.com/goodfire-ai/arithmetic-wild)
  - Causal-abstraction toolkit (weekdays geometry demo, `hex_color` task, steering tutorial): [causalab](https://github.com/goodfire-ai/causalab)
- Colour data: [burkelibbey/colors](https://huggingface.co/datasets/burkelibbey/colors)
- Valence-arousal subspace: [Sun et al. 2026](https://arxiv.org/abs/2604.03147)
- Circular features: [Engels et al. 2024](https://arxiv.org/abs/2405.14860)
- Related steering work: see the table in §8
- GeoSteer: [paper and method](https://arxiv.org/html/2609.10658)
