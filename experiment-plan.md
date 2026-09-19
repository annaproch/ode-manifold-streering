# Draft experiment plan: ODE steering constrained to a concept manifold

*Status: draft, revised 2026-09-19 after a design review (§9). Nothing has been run on an LLM yet; only a synthetic CPU toy (§9.1).*

## 1. Idea in one paragraph

ODESteer steers an LLM by moving a hidden activation along the gradient of a classifier ("more positive-like"). It knows *which way is better*, but nothing keeps the moving activation where real activations actually live, so it can cut through empty regions of activation space. Concept-geometry work (Goodfire's Neural Geometry series) shows that many concepts lie on curved, low-dimensional manifolds. **Proposal:** keep ODESteer's gradient flow, but constrain it to the concept manifold: (a) restrict it to the concept's subspace and (b) keep it on the data inside that subspace. This is the manifold-preserving guidance idea from diffusion models (MPGD, DPS, MCG), applied inside interpretable concept coordinates. With the constraint switched off, it reduces to ODESteer exactly.

**Scope of the novelty claim (see §8).** Density-weighted steering geometry already appears in Wurgaft et al. (2605.05115, Definition 1), and geodesic and kernel-based curved steering exist (Riemannian-Manifold Steering, Curveball). The defensible contribution is: *ODESteer's gradient flow constrained to a manifold, compared with spline and geodesic manifold steering on shared metrics.* "Classifier guidance in concept coordinates" on its own is not new.

## 2. Hypothesis

At matched target success, the manifold-constrained flow keeps its intermediate states on-concept (colours: it stays saturated instead of passing through grey or brown; behavioural concepts such as emotion: it moves through neighbouring states instead of cutting across), and it causes fewer off-target side effects and less fluency damage than ODESteer.

**Known limit (from §9.1):** when the start is directly opposite the target (e.g. cyan → red), the part of ∇h that points along the manifold is close to zero, so no manifold-constrained gradient flow has a preferred direction there. Treat the antipodal case as its own condition and expect slow or stalled progress from it; its outcome is not a test of the main hypothesis.

## 3. Background: what ODESteer actually does (verified in its code)

Source: [ZhaoHongjue/odesteer](https://github.com/ZhaoHongjue/odesteer), `src/odesteer/steer/_ode_steer.py`, `src/odesteer/utils/kernels/`, `src/odesteer/lm/_huggingface_lm.py`, `confs/steer/ODESteer.yaml`.

**Offline (the LLM is frozen):**
1. Run the LLM on positive and negative texts. Keep the layer-L activation at the last token → `pos_X`, `neg_X` (N × d).
2. L2-normalise each activation, then map it through a **fixed random** degree-2 polynomial count sketch (`NormedPolyCntSketch`) with 8000 features, `gamma = 0.1`, `coef0 = 1.0`. This step is not trained.
3. Fit an sklearn `LogisticRegression(max_iter=1000)` with its default `C = 1` on those features. **This is the only training.** Its logit is h(x), interpreted as the log density ratio log p₊(x)/p₋(x).

**At generation:** a forward hook on `model.model.layers[L]` (the decoder layer's output) replaces the activation at position −1 by integrating

    dx/dt = ∇h(x) / ‖∇h(x)‖,   t ∈ [0, T],   Euler, 10 steps of size T/10

Position −1 means the prompt's last token during prefill, then every newly generated token (with the KV cache each step only sees the new token). Because the gradient is unit-normalised, T is the total distance moved.

**Details that matter for this project:**
- **Steering strength:** the yaml default is T = 1, but the paper uses T = 4–6 for Llama-3.1-8B (Table 5; Falcon-7B 20–23, Mistral-7B 3–4, Qwen2.5-7B 13–16), and the README example uses T = 5.
- **Layer:** the code's `layer_idx = 13` counts from 0; the paper calls this layer 14.
- **Gradient direction:** h is computed on the L2-normalised x, so the code applies (I − x̂x̂ᵀ)/‖x‖ and ∇h is always at right angles to x. ODESteer only rotates the activation; it never shrinks it.
- **The classifier is smooth.** In the toy (§9.1) it reached only 83% training accuracy on a cleanly separable ring, and it labelled points "positive" only after they had left the data. **Never use h to judge success**; use the model's output.
- **No stopping rule:** the barrier-function view is part of the paper's theory (it proves h increases along trajectories), not a termination condition.

## 4. Method

### 4.1 Offline: what gets fitted (the LLM is frozen)

All components are fitted on **the same prompt template and the same token positions that will be steered** (see the position-mismatch risk in §7).

| Component | How | Trained? |
|---|---|---|
| Concept activations | Run the LLM on concept prompts; keep the layer-L activation at the steered position | no |
| Subspace U (d × k), mean μ | PCA on the concept activations (k chosen from the spectrum; check which PCs carry lightness vs hue for colours) | PCA only |
| Local manifold model in z = Uᵀ(x − μ) | Stored projected points, used for kNN tangent estimates and a mean-shift retraction (bandwidth by cross-validation) | no, it just stores points |
| Target score h(x) | ODESteer's recipe, unchanged: polynomial sketch + logistic regression, positive vs. negative | **yes, one logistic regression** |

### 4.2 Generation: the only code change is the vector field

    ODESteer:            v(x) = ∇h(x) / ‖∇h(x)‖
    Ours (main):         z ← z + (T/n) · normalize( P_T(z) · Uᵀ∇h(x) )     # tangent step
                         z ← z + bw² · ∇_z log p(z)   (repeat r times)       # retraction (mean shift)
                         P_T(z) = projector onto the top-m local PCs of the kNN of z in the stored points
    Ablation (additive): v(x) = U · ( â + λ · ĝ ),  â = normalize(Uᵀ∇h),  ĝ = normalize(∇_z log p(z))

Integrate inside the same forward hook, at the same layer and position. Only the part of x inside the concept subspace changes; the rest of the activation is untouched. With U = I and the tangent projection and retraction disabled, it is ODESteer exactly.

**Why not the original additive form `normalize(U(Uᵀ∇h + λ∇log p))`:** (1) the two terms have unrelated scales (∇h carries a 1/‖x‖ factor, ∇log p scales with 1/bandwidth²), so a λ grid is meaningless unless each term is normalised first; (2) even normalised, a density-gradient term pulls toward nearby dense data from every direction, including along the manifold, and in the toy it either traps the flow near its start (λ ≥ 1) or is too weak to stop it crossing the centre (λ = 0.3). Keeping only the component at right angles to ∇h fails the same way (§9.1). It stays in the study as an ablation.

Implementation: subclass `BaseODESteer` and override `steer` (the retraction is not a vector field, so `vector_field` alone is not enough). `fit` additionally stores U, μ, the projected points and the bandwidth.

## 5. Testbeds

### 5.1 Validation check: weekdays (not a research result)

**Purpose:** confirm that the pipeline, baselines and metrics are correct by reproducing published numbers before moving to concepts where the answer is unknown. A success here is an engineering check, not a finding.

**What already exists for this exact task and model (Llama-3.1-8B):**

| Work | What it shows |
|---|---|
| [Engels et al. 2024](https://arxiv.org/abs/2405.14860) | Days of the week form a circle in activation space; intervening inside that circle changes weekday-arithmetic answers |
| [Wurgaft et al. 2026](https://arxiv.org/abs/2605.05115) | Weekdays are the main example. Spline steering through the 7 day-centres gives outputs 2.8× more natural (E_BC) than a straight line; activation-path vs output-path distance correlation r = 0.99 vs 0.89; reconstructing activation paths from desired outputs recovers the circle (R² 0.77 vs 0.42) |
| [Riemannian-Manifold Steering](https://arxiv.org/abs/2605.24942) | Shortest paths under an output-based distance on "a standard four-task arithmetic benchmark" with no per-prompt labels (task list not confirmed; weekdays is probably one); reliably reaches the target with natural paths |
| [causalab](https://github.com/goodfire-ai/causalab) `demos/weekdays_geometry` | Straight-line baseline: 49 prompts ("What day is four days after Sunday?"), model gets 45/49 right; answer day at the last token from layer 18 on (interchange accuracy 0.878 at L18, plateau 0.9184 from L25); a straight line between two answers passes through a third day on **14 of 22** pairs two days apart, **3 of 11** pairs three apart, **never** on the 10 adjacent pairs |

**What we run:** reproduce the straight-line numbers (A) and spline steering (F), then run B, C and E2 once. Readout is a single token over 7 days, so the metrics in §5.3 apply directly. Start/target pairs: adjacent, distance 2, distance 3.

**Pass criterion:** A and F land close to the published numbers (14/22 crossings, E_BC gap of roughly 2.8×). If they don't, fix the setup before anything else.

**What a success would add (small):** (1) the setup is trustworthy; (2) a modest methods point: E2 needs only a "target vs rest" classifier plus unlabelled activations, with no day-centres, ordering or trained encoder, so matching F under those weaker inputs is mildly interesting (it matches RMS on "no labels" and differs only on "no training"); (3) the first report of how ODESteer behaves on a curved concept. None of this is publishable on its own.

**Caveat specific to weekdays:** the manifold is **7 clusters with no data between them**, because every answer is one day. Splines *assume* a curve between the centres; E2 relies on data along the path. The nearest-neighbour tangent has nothing to fit in the gaps, and the retraction will snap the flow to the nearest day and hold it there (the same stalling seen in §9.1). So E2 may look worse here for a reason that does not apply to continuous concepts. Filling the gaps with prompts like "half a day after…" is possible, but it is unclear whether those states lie on the curve. Report E2's weekday result with this caveat and don't draw conclusions about the method from it.

### 5.2 Testbed 1: colours (first real test)

**Why colours:** hue is periodic. A path that cuts across the hue circle passes through desaturated colours; a path that stays on the surface goes around it. But see the caveats below; the geometry is richer than "a circle with grey in the middle".

- **Prompt and readout:** use causalab's `hex_color` task: `"Question: Which color name best describes the hex code {hex}? Choose one of: red, orange, yellow, green, blue, purple.\nAnswer:"`. All six answers are single tokens, the answer variable is set up as periodic hue (red 0°, orange 30°, yellow 58°, green 120°, blue 235°, purple 285°), and 600 bundled stimuli exist. Read the hue off the 6-way output distribution instead of parsing generated hex. (Indigo was dropped because Qwen3-4B could not separate it from purple.)
- **Manifold data:** [goodfire-ai/sae-manifold](https://github.com/goodfire-ai/sae-manifold) builds colours from [`burkelibbey/colors`](https://huggingface.co/datasets/burkelibbey/colors) with `"The hex code {hex} is for the color"`, Llama-3.1-8B layer 19, every 10th row, lightness and saturation both in [0.2, 0.8]. Note that `burkelibbey/colors` has **no clean colour names** (33,887 rows of `hex` + a generated description such as "Very dark blue: A nearly black shade…"), so a name → hex direction would need name extraction and would be many-to-one. Refit on the exact `hex_color` template and steered position.
- **Geometry check first:** sae-manifold describes colours as a **paraboloid** over hue, lightness and saturation, so a 3-component PCA will mix in lightness. Plot PC pairs coloured by hue, lightness and saturation, and project grey-type colours into the subspace to check they really sit at the centre before using saturation as a metric.
- **Target h:** positive = stimuli labelled red; negative = all other colours. Report the neighbour colours separately (the negative class includes orange and purple).
- **Starting points:** green (120° from red) is a *weak* test: a straight flow from there only dips to ≈ 0.87 R (toy: 0.88 R). Use cyan / blue-green starts (≈ 180°) for the strong "passes through grey" prediction, and treat them as the antipodal condition (§2).
- **Sweep:** T calibrated against ‖x‖ at the chosen layer (paper scale for Llama-3.1-8B is T ≈ 4–6), and for the ablation λ ∈ {0.1, 0.3, 1, 3} with both terms normalised.

**Why colours come first:** hue is continuous in the data, so E2's tangent and retraction steps have points to work with along the whole path, and there is ground truth (the hex code) for every stimulus.

### 5.3 Conditions, metrics and analysis (shared by all testbeds)

#### Conditions (ablations separate the changes)

| | Subspace restriction U | On-manifold mechanism |
|---|---|---|
| A. Linear steering (CAA-style mean difference) | – | – |
| B. ODESteer | full space | none |
| C. Projection only | U | none |
| D. Density only | full space | additive normalised density term |
| E1. Additive (original proposal) | U | additive normalised density term, λ sweep |
| E2. **Ours** | U | tangent projection + mean-shift retraction |
| F. Reference: spline manifold steering (Wurgaft et al.: cubic splines through class centroids in a 64-D PCA) | n/a | n/a |
| G. Reference: pullback-Hellinger geodesics (Riemannian-Manifold Steering), if code is available | n/a | n/a |

#### Metrics

| Metric | Measures |
|---|---|
| Target success: argmax output = target class (weekdays / 6 colours) | does it get there |
| Cumulative output energy E_BC = ∫ d_BC(γ(t), M_y) dt (Bhattacharyya distance to the behaviour manifold, Wurgaft et al.) | naturalness of the output path (**primary**) |
| Pass-through count: does the argmax visit a non-adjacent class along the path (weekdays); minimum saturation of the output hue (colours, after the geometry check) | does it cut across |
| Isometry: correlation of geodesic distances on M_h and M_y along the path | does activation movement match behaviour movement |
| Output trajectory (hue / day) vs. t | around or across |
| Perplexity of free continuations under the unsteered model | fluency |
| Off-target KL on unrelated prompts (same hook, same T) | side effects (compare with FishBack / dual steering numbers) |
| log p(z) along the path | on-manifold sanity check (E1/E2 optimise it by construction) |
| ‖Uᵀ∇h‖ / ‖∇h‖ per layer | how much of ODESteer's direction lies in the subspace; if small, C mostly just shortens the step |

**Comparisons:** primary = **matched target success**. Also report matched T (equal distance moved; this penalises curved paths, which spend their budget going around) and matched endpoint displacement ‖x_T − x_0‖.

**Statistics:** all prompts per start/target pair (49 weekday prompts; 100 stimuli per colour), ≥ 3 seeds for the random sketch and data splits, bootstrap 95% CIs, paired comparisons across conditions on the same prompts.

#### How to read the outcomes

| Result | Meaning |
|---|---|
| E2 stays on the manifold and reaches the target; B cuts across or breaks | The constraint makes the flow follow the concept's real shape. The mechanism works |
| Everything cuts across | The internal geometry doesn't drive the output at this layer/position, or the manifold fit is wrong; check against F |
| Everything goes around | ∇h already follows the curve; the manifold adds nothing for this concept |
| C ≈ E2 | The benefit comes from the subspace restriction, not the on-manifold mechanism |
| E2 ≈ F on E_BC | ODESteer-on-manifold matches supervised spline steering without needing class centroids |
| E2 stalls from antipodal starts | Expected (§2); report separately |
| E2 stalls or snaps to a day on weekdays but works on colours | Expected: weekdays have no data between the 7 days (§5.1); not evidence against the method |

## 6. Phases

0. **CPU sanity check (this machine).** Reproduce the toy (§9.1) as a unit test for the steering code. Then a small model: Gemma 3 270M / Qwen2.5-0.5B are probably too weak for the colour readout; the `hex_color` task was validated on Qwen3-4B-Instruct (≈ 16 GB in fp32 on CPU). Go/no-go: does a periodic structure appear at some layer, and does steering change the argmax at all?
1. **Weekdays validation check on Llama-3.1-8B (GPU, ≥ 40 GB), short.** Start from causalab's weekdays demo (it harvests at layer 25, last token; ODESteer's layer 14 is before the answer appears at L18). Reproduce A and F against the published numbers, then run B, C and E2 once (§5.1). *Contributes:* confidence in the setup only.
2. **Colours on Llama-3.1-8B** with the `hex_color` prompt: the first real test of the method. Pick layers by sweeping (sae-manifold uses 19, the manifold-steering paper 28). *Contributes:* evidence that E2 follows a continuous concept manifold where splines and geodesics also apply; a head-to-head with F (and G if available).
3. **Emotion (first result that could be a genuine contribution):** valence-arousal plane following [Sun et al. 2026](https://arxiv.org/abs/2604.03147) (PCA + ridge regression over ≈ 44.7k words; Llama-3.1-8B, Qwen3-8B, Qwen3-14B). Steer angry → calm. **Caveat:** Sun et al. report that VA steering changes refusal *through* the refusal/compliance tokens themselves ("lexical mediation"), so a refusal change is an expected effect, not a side effect; measure side effects on something else (e.g. unrelated-prompt KL, task accuracy). *Contributes:* here the target is defined by a classifier (angry vs calm) and there are no class centres or ordering to fit, so the spline and RMS methods don't apply directly; "ODESteer kept on the manifold gives fewer side effects at the same success rate" would be new.
4. **Unsupervised manifold:** replace PCA-on-labels with an SAE feature cluster from sae-manifold (Ising + Leiden). Candidate SAEs: `Goodfire/Llama-3.1-8B-Instruct-SAE-l19`, Gemma Scope 2. Optionally compare with BSF. *Contributes:* removes the need for concept labels to build U.
5. **ODESteer benchmarks:** TruthfulQA and RealToxicityPrompts. **Open:** these have no obvious concept manifold, so first define U (e.g. PCA of the positive + negative activations) and say what "on-manifold" means there. Toxicity evaluation needs a Perspective API key. *Contributes:* shows the constraint doesn't cost task performance on ODESteer's own benchmarks.
6. **Honesty (open question):** check whether honesty features form anything richer than a line. Seed with Neuronpedia "lying" and "deception" features on Gemma 3 plus Gemma Scope 2, and use Apollo or Geometry-of-Truth data. If honesty is linear, this method adds little there; that is itself a result.

## 7. Risks and open decisions

- **Choice of U is the fragile step.** If the target isn't inside U, the flow can't reach it (C/E fail while B succeeds). That failure is informative: it says whether the concept lives in the subspace.
- **Antipodal starts have no preferred direction** for any manifold-constrained gradient flow (§2, §9.1). Options if needed: a small tie-breaking tangent bias, or a two-point geodesic (F/G) instead of a gradient flow.
- **Local tangent estimates and KDE don't scale** to large k; then switch to a GMM, a learned score model or a learned chart (e.g. Riemannian-Manifold Steering's encoder).
- **Step-size scale** depends on activation norms per layer; calibrate T against ‖x‖.
- **Which positions to steer:** ODESteer steers position −1 at each step. Keep that for comparability, and fit the manifold at those positions.
- **Position mismatch:** the manifold and the steered position must use the same template and token; the original plan fitted on `"The color {name}"` and steered at `…hex code is #` and the generated hex tokens.
- **Readout noise:** use single-token multiple-choice readouts (weekdays, `hex_color`), not parsed hex.

## 8. Related work and novelty

| Work | What it does | Relation to this plan |
|---|---|---|
| [Wurgaft et al. 2026, Manifold Steering](https://arxiv.org/abs/2605.05115) ([html](https://arxiv.org/html/2605.05115), [blog](https://www.goodfire.com/research/manifold-steering)) | Cubic splines (thin-plate for 2-D) through class centroids in a 64-D PCA, Llama-3.1-8B layer 28, K = 50 points per path; behaviour manifold in Hellinger space; E_BC metric (manifold 2.8× lower than linear); isometry correlations; pullback optimisation recovers M_h. **Definition 1 includes a density-weighted metric (αe^{−E(h)}+β)⁻¹I with E ∝ −log p.** | Closest prior work; source of condition F and the primary metrics. The density idea is not new |
| [Riemannian-Manifold Steering, 2605.24942](https://arxiv.org/html/2605.24942v2) | Pulls output-space Hellinger distance back to activations with a learned encoder; geodesic steering without per-prompt labels; linear and spline steering as special cases | Direct competitor (condition G) |
| [Curveball Steering, 2603.09313](https://arxiv.org/pdf/2603.09313) | Polynomial kernel PCA steering; measures geodesic-to-Euclidean distortion per concept | Curved steering baseline; its distortion measure can pick concepts where curvature should matter |
| [FishBack, 2605.17231](https://arxiv.org/pdf/2605.17231) | Closed-form direction from the Fisher metric pulled back to the layer; 1.4–6.5× lower off-target KL | Side-effect baseline |
| [Park et al., Information Geometry of Softmax (ICML 2026)](https://arxiv.org/html/2602.15293v2) ([code](https://github.com/KihoPark/dual-steering)) | Dual steering, provably minimises off-target change | Side-effect baseline |
| [FLAS, 2605.05892](https://arxiv.org/pdf/2605.05892) | Learned concept-conditioned velocity field; curved, token-varying trajectories; beats prompting on AxBench | Learned-flow alternative |
| [UniSteer, 2605.30076](https://arxiv.org/html/2605.30076v1) | Text-conditioned flow matching over residual activations | Learned-flow alternative |
| [Steered LLM activations are non-surjective, 2604.09839](https://arxiv.org/abs/2604.09839) | Proves steering pushes activations off the set reachable from prompts | Motivation for on-manifold steering |
| [When is your LLM steerable?, 2606.11599](https://arxiv.org/abs/2606.11599) | Predicts steering efficacy from early hidden states | Possible analysis tool |
| Diffusion guidance: [MPGD](https://arxiv.org/abs/2311.16424), [DPS](https://arxiv.org/abs/2209.14687), [MCG](https://arxiv.org/abs/2206.00941) | Manifold-preserving / manifold-constrained guidance; MPGD projects guidance onto the data tangent space | E2 is the MPGD analogue |
| [Engels et al. 2024](https://arxiv.org/abs/2405.14860) | Circular features (days, months) | Origin of the weekday circle |

## 9. Design review notes (2026-09-19)

### 9.1 Toy simulation

Synthetic check of the mechanism, not evidence about any LLM. Setup: a hue ring of radius R = 3 in a 3-D subspace (plus a lightness axis) of a 256-D space, mean offset of norm 20, isotropic noise 0.3, 3000 points; positive = hue within ±30° of red. Classifier = ODESteer's own `NormedPolyClassifier` (8000 features, gamma 0.1, coef0 1); training accuracy 0.83. KDE bandwidth by 5-fold CV: 0.23. 64 starts per condition.

| Condition | From green (120°): lowest radius / R | From cyan (180°): lowest radius / R | Where the flow ends (T = 8) |
|---|---|---|---|
| B: ODESteer | 0.88 | 0.16 | red-ish, but off the data (log p = −103 from green) |
| C: projection only | 0.84 | 0.15 | same as B |
| D: density only | 0.98 | 0.93 | leaves the subspace; never reaches red |
| E1, λ = 0.3 | 0.93 | 0.23 | cuts across the centre |
| E1, λ = 1 or 3 | ≥ 0.97 | ≥ 0.84 | stuck partway (hue 35–118°) |
| E1 with density term orthogonal to ∇h | ≥ 0.97 | 0.37–0.86 | same trade-off: either crosses or stalls |
| **E2: tangent projection + retraction** (T = 10) | 0.97 | 0.94 | reaches red from green (hue ≈ 2°, on the ring); stalls from cyan (hue 153–165°) |

Also observed: the classifier labelled the on-ring red endpoint of E2 as "not red", and B's off-manifold endpoints as "red". Success must be judged from the model output, not from h.

Scripts (scratch, not committed): `toy/ring.py`, `toy/ring_perp.py`, `toy/ring_tangent.py`. Worth porting into the repo as a unit test in phase 0.

### 9.2 Summary of changes from the first draft

1. Main method changed from an additive density term to tangent projection + retraction; the additive form is kept as ablation E1 with both terms normalised.
2. Antipodal starts added as a separate condition with an explicit "no preferred direction" caveat.
3. Weekdays (causalab) added as a short validation check only (§5.1): the manifold-vs-straight-line result on weekdays is already published (Engels, Wurgaft, RMS), and weekdays have no data between the 7 days, which disadvantages E2. Colours are the first real test; emotion is the first phase that could give a new result. Each phase now states what it contributes.
4. Colour readout switched to causalab's single-token `hex_color` prompt; manifold fitted at the steered template and position; geometry check (paraboloid, where grey sits) before using saturation.
5. Primary metrics now Goodfire's E_BC and isometry; primary comparison at matched success; statistics specified.
6. §3 corrected: T scale (4–6 for Llama-3.1-8B), 0-based layer index, ∇h ⟂ x, smooth classifier, h not a success measure.
7. Emotion phase: refusal changes are the expected mechanism, not a side effect. Benchmark phase: U undefined, needs a Perspective API key.
8. Novelty narrowed; related-work table added (§8).

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
