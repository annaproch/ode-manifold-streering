# Draft experiment plan: ODE steering constrained to a concept manifold

*Status: draft, revised 2026-09-19 after a design review (§9). Nothing has been run on an LLM yet; only a synthetic CPU toy (§9.1, §9.3).*

## 1. Idea in one paragraph

ODESteer steers an LLM by moving a hidden activation along the gradient of a classifier ("more positive-like"). It knows *which way is better*, but nothing keeps the moving activation where real activations actually live, so it can cut through empty regions of activation space. Concept-geometry work (Goodfire's Neural Geometry series) shows that many concepts lie on curved, low-dimensional manifolds. **Proposal:** keep ODESteer's gradient flow, but constrain it to the concept manifold: (a) restrict it to the concept's subspace and (b) keep it on the data inside that subspace. This is the manifold-preserving guidance idea from diffusion models (MPGD, DPS, MCG), applied inside interpretable concept coordinates. With the constraint switched off, it reduces to ODESteer exactly.

**Scope of the novelty claim (see §8).** Density-weighted steering geometry already appears in Wurgaft et al. (2605.05115, Definition 1), and geodesic and kernel-based curved steering exist (Riemannian-Manifold Steering, Curveball). The defensible contribution is: *ODESteer's gradient flow constrained to a manifold, compared with spline and geodesic manifold steering on shared metrics.* "Classifier guidance in concept coordinates" on its own is not new.

## 2. Hypothesis

At matched target success, the manifold-constrained flow keeps its intermediate states on-concept (colours: it stays saturated instead of passing through grey or brown; behavioural concepts such as emotion: it moves through neighbouring states instead of cutting across), and it causes fewer off-target side effects and less fluency damage than ODESteer.

**Known limit (from §9.1 and §9.3):** when the start is directly opposite the target (e.g. cyan → red), the part of ∇h that points along the manifold is close to zero, so a manifold-constrained gradient flow has no preferred direction at first. In the toy this is a slow start, not a permanent stall: noise breaks the symmetry, and E2 reaches the target from 180° at about twice the T needed from 120°. Treat the antipodal case as its own condition, reported separately, and never use it as the main test in any phase.

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
| Local manifold model in z = Uᵀ(x − μ) | Stored projected points, used for kNN tangent estimates and the SCMS retraction (bandwidth by cross-validation) | no, it just stores points |
| Target score h(x) | ODESteer's recipe, unchanged: polynomial sketch + logistic regression, positive vs. negative | **yes, one logistic regression** |

### 4.2 Generation: what changes in the steering step

    ODESteer:            v(x) = ∇h(x) / ‖∇h(x)‖
    Ours (main, E2):     z ← z + (T/n) · normalize( P_T(z) · Uᵀ∇h(x) )     # tangent step, fixed length like ODESteer
                         z ← z + (I − P_T(z)) · bw² · ∇_z log p(z)   (r times) # retraction: subspace-constrained mean shift (SCMS)
                         P_T(z) = projector onto the top-m local PCs of the kNN of z in the stored points
    Ablation (E2-ms):    same, but plain mean shift  z ← z + bw² · ∇_z log p(z)  in the retraction
    Ablation (E2-h):     same as E2, but halve the step for a sample whenever its tangent direction reverses
    Ablation (additive): v(x) = U · ( â + λ · ĝ ),  â = normalize(Uᵀ∇h),  ĝ = normalize(∇_z log p(z))

Integrate inside the same forward hook, at the same layer and position. Only the part of x inside the concept subspace changes; the rest of the activation is untouched (except in variant E2-full, §7). With U = I and the tangent projection and retraction disabled, it is ODESteer exactly.

**Why not the original additive form `normalize(U(Uᵀ∇h + λ∇log p))`:** (1) the two terms have unrelated scales (∇h carries a 1/‖x‖ factor, ∇log p scales with 1/bandwidth²), so a λ grid is meaningless unless each term is normalised first; (2) even normalised, a density-gradient term pulls toward nearby dense data from every direction, including along the manifold, and in the toy it either traps the flow near its start (λ ≥ 1) or is too weak to stop it crossing the centre (λ = 0.3). Keeping only the component at right angles to ∇h fails the same way (§9.1). It stays in the study as an ablation.

**Why SCMS, not plain mean shift (review point, confirmed in §9.3):** repeated plain mean-shift steps climb to local density bumps along the ring, not to the nearest point on the curve, so with more retraction steps the flow is dragged back and stalls (toy: success 1.00 at r = 1–3, **0.00 at r = 10 and 30**). SCMS ([Ozertem & Erdogmus 2011](https://jmlr.org/papers/v12/ozertem11a.html)) keeps only the part of the mean-shift step at right angles to the local tangent, which matches E2's tangent/normal split; in the toy the main E2 (SCMS, fixed step) is not stalled by larger r: success 1.00 / 1.00 / 1.00 / 0.99 at r = 1 / 3 / 10 / 30 (120°, T = 20), and 0.84–0.95 from 180° at T = 20, 1.00 at T = 40. Report sensitivity to r ∈ {1, 3, 10, 30} on real data.

**Step length (review point, tested in §9.3):** with a fixed step length the flow hovers around the target within about one step once it gets there (toy: final swing ≈ 4°, 8°, 16° at T = 10, 20, 40). Two step-control rules were tested and neither is adopted as the default: *stop when the tangent direction reverses* fires far too early because the kNN tangent is noisy (success 0.5–0.95; rejected); *halve the step on reversal* (E2-h) removes the swing but spends the T budget (success ≈ 0.4 / 0.8 / 1.0 at T = 10 / 20 / 40 from 120°), so it stays an ablation. Keeping ODESteer's fixed step also keeps T = distance moved for both B and E2. The swing does not affect the primary comparison, which uses the smallest T at which each condition reaches the target (§5.3); report the swing at larger T as a secondary number. Apply any step-control rule to B as well, or to neither.

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

**What we run:** reproduce the straight-line numbers (A) and spline steering (F), then run B, C and E2 over the T grid (r ∈ {1, 10} for E2). Readout is a single token over 7 days, so the metrics in §5.3 apply directly. Start/target pairs: adjacent, distance 2, distance 3.

**Pass criterion:** A and F land close to the published numbers (14/22 crossings, E_BC gap of roughly 2.8×). If they don't, fix the setup before anything else.

**What a success would add (small):** (1) the setup is trustworthy; (2) a modest methods point: E2 needs only a "target vs rest" classifier plus unlabelled activations, with no day-centres, ordering or trained encoder, so matching F under those weaker inputs is mildly interesting (it matches RMS on "no labels" and differs only on "no training"); (3) the first report of how ODESteer behaves on a curved concept. None of this is publishable on its own.

**Caveat specific to weekdays:** the manifold is **7 clusters with no data between them**, because every answer is one day. Splines *assume* a curve between the centres; E2 relies on data along the path. The nearest-neighbour tangent has nothing to fit in the gaps, and the retraction will pull the flow toward the nearest day cluster (SCMS removes the along-tangent drag, but only if the tangent is estimated well, which it can't be in a gap). So E2 may look worse here for a reason that does not apply to continuous concepts. Filling the gaps with prompts like "half a day after…" is possible, but it is unclear whether those states lie on the curve. Report E2's weekday result with this caveat and don't draw conclusions about the method from it.

### 5.2 Testbed 1: colours (first real test)

**Why colours:** hue is periodic. A path that cuts across the hue circle passes through desaturated colours; a path that stays on the surface goes around it. But see the caveats below; the geometry is richer than "a circle with grey in the middle".

- **Prompt and readout:** use causalab's `hex_color` task: `"Question: Which color name best describes the hex code {hex}? Choose one of: red, orange, yellow, green, blue, purple.\nAnswer:"`. All six answers are single tokens, the answer variable is set up as periodic hue (red 0°, orange 30°, yellow 58°, green 120°, blue 235°, purple 285°), and 600 bundled stimuli exist. Read the hue off the output distribution instead of parsing generated hex. (Indigo was dropped because Qwen3-4B could not separate it from purple.)
- **Extended answer set, so "grey" is visible.** The 6 validated options contain no grey, so a path through the centre could only show up indirectly. Add `gray` and `brown` as options (8-way). This departs from the validated task, so first check on Llama-3.1-8B, before any steering: (i) accuracy on the 600 chromatic stimuli stays ≥ 0.9 of the 6-way accuracy; (ii) grey and brown stimuli (to be added from `burkelibbey/colors`, saturation < 0.15 for grey, dark orange hues for brown) are answered `gray`/`brown`; (iii) both new options are single tokens for the tokenizer (otherwise score the first token only and say so). If (i)–(ii) fail, fall back to the 6-way set and rely on the indirect signals below.
- **Output saturation, defined:** ρ(p) = |Σ_c p_c · e^{iθ_c}| over the chromatic options, with θ_c the hue centres above and p normalised over *all* options, so probability on gray/brown lowers ρ. ρ = 1 when all mass is on one colour. The hue centres are unevenly spaced, so even a spread-out distribution doesn't give exactly 0; report ρ relative to its value at the unsteered start, plus the probability on gray, on brown, and on colours not adjacent to the start or target, as separate numbers.
- **Manifold data:** [goodfire-ai/sae-manifold](https://github.com/goodfire-ai/sae-manifold) builds colours from [`burkelibbey/colors`](https://huggingface.co/datasets/burkelibbey/colors) with `"The hex code {hex} is for the color"`, Llama-3.1-8B layer 19, every 10th row, lightness and saturation both in [0.2, 0.8]. Note that `burkelibbey/colors` has **no clean colour names** (33,887 rows of `hex` + a generated description such as "Very dark blue: A nearly black shade…"), so a name → hex direction would need name extraction and would be many-to-one. Refit on the exact `hex_color` template and steered position.
- **Geometry check first:** sae-manifold describes colours as a **paraboloid** over hue, lightness and saturation, so a 3-component PCA will mix in lightness. Plot PC pairs coloured by hue, lightness and saturation, and project grey-type colours into the subspace to check they really sit at the centre before using saturation as a metric.
- **Target h:** positive = stimuli labelled red; negative = all other colours. Report the neighbour colours separately (the negative class includes orange and purple).
- **Starting points:** the main test uses starts **90°, 120° and 150°** from red; **150° is the key one**. Green (120°) alone is a weak test: a straight flow from there only dips to ≈ 0.87 R (toy: 0.87–0.88 R). At 150° ODESteer already cuts well inside the ring (toy: 0.52 R) while E2 stays on it (0.94 R) and still reaches red (success 0.98–1.00 at T ≥ 20). 180° (cyan) gives the most dramatic "through grey" contrast (toy: ODESteer 0.10 R) but is the antipodal condition (§2): run it, report it separately, and don't use it as the main test. The bundled 600 stimuli are jittered around the six class centres, so check whether 150° and 180° hues are covered; if not, generate extra hex stimuli at those hues (and check the model labels them consistently).
- **Sweep:** T calibrated against ‖x‖ at the chosen layer (paper scale for Llama-3.1-8B is T ≈ 4–6); for E2, r ∈ {1, 3, 10, 30}, neighbour count and tangent dimension m; k ∈ {3, 8, 16, 64} (§7); for the E1 ablation, λ ∈ {0.1, 0.3, 1, 3} with both terms normalised. Choose E2's r, neighbour count and m on a held-out split of stimuli, not on the test prompts.

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
| E2. **Ours** | U | tangent projection + SCMS retraction, fixed step |
| E2-ms / E2-h (ablations) | U | plain mean-shift retraction / step halving on reversal |
| E2-full (only if the residual probe fails, §7) | U + residual | E2, plus the residual moved by a kernel regression of x on z |
| F. Reference: spline manifold steering (Wurgaft et al.: cubic splines through class centroids in a 64-D PCA) | n/a | n/a |
| G. Reference: pullback-Hellinger geodesics (Riemannian-Manifold Steering), if code is available | n/a | n/a |

#### Metrics

| Metric | Measures |
|---|---|
| Target success: argmax output = target class (weekdays / colours); for emotion see §6 phase 3 | does it get there |
| Cumulative output energy E_BC = ∫ d_BC(γ(t), M_y) dt (Bhattacharyya distance to the behaviour manifold, Wurgaft et al.) | naturalness of the output path (**primary for weekdays and colours only**: it needs class centres to build M_y; primary metrics for the other phases are fixed in §6) |
| Pass-through count: does the argmax visit a non-adjacent class along the path (weekdays); minimum ρ along the path and peak probability on gray / brown / non-adjacent colours (colours, §5.2) | does it cut across |
| Isometry: correlation of geodesic distances on M_h and M_y along the path | does activation movement match behaviour movement |
| Output trajectory (hue / day) vs. t | around or across |
| Perplexity of free continuations under the unsteered model | fluency |
| Off-target KL on unrelated prompts (same hook, same T) | side effects (compare with FishBack / dual steering numbers) |
| log p(z) along the path | on-manifold sanity check (E1/E2 optimise it by construction) |
| Final swing: range of the output hue/day over the last steps at T above the matched-success T | hovering around the target with fixed-length steps (§4.2) |
| ‖Uᵀ∇h‖ / ‖∇h‖ per layer | how much of ODESteer's direction lies in the subspace; if small, C mostly just shortens the step |
| Residual hue probe: a linear probe for hue (cos θ, sin θ) trained on the residual (I − UUᵀ)x of unsteered activations, applied after steering | whether the unmoved part of the activation still encodes the *starting* concept (§7, residual conflict) |

**Comparisons:** primary = **matched target success**: for each condition, the smallest T at which success reaches a pre-set level (e.g. 80% of prompts), and all path metrics measured at that T. Also report matched T (equal distance moved; this penalises curved paths, which spend their budget going around) and matched endpoint displacement ‖x_T − x_0‖.

**Statistics:** all prompts per start/target pair (49 weekday prompts; 100 stimuli per colour, plus any generated at 150°/180°), ≥ 3 seeds for the random sketch and data splits, bootstrap 95% CIs, paired comparisons across conditions on the same prompts.

#### How to read the outcomes

| Result | Meaning |
|---|---|
| E2 stays on the manifold and reaches the target; B cuts across or breaks | The constraint makes the flow follow the concept's real shape. The mechanism works |
| Everything cuts across | The internal geometry doesn't drive the output at this layer/position, or the manifold fit is wrong; check against F |
| Everything goes around | ∇h already follows the curve; the manifold adds nothing for this concept |
| C ≈ E2 | The benefit comes from the subspace restriction, not the on-manifold mechanism |
| E2 ≈ F on E_BC | ODESteer-on-manifold matches supervised spline steering without needing class centroids |
| E2 needs a much larger T, or fails, from antipodal starts | Expected (§2); report separately |
| E2-ms stalls as r grows but E2 (SCMS) doesn't | Confirms the retraction choice (§4.2) |
| Residual probe still decodes the start after E2 | The concept is not contained in U at this k; increase k or use E2-full before interpreting E2 (§7) |
| E2 stalls or snaps to a day on weekdays but works on colours | Expected: weekdays have no data between the 7 days (§5.1); not evidence against the method |

## 6. Phases

0. **CPU sanity check (this machine).** Port the toy and its sweep (§9.1, §9.3) into the repo as a test for the steering code: it must reproduce the r-sensitivity of E2-ms vs E2 and the antipodal slow start. Then a small model: Gemma 3 270M / Qwen2.5-0.5B are probably too weak for the colour readout; the `hex_color` task was validated on Qwen3-4B-Instruct (≈ 16 GB in fp32 on CPU). Go/no-go: does a periodic structure appear at some layer, and does steering change the argmax at all?
1. **Weekdays validation check on Llama-3.1-8B (GPU, ≥ 40 GB), short.** Start from causalab's weekdays demo (it harvests at layer 25, last token; ODESteer's layer 14 is before the answer appears at L18). Reproduce A and F against the published numbers, then run B, C and E2 over the T grid, with r ∈ {1, 10} for E2 (§5.1). *Contributes:* confidence in the setup only.
2. **Colours on Llama-3.1-8B** with the `hex_color` prompt: the first real test of the method. Pick layers by sweeping (sae-manifold uses 19, the manifold-steering paper 28). *Contributes:* evidence that E2 follows a continuous concept manifold where splines and geodesics also apply; a head-to-head with F (and G if available).
3. **Emotion (first result that could be a genuine contribution):** valence-arousal plane following [Sun et al. 2026](https://arxiv.org/abs/2604.03147) (PCA + ridge regression over ≈ 44.7k words; Llama-3.1-8B, Qwen3-8B, Qwen3-14B). **Emotion pairs are chosen from measured angles, not labels.** First fit the VA plane and measure each emotion's angle in *this model*. Main pairs: those ≤ 120° apart (e.g. angry → sad, excited → calm, if the measured angles agree with the usual circumplex). Near-opposite pairs (angry → calm, and also anxious → calm, which sits close to opposite on the standard circumplex) are run as the separately reported antipodal condition (§2), not as the main test. **Primary metrics, fixed now (E_BC does not apply here: there are no class centres to build M_y):** (a) success = the generated continuation's valence-arousal angle within ±30° of the target emotion's angle, with VA scored from the text by the same human word-rating lexicon Sun et al. compare against (confirm which one); (b) path naturalness = minimum VA radius of the output along the path, relative to the unsteered radius (the analogue of ρ for colours); (c) side effects = off-target KL on unrelated prompts and accuracy on an unrelated task. These are text-scored, so they are not directly comparable with the colour numbers. **Caveat:** Sun et al. report that VA steering changes refusal *through* the refusal/compliance tokens themselves ("lexical mediation"), so a refusal change is an expected effect, not a side effect; measure side effects on something else (e.g. unrelated-prompt KL, task accuracy). *Contributes:* here the target is defined by a classifier (e.g. angry vs sad) and there are no class centres or ordering to fit, so the spline and RMS methods don't apply directly; "ODESteer kept on the manifold gives fewer side effects at the same success rate" would be new.
4. **Unsupervised manifold:** replace PCA-on-labels with an SAE feature cluster from sae-manifold (Ising + Leiden). Candidate SAEs: `Goodfire/Llama-3.1-8B-Instruct-SAE-l19`, Gemma Scope 2. Optionally compare with BSF. *Contributes:* removes the need for concept labels to build U.
5. **ODESteer benchmarks:** TruthfulQA and RealToxicityPrompts. **Open:** these have no obvious concept manifold, so first define U (e.g. PCA of the positive + negative activations) and say what "on-manifold" means there. Toxicity evaluation needs a Perspective API key. *Contributes:* shows the constraint doesn't cost task performance on ODESteer's own benchmarks.
6. **Honesty (open question):** check whether honesty features form anything richer than a line. Seed with Neuronpedia "lying" and "deception" features on Gemma 3 plus Gemma Scope 2, and use Apollo or Geometry-of-Truth data. If honesty is linear, this method adds little there; that is itself a result.

## 7. Risks and open decisions

- **Choice of U is the fragile step.** If the target isn't inside U, the flow can't reach it (C/E fail while B succeeds). That failure is informative: it says whether the concept lives in the subspace.
- **Antipodal starts have no preferred direction at first** for any manifold-constrained gradient flow (§2, §9.3): in the toy, a slow start needing about 2× T, not a stall; on real data with a noisier tangent it may be worse. This applies to every testbed, including the colour starts (§5.2) and the emotion pairs (§6 phase 3). Options if needed: a small tie-breaking tangent bias, or a two-point geodesic (F/G) instead of a gradient flow.
- **Residual conflict.** Only the k-dimensional part of x moves. If the concept is also encoded outside U (likely with k ≈ 3; Wurgaft et al. use a 64-D PCA), the rest of the activation still says "green" and the model gets conflicting signals. Check with the residual hue probe (§5.3) and run k ∈ {3, 8, 16, 64}. Tension: the local-tangent and SCMS steps rely on nearest neighbours, which get unreliable as k grows with only hundreds of stimuli. If the probe still decodes the start at the k where neighbours still work, add variant E2-full: move the residual too, by the change in a kernel regression of x on z (x ← x + m̂(z_new) − m̂(z_old)), which is close in spirit to spline steering.
- **Tangent noise.** The kNN tangent is noisy even in the toy (3000 points); with 600 colour stimuli it will be noisier. Sweep the neighbour count and the tangent dimension m, and report them.
- **Local tangent estimates and KDE don't scale** to large k; then switch to a GMM, a learned score model or a learned chart (e.g. Riemannian-Manifold Steering's encoder).
- **Step-size scale** depends on activation norms per layer; calibrate T against ‖x‖.
- **Which positions to steer:** ODESteer steers position −1 at each step. Keep that for comparability, and fit the manifold at those positions.
- **Position mismatch:** the manifold and the steered position must use the same template and token; the original plan fitted on `"The color {name}"` and steered at `…hex code is #` and the generated hex tokens.
- **Readout noise:** use single-token multiple-choice readouts (weekdays, `hex_color`), not parsed hex.
- **Readout can hide the effect:** without a gray option the colour readout cannot show a path through grey (§5.2). Validate the extended 8-way set before using it.

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
| E2 first version: tangent projection + plain mean shift, r = 3 (T = 10) | 0.97 | 0.94 | reaches red from green (hue ≈ 2°, on the ring); from cyan still at hue 153–165° at T = 10. **Superseded by §9.3:** this was a single setting; the sweep shows it stalls for r ≥ 10 and reaches red from cyan at larger T |

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
9. Second review (§9.3): retraction switched to SCMS with an r sweep; fixed-length steps kept, step-control rules tested (stop rejected, halving kept as ablation); antipodal limit corrected to "slow start"; residual-conflict check (probe, k sweep, E2-full); 8-way colour readout with gray/brown and a defined saturation ρ; emotion pairs chosen from measured angles with pre-registered text-scored metrics; E_BC restricted to weekdays and colours.
10. Consistency pass after point 9: colour main test moved from the antipodal 180° start to 90°/120°/150° (150° key; 180° reported separately), since §2 forbids antipodal starts as the main test; start-hue coverage check for the stimuli; E2 hyperparameters chosen on held-out stimuli; weekday validation runs a T and r grid, not a single setting; §4.1/§4.2 wording aligned with SCMS and the overridden `steer`.

### 9.3 Second review and sweep of the final method

An outside review raised six points; all six were accepted. Points 2 and 3 were tested in the toy before being written into the method, with a sweep this time rather than a single setting: start ∈ {60, 90, 120, 150, 180}°, T ∈ {10, 20, 40}, r ∈ {1, 3, 10, 30}, 3 seeds (8 for the 120° and 180° rows of E2-h), 32 starts per run.

Success (final hue within ±30° of red) / lowest radius ÷ R, r = 3, mean over seeds:

| Condition | from 120°, T = 10 / 20 / 40 | from 180°, T = 10 / 20 / 40 |
|---|---|---|
| B: ODESteer | 1.00 / 1.00 / 1.00 · radius 0.87 | 1.00 / 1.00 / 1.00 · radius **0.10–0.15** (through the centre) |
| E2-ms: plain mean shift | 0.93 / 0.99 / 1.00 · radius 0.95 | 0.11 / 0.98 / 1.00 · radius 0.90 |
| E2: SCMS | 0.71 / 1.00 / 1.00 · radius 0.94 | 0.32 / 0.82 / 1.00 · radius 0.89 |
| E2 + stop on reversal (rejected) | 0.52 / 0.76 / 0.88 | 0.25 / 0.29 / 0.62 |
| E2-h: halve on reversal | 0.40 / 0.81 / 0.99 | 0.23 / 0.39 / 0.78 |

- **Retraction (point 2):** at 120°, T = 20, success at r = 1 / 3 / 10 / 30: E2-ms 1.00 / 0.99 / **0.00 / 0.00** (final hue 33° and 93°: stalled); E2 (SCMS, fixed step) 1.00 / 1.00 / 1.00 / 0.99, lowest radius 0.94–0.96. From 180°, T = 20: E2 0.84 / 0.82 / 0.93 / 0.95; at T = 40 all ≥ 0.99. With step halving (E2-h): 0.96 / 0.83 / 0.99 / 0.99.
- **Step length (point 3):** with fixed steps, the final swing is about one step length (≈ 4° / 8° / 16° at T = 10 / 20 / 40), not a large overshoot. B barely swings (0.2–2°) because it converges to the classifier's off-manifold peak.
- **Antipodal starts:** E2 needs about 2× the T of a 120° start but reaches red while staying on the ring (radius ≈ 0.9); ODESteer passes through the centre (radius 0.10). §2 was corrected from "stalls" to "slow start".
- Not testable in the toy: point 4 (residual conflict; the toy's hue lives only in U by construction), point 5 (readout) and point 6 (emotion metric). These are handled by checks on real data (§5.2, §5.3, §6, §7).

Scripts: `toy/ring_review.py`, `toy/ring_sweep.py`, `toy/ring_halve.py`, `toy/ring_scms_r.py` (results in `sweep.json`, `sweep_halve.json`, `sweep_scms_r.json`).

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
