# Draft experiment plan: ODE steering constrained to a concept manifold

*Status: draft, 2026-09-19. Nothing has been run yet.*

## 1. Idea in one paragraph

ODESteer steers an LLM by moving a hidden activation along the gradient of a classifier ("more positive-like"). It knows *which way is better*, but nothing keeps the moving activation where real activations actually live, so it can cut through empty regions of activation space. Concept-geometry work (Goodfire's Neural Geometry series) shows that many concepts lie on curved, low-dimensional manifolds. **Proposal:** keep ODESteer's gradient flow, but (a) restrict it to the concept's subspace and (b) add a pull toward the data density inside that subspace. This is classifier guidance from diffusion models, applied inside interpretable concept coordinates. With λ = 0 and the subspace set to the full space, it reduces to ODESteer exactly.

## 2. Hypothesis

At the same steering strength, the manifold-constrained flow reaches the target about as often as ODESteer, but its intermediate states stay on-concept (for colours: they stay saturated instead of passing through grey or brown), and it causes fewer off-target side effects and less fluency damage.

## 3. Background: what ODESteer actually does (verified in its code)

Source: [ZhaoHongjue/odesteer](https://github.com/ZhaoHongjue/odesteer), `src/odesteer/steer/_ode_steer.py`, `utils/kernels/`.

**Offline (the LLM is frozen):**
1. Run the LLM on positive and negative texts. Keep the layer-L activation at the last token → `pos_X`, `neg_X` (N × d).
2. L2-normalise each activation, then map it through a **fixed random** degree-2 polynomial sketch with 8000 features. This step is not trained.
3. Fit an sklearn `LogisticRegression` on those features. **This is the only training.** Its logit is h(x), interpreted as the log density ratio log p₊(x)/p₋(x).

**At generation (every new token):** a forward hook on layer L takes the activation x at the last position and integrates

    dx/dt = ∇h(x) / ‖∇h(x)‖,   t ∈ [0, T],   10 Euler steps (defaults: T = 1)

then writes x back. Because the gradient is unit-normalised, T is the total distance moved (the steering strength). There is no stopping rule; the barrier-function view is part of the paper's theory, not an actual stopping condition.

## 4. Method

### 4.1 Offline: what gets fitted (the LLM is frozen)

| Component | How | Trained? |
|---|---|---|
| Concept activations | Run the LLM on concept prompts; keep the layer-L last-token activation | no |
| Subspace U (d × k), mean μ | PCA on the concept activations (k ≈ 3 for colours) | PCA only |
| Density p(z) on z = Uᵀ(x − μ) | KDE over the projected points (bandwidth by cross-validation) | no, it just stores points |
| Target score h(x) | ODESteer's recipe, unchanged: polynomial sketch + logistic regression, positive vs. negative | **yes, one logistic regression** |

### 4.2 Generation: the only code change is the vector field

    ODESteer:  v(x) = ∇h(x) / ‖∇h(x)‖
    Ours:      v(x) = normalize( U · ( Uᵀ∇h(x) + λ · ∇_z log p(z) ) ),   z = Uᵀ(x − μ)

Integrate the same way (Euler, 10 steps, time T), inside the same forward hook, at the same layer and position. Only the part of x inside the concept subspace changes; the rest of the activation is untouched.

Implementation: subclass `BaseODESteer` and override `vector_field`. `fit` additionally stores U, μ and the KDE points.

## 5. Testbed 1: colours (known ground-truth geometry)

**Why colours:** hue is a circle whose centre is grey. A straight or unconstrained path from green to red passes near the centre, i.e. through desaturated colours. A path that stays on the surface should go around the circle, via yellow and orange (or via blue and purple), and stay saturated. That makes a visible, measurable prediction.

- **Data:** `burkelibbey/colors` (name → RGB), as shipped with [goodfire-ai/sae-manifold](https://github.com/goodfire-ai/sae-manifold). Prompt template, e.g. `"The color {name}"`. Try 2–3 templates.
- **Manifold:** PCA of colour activations. First check that hue shows up as a circle (plot PC1–PC2 coloured by the true hue).
- **Target h:** positive = colours with hue within ±30° of red; negative = all other colours.
- **Steering task:** start from green prompts. Sweep T ∈ {0, 0.25, …, 4} (to be rescaled to activation norms at the chosen layer) and λ ∈ {0, 0.1, 0.3, 1, 3}.
- **Readout:** at each T, have the model continue a template such as `"… Its hex code is #"`. Parse the hex (fall back to a name lookup in the dataset) → HSL.

### Conditions (ablations separate the two changes)

| | Subspace restriction U | Density term λ |
|---|---|---|
| A. Linear steering (CAA-style mean difference) | – | – |
| B. ODESteer | full space | 0 |
| C. Projection only | U | 0 |
| D. Density only | full space | > 0 (KDE in the PCA coordinates, gradient mapped back) |
| E. **Ours** | U | > 0 |
| F. Reference: supervised manifold steering (move along a fitted hue curve, Goodfire-style) | n/a | n/a |

### Metrics

| Metric | Measures |
|---|---|
| Target success: output hue within ±30° of red | does it get there |
| **Minimum saturation along the path** (the key metric) | does it cut through grey |
| Hue trajectory vs. T | around the circle or across it |
| Parse rate / coherent-output rate | breakage |
| Perplexity of the continuation under the unsteered model | fluency |
| Off-target KL on unrelated prompts (same hook, same T) | side effects |
| log p(z) along the path | stays on-manifold (a sanity check; E optimises it by construction) |

Compare conditions at **matched T** (equal distance moved) and at **matched target success**.

### How to read the outcomes

| Result | Meaning |
|---|---|
| E stays saturated and goes around the circle; B passes through grey or breaks | The density term makes the flow follow the concept's real shape. The mechanism works |
| Everything passes through grey | The internal geometry doesn't drive the output, or λ is too weak; check against F |
| Everything goes around the circle | ∇h already follows the curve; the manifold adds nothing for this concept |
| C ≈ E | The benefit comes from the subspace restriction, not the density term |

## 6. Phases

0. **CPU sanity check (on this machine):** small model (Qwen2.5-0.5B or Gemma 3 270M). Does a hue circle appear at any layer? Does steering change the named colour at all? Go/no-go for using a small model.
1. **Colours at full scale:** Llama 3.1 8B on a GPU (the default for both the ODESteer and sae-manifold code). Pick layers by sweeping; ODESteer used layer 13 for TruthfulQA.
2. **Emotion (a curved concept relevant to behaviour):** valence-arousal plane following [Sun et al. 2026](https://arxiv.org/abs/2604.03147). Steer angry → calm and measure the side effects on refusal and compliance tokens.
3. **Unsupervised manifold:** replace PCA-on-labels with an SAE feature cluster from sae-manifold (Ising + Leiden). Candidate SAEs: `Goodfire/Llama-3.1-8B-Instruct-SAE-l19`, Gemma Scope 2. Optionally compare with BSF.
4. **ODESteer benchmarks:** TruthfulQA and RealToxicityPrompts, to check that the constraint doesn't cost task performance compared with their reported gains.
5. **Honesty (open question):** check whether honesty features form anything richer than a line. Seed with Neuronpedia "lying" and "deception" features on Gemma 3 plus Gemma Scope 2, and use Apollo or Geometry-of-Truth data. If honesty is linear, this method adds little there; that is itself a result.

## 7. Risks and open decisions

- **Choice of U is the fragile step.** If the target isn't inside U, the flow can't reach it (condition C/E fails while B succeeds). That failure is informative: it says whether the concept lives in the subspace.
- **The KDE gradient in k = 3 is fine, but it won't scale** to large k; then switch to a GMM or a learned score model.
- **The step-size scale** depends on activation norms per layer; calibrate T against ‖x‖.
- **Which positions to steer:** ODESteer steers only the last position at each generation step. Keep that for comparability.
- **Readout noise:** hex parsing from a small model may be unreliable. A backup readout is the logits over a fixed list of colour names.
- **Prior work to check before claiming novelty:** manifold-constrained and classifier guidance in diffusion models (DPS, MCG), and any LLM steering that uses a density or score term.

## 8. References and code

- ODESteer: [paper](https://arxiv.org/abs/2602.17560) · [code](https://github.com/ZhaoHongjue/odesteer)
- Goodfire Neural Geometry series: [overview](https://www.goodfire.com/research/neural-geometry)
  - Manifold steering: [paper 2605.05115](https://arxiv.org/abs/2605.05115)
  - SAEs and manifolds: [paper 2604.28119](https://arxiv.org/abs/2604.28119) · [code](https://github.com/goodfire-ai/sae-manifold)
  - Block-sparse featurizers: [paper 2606.25234](https://arxiv.org/abs/2606.25234) · [code](https://github.com/goodfire-ai/block-sparse-featurizer)
  - Arithmetic in the Wild: [paper 2605.01148](https://arxiv.org/abs/2605.01148) · [code](https://github.com/goodfire-ai/arithmetic-wild)
  - Causal-abstraction toolkit (weekday geometry demo, steering tutorial): [causalab](https://github.com/goodfire-ai/causalab)
- Valence-arousal subspace: [Sun et al. 2026](https://arxiv.org/abs/2604.03147)
- Circular features: [Engels et al. 2024](https://arxiv.org/abs/2405.14860)
