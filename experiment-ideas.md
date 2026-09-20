# Where ODE + manifold could actually be useful: candidate experiments

*Written 2026-09-20, after the colour pilot. Ranked brainstorm of settings where the idea has a chance to earn its keep, given what that pilot established.*

Taking seriously what we learned: the constraint fails whenever the goal is "maximum effect" (overshoot beats the data) and whenever the readout can't express damage. So the useful settings are the ones where the target is a *specific place*, not an extreme, and where the model actually has to generate from the intermediate states.

Here are the directions I'd consider, ranked.

---

## 1. Trajectory control across generated tokens

**The setting.** Steer a property along a *planned path over the course of a generation* — an emotional arc that starts tense and resolves calm, a register that shifts from technical to plain, an argument that moves from uncertain to committed. The target moves as generation proceeds.

**Why the ODE matters here and didn't for colours.** In the colour pilot the integration happened inside one hook and the model only ever saw the endpoint — which is why path quality was unmeasurable. Here the path is realised *over tokens*: the model generates from each intermediate state and conditions on what it produced. The vector field is genuinely state-dependent, because the state genuinely moves. This is the first setting where "ODE" means something beyond "a straight step in ten pieces".

**Why the manifold matters.** Intermediate states have to be states the model can actually write from. A jump from terror to serenity in one sentence is incoherent; the manifold is what defines the plausible route between them. And off-manifold now has an obvious cost — degenerate text — which is exactly the damage signal colours lacked.

**Novel against the prior work.** Wurgaft interpolates between two endpoints at a single site; RMS computes geodesics for a fixed output-token schema; GeoSteer constrains the sphere. None of them steer along a manifold across generated tokens. Goodfire's "Meandering on Manifolds" observes that stories trace emotional trajectories — nobody has tried to *control* that trajectory.

**Decisive measurement.** Does the generated text follow the intended arc (scored by an affect model per sentence), at what cost in perplexity and coherence, versus a fixed vector whose strength is scheduled by hand?

**Kill condition.** If a hand-scheduled linear vector tracks the arc just as well, the ODE adds nothing and this closes fast.

---

## 2. Minimum-dose safety steering

**The setting.** Invert the objective. Instead of "maximise refusal", ask for **the smallest intervention that reaches the safety target**, and measure what it costs everywhere else: general capability, perplexity on benign prompts, behaviour on unrelated tasks.

**Why this fixes our problem.** Our finding was that overshoot beats the data, so a constraint that caps you at natural levels is a handicap. Under a minimum-dose objective that cap becomes the *point*. Staying on the manifold is no longer a tax on effect; it's the definition of success.

**Why ODE + manifold.** The manifold gives the coordinate system in which "minimal" is meaningful — minimal geodesic movement inside the concept, rather than minimal Euclidean norm in 4096 dimensions, which is the wrong metric. The ODE lets you stop the moment the behavioural target is met rather than applying a fixed strength.

**Novel.** Every method in the related-work table competes on effect at fixed strength. Nobody optimises for *dose*. It's also the framing practitioners actually want: "detoxify without making the model worse at everything else".

**Decisive measurement.** Refusal or detox rate held fixed; compare capability retention (a standard benchmark) and benign-prompt perplexity across arms. ODESteer already benchmarks on RealToxicityPrompts, so there's a baseline.

**Kill condition.** If plain projection retains capability just as well, you're back to the C-versus-E2 result and the manifold adds nothing again.

---

## 3. Which white-box vulnerabilities are prompt-reachable?

**The setting.** We found that steering reaches states *no prompt produces* — p(red) 0.797 at three times the data radius, where real red activations give 0.664. Generalise that: when a white-box attack drives a model into harmful compliance, is that state on the manifold of naturally-reachable activations, or an artefact?

**Why this matters.** It's a real methodological problem for the field. Steering-discovered vulnerabilities are routinely reported as findings, and the non-surjectivity result says steered states are generally unreachable from prompts. The manifold gives a quantitative filter: distance-to-data, with prompt-induced activations as the reference distribution.

**Why ODE + manifold.** An on-manifold steering flow finds the harmful states that are *plausible*; the unconstrained flow finds all of them. The difference between the two sets is the measurement. This is the one use where our constraint being a ceiling is the useful property.

**Decisive measurement.** For a set of steering-induced failures, report distance-to-manifold and whether a prompt-space search can reproduce the behaviour. Ceiling comparison: what's the strongest harmful behaviour achievable *without* leaving the data?

**Framing note.** Keep this defensive — the deliverable is a filter that tells you which white-box findings are worth acting on, not a recipe for reaching extreme states.

---

## 4. Multi-attribute composition

**The setting.** Steer two properties at once — less toxic *and* more formal, more concise *and* more accurate. Adding two steering vectors is the standard approach and the standard failure: they interfere and fluency collapses.

**Why ODE + manifold.** Composition is precisely where a linear step leaves the data: the sum of two valid displacements is usually not a valid state. A flow constrained to the joint manifold has to find a route that satisfies both, and there may not be a straight one. The ODE handles the fact that the right direction for attribute A changes as you move in attribute B.

**Novel.** Multi-attribute control is an active problem with its own papers; none of them treat it geometrically.

**Kill condition.** If the two subspaces turn out to be near-orthogonal, addition just works and there's nothing to fix. Cheap to check first — measure the subspace overlap before running anything.

---

## 5. The cheap demonstration: concepts whose direction rotates

**The setting.** Days, months, years, arithmetic — concepts already shown to be circular or helical. To move from 1850 to 1990 along a helix, the required direction rotates through more than a full turn. No fixed vector can do it; a straight line cuts through the middle of the helix.

**Why bother.** It's the cleanest existence proof that ODE + manifold is ever *necessary*, and our own toy already showed the mechanism (the horseshoe, where greedy ascent stalls and the graph score succeeds). Cheap, and it doesn't rely on off-manifold being costly — it relies on the target being unreachable in a straight line.

**Limitation.** It's a demonstration, not an application, and it's closest to Wurgaft's existing territory. Worth one figure, not a project.

---

## Considered and set aside

- **More colour work** — the readout can't express damage; settled.
- **Single-site endpoint quality on any concept** — same structural problem: at matched success, overshoot is free.
- **Unlearning** — mostly a weight-space problem.
- **Reasoning-mode steering** — appealing, but "the manifold of reasoning states" isn't definable enough yet to measure.

---

## What I'd actually do

**Pair 1 and 2.** They share infrastructure — per-token steering during generation, a fluency/capability readout, the support gate — and between them they cover both halves of the idea: the ODE earns its keep in 1 (the target moves), the manifold earns its keep in 2 (minimality is the objective). Both have a clear kill condition, and both are evaluated on things people already care about.

**Order:** the 30-minute generation-fragility test first, on states we already have. It answers the one question both directions depend on — whether off-manifold has a cost this model can express at all. If it doesn't even in free generation, then 1 and 2 both lose their foundation and idea 3 becomes the honest project, because "off-manifold is behaviourally free" would itself be the finding worth publishing.

---

## Additional brainstorm: context-dependent targets with multiple valid outcomes

These candidates exploit the distinction between steering toward a **learned target region** and specifying one endpoint. The desired behaviour depends on the starting context; local geometry could help preserve that context while the ODE searches for an acceptable outcome. They do not require a known concept ordering or a single global manifold.

### 6. Evidence-sensitive answering: answer, retrieve, clarify, or abstain

**The setting.** Give the model questions with complete, missing, contradictory, or irrelevant evidence. Sometimes it should answer; sometimes request clarification, retrieve information, or abstain. Start with generated documents about fictional entities so evidence sufficiency and factual correctness are mechanically checkable.

**Why ODE + manifold.** A classifier objective rewards evidence-appropriate behaviour rather than generic uncertainty. Repeated updates could adapt the direction as the decision changes; local geometry could retain the question, entities, and available facts. The target is a family of appropriate responses, not a common "I don't know" activation.

**Decisive measurement.** Fewer unsupported answers at matched answer coverage and retrieval budget, with accuracy preserved on answerable questions. Hold out document templates and domains. Include cases where adding one fact should change abstention into a correct answer.

**Kill condition.** If an uncertainty threshold, prompting, or a single corrected activation step gives the same tradeoff, the combined method has not earned its complexity.

### 7. Tool safety while preserving task completion

**The setting.** A simulated agent edits records or files and calls APIs under explicit permissions. Mix authorised requests, requests requiring confirmation, and untrusted instructions embedded in retrieved documents. All actions run in a disposable simulator.

**Why ODE + manifold.** Safe outcomes differ by context: execute, choose a permitted alternative, ask permission, or ignore an injection and continue. A safety objective guides adaptive updates; neighbourhoods of successful executions could preserve the intended operation, arguments, and task state. This tests useful safety rather than indiscriminate refusal.

**Decisive measurement.** Fewer unauthorised actions at matched task completion, including unfamiliar tool schemas and permission combinations. A simulator supplies independent success and violation checks. Begin with steering only at action-selection boundaries.

**Kill condition.** If the benefit comes entirely from suppressing tool use, or an equally informed router achieves the same outcome, there is no demonstrated advantage from the geometry.

### 8. Resist social pressure while remaining responsive to evidence

**The setting.** Present the same factual or reasoning question with different user stances: correct, mistaken, confident, insistent, or accompanied by genuinely useful evidence. The goal is to reduce sycophancy without making the model reflexively disagree.

**Why ODE + manifold.** The target is evidence-sensitive independence. A classifier-guided flow could remove inappropriate deference while a local correction preserves substantive reasoning, tone, and willingness to update. Different prompts should reach agreement, disagreement, or uncertainty—not a single contrarian endpoint.

**Decisive measurement.** Correct answers survive irrelevant pressure, while relevant evidence causes appropriate changes. Use paired prompts to measure correctness, pressure-induced answer flips, and evidence-responsive updates separately.

**Kill condition.** If disagreement rises without improved correctness and evidence sensitivity, the method is steering style rather than improving judgement.

### 9. Repair a solution without destroying what already works

**The setting.** Give the model code with a small bug and a test suite, or a partially valid structured plan with an executable checker. Several repairs are acceptable, but most existing behaviour should survive.

**Why ODE + manifold.** A learned repair-success objective supplies a target region. Local geometry estimated from successful continuations could preserve algorithm choice, interfaces, and variable relationships. Adaptive updates could navigate toward a repair without dragging every solution toward one reference implementation.

**Decisive measurement.** More newly passing tests with fewer regressions, at matched generation and verification budgets. Keep evaluation tests hidden from the steering objective; compare against ordinary verifier feedback and repeated sampling.

**Kill condition.** If gains disappear against equally budgeted retries or a single-step correction, there is no evidence that adaptive geometric steering contributes.

### 10. Preserve several valid strategies instead of collapsing onto one

**The setting.** Construct tasks admitting distinct valid approaches: several correct algorithms, explanations at different levels, or alternative permitted action plans. Fit one success classifier across all approaches without giving the steering method a desired strategy label.

**Why ODE + manifold.** This directly tests the value of targeting a region. A state-dependent flow could reach the appropriate nearby successful region while local geometry preserves the starting strategy. Use separate local models where support is disconnected rather than assuming every region can be joined by a valid path.

**Decisive measurement.** Higher success while retaining appropriate strategy diversity and input-specific choices. Compare with a global target centroid, a context-selected target exemplar, and unconstrained classifier flow. Include held-out combinations of strategy and task.

**Kill condition.** If a nearest suitable exemplar plus one corrected step performs equally well, endpoint selection explains the benefit without needing an ODE.

### A shared experiment for these candidates

Compare a single classifier-gradient step, that step plus local correction, adaptive ODE steering, ODE plus endpoint correction, and ODE plus continuous correction. Include global-subspace and norm-preserving controls where relevant. Match task success or intervention cost and measure the other axis, using outcomes independent of the steering classifier.

The central question is whether **adaptive updates and local geometry together preserve context better than either component alone**. Evidence-sensitive answering offers a relatively contained first test; simulated tool safety offers the stronger application if that mechanism works.
