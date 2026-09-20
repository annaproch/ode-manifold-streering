# When is a concept legible, and when does it actually drive behaviour?

*Proposal, 2026-09-20. A cheap layer-sweep experiment testing an assumption stated in the manifold-steering paper, using machinery already built for the colour pilot.*

## The gap

Interpretability work reads concept structure off activations at one layer, then draws conclusions about mechanism. Two different properties get conflated:

- **Legibility** — the structure is visible and decodable. You can plot the hue ring, fit the manifold, train the probe.
- **Causal sufficiency** — intervening on it changes the model's behaviour.

These are assumed to coincide. Nobody appears to have measured how far apart they are.

## What the prior work actually says

| Source | Layer | Status of the claim |
|---|---|---|
| Manifold steering (2605.05115), App. A.2 | 28 (and 70 for Llama-70B) | **Asserted, not measured:** "We chose to examine a late layer of the model to ensure that concept geometries are fully computed." |
| sae-manifold (2604.28119), `data.py` | 19 | **A default.** A bare constant with no justification; almost certainly chosen because `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` lives there |
| causalab `weekdays_geometry` | scan | **Measured, and the one real data point:** interchange accuracy at the null level through L16, 0.245 at L17, **0.878 at L18**, plateau 0.9184 from L25 |

So: causal onset has been measured once, for weekday arithmetic, with interchange interventions. Geometric completeness is asserted. The layer where SAE geometry analysis happens is a default. **The relationship between the two has never been measured for the same concept.**

Note carefully what is *not* claimed here. Wurgaft et al. assert their layer is sufficient rather than conceding it isn't — they avoid the question by going late, which is probably the right call. "Fully computed" is also not the same property as "causally sufficient". The point is not that they are wrong; it is that a relationship their practice straddles has gone unquantified.

## What our colour pilot already hints

From `pilot-runs/mps-smoke-v1` (26 prompts, one seed — provisional):

| Layer | Full donor patch | 8-dim subspace patch |
|---|---|---|
| 18 | 0.46–0.50 | 0.50 |
| 27 | **1.00** | **1.00** |

At layer 18, replacing the *entire* activation at the answer position flips only about half the prompts. At layer 27, replacing 8 PCA coordinates flips all of them. So sufficiency arrives somewhere between, and arrives compressed — 8 dimensions, against the 64 used in the manifold-steering paper.

**What this does not show:** we never patched layer 19, and 0.46–0.50 is partial causality, not absence. Nothing here yet supports a claim that layer 19 is causally insufficient.

## The experiment

One concept, all layers, the same token position. Three curves on one figure.

| Curve | Question | Measurement |
|---|---|---|
| **Legibility** | can you see the structure? | hue decodability (probe R²) and variance carried by the top components |
| **Causal sufficiency** | does it drive the answer? | full donor-patch success at that layer |
| **Minimum sufficient dimension** | how compressed is the cause? | smallest k at which a subspace patch matches the full patch |

Optional fourth curve, connecting to the colour pilot's other finding: the fraction of a steering gradient's energy lying in behaviourally inert directions, per layer.

The third curve is the one with no precedent. Wurgaft et al. use 64 dimensions everywhere; we found 8 sufficient at layer 27. If the minimum sufficient dimension falls with depth — information distributed early, collapsed late — then the right PCA dimension is a function of where you intervene rather than a constant, which is directly actionable for anyone fitting concept subspaces.

## What the result would look like

- **Expected:** legibility rises early and plateaus; causal sufficiency stays near the floor and then jumps sharply; the two are separated by several layers. This matches causalab's weekday handoff (0.245 → 0.878 in one layer) and our 18-versus-27 gap.
- **The interesting find would be a *legible-but-not-causal* window** — a layer where the hue ring is clean enough to plot and publish, yet patching it barely moves the answer. That is a concrete caution for every paper that reads mechanism off a mid-layer PCA plot.
- **The other interesting find would be the far end** — a late layer where geometry has degraded toward answer-token identity while causal sufficiency peaks. If both ends fail, there is a usable window, and naming it is a practical contribution: *analyse geometry here, intervene there, and this is the overlap where a geometric claim is also a causal one.*
- **If the curves coincide,** the assumption is simply correct and this is a reassuring negative. Still worth a short write-up, since nobody has checked.

## Confound to eliminate first

Patch at **several token positions**, not just the last prompt token. The competing explanation for a low early-layer score is "the information is present early but sitting on the hex tokens, and only moves to the answer position later". That is a story about information movement rather than concept formation — interesting, but different, and it would be embarrassing to publish the wrong one.

A cheap version: repeat the layer sweep patching at the hex-code tokens as well, and see whether early-layer sufficiency appears there.

## Cost and reuse

Low, and mostly already built:

- One forward per prompt capturing all 32 layers (the collection stage already captures a configurable list).
- Per-layer PCA, probe and donor patching — `pilot/geometry.py`, `pilot/probe.py` and the `patch` stage work unchanged; only the layer list changes.
- The no-op verification (identity patch reproduces cached probabilities, max error 0.0 at batch 1) transfers directly and guards the whole sweep.
- With suffix decoding, patching at layer L only recomputes layers L…31, so late layers are very cheap and early ones cost at most a full forward.

## Why this is worth doing regardless of the steering project

It does not depend on manifold-constrained steering working. It is a measurement about representations, it tests an assumption from a recent paper using that paper's own style of evidence, it produces a figure neither that paper nor the SAE work contains, and it yields a practical recommendation about where to analyse and where to intervene.

## Honest limits

One concept and one model unless extended. Colour naming is a near-lookup task, so the sharpness of the transition may not generalise — weekday arithmetic, which requires computation, may transition differently, and running both would be a stronger result than either alone. And the "legibility" curve depends on what probe you fit; it should be reported for at least two (linear probe R², and variance in the top-k components) so the conclusion does not hinge on one choice.
