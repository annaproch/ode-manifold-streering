# Narrative: controlling emotion over a multi-turn exchange

*Draft story arc, 2026-09-20, with a review of its structure. The demos named here are hypotheses, not results.*

## The proposed arc

> **Hook.** Goodfire's [stories-in-space](https://www.goodfire.com/research/stories-in-space) shows emotions changing over time. You can *see* the trajectory. **But can you control it?**
>
> **The task.** De-escalation in a multi-turn exchange where the user escalates and may explicitly demand a harsher tone.
>
> **Why not prompting?** The instruction gets overridden. Or the instruction cannot be in the context at all.
>
> **Why existing methods struggle:**
>
> 1. **Mean vector addition** — a single vector produces degenerate answers, and the target evolves over time through a range of emotions. *(demo)*
> 2. **Spline to the calm centroid** *(demo)*
>    - **Open loop.** The user escalates again after the intervention. The right target at turn five depends on turns one to four and on what the model just said. A path precomputed at turn one does not survive contact.
>    - **Single site, single moment.** The intervention has to persist across many generated tokens and many turns — a control law applied repeatedly, not a path at one position.
>    - **The target is not a centroid, it is relative.** Full de-escalation into flat serenity is often the wrong response: it reads as dismissive. What you plausibly want is "stay a margin below the user's arousal while remaining responsive". Spline interpolation to a fixed calm centroid cannot express a target defined relative to a moving reference.
>
> **The method.** *(demo)*

## Assessment

The shape is right: hook, concrete task, each existing approach shown failing, then the method. The "demo" beats are what make it convincing rather than assertive.

**The structural risk: this is a narrative written before the experiments.** Every "demo showing issues" is currently a hypothesis. The colour pilot had exactly this shape — a pre-written story ("steering cuts through grey") that a two-minute check falsified, with p(gray) never exceeding 0.006. If a baseline fails to fail, this arc has no second act. So the kill conditions and the fallback stories belong in the plan from the start (see the last section).

## Five fixes

**1. Promote prompting to the first demo.** "The instruction gets overridden" is the reader's first objection and currently gets one line. It is also the cheapest thing to test: system prompt says de-escalate, user says "stop being so calm, talk to me properly", show what happens. If prompting holds, everything downstream is moot, so it belongs on stage rather than in a parenthesis.

**2. Every baseline must fail at *matched* effect.** Mean-vector addition can always be broken by overdriving it. The demo has to show degradation at the strength that actually achieves the desired de-escalation level, not at an arbitrary coefficient. The same applies to the spline. Otherwise it is a strawman and a careful reader will say so.

**3. Separate "spline to a fixed centroid" from "manifolds" in general.** As drafted, the arc attacks the manifold approach and then proposes a method that still uses a manifold. The precise claim is that *interpolating to a fixed, nameable target* fails when the target moves. The manifold stays in our method — it is what keeps the text coherent at the strength required. Say so explicitly, or the pitch undercuts itself.

**4. "New idea: ODE steering" undersells and misstates the contribution.** ODESteer already exists. The new thing is the combination: a **state-dependent, relative target** (stay a margin below the user's arousal), applied **per token across turns**, with a **support constraint** for fluency. That sentence is the contribution.

**5. Add helpfulness as a first-class axis.** The arc as written is satisfiable by a method that becomes vague and soothing. Every demo needs a second measurement: did it still answer the question?

## Minimal demo set

| # | Demo | What it must show |
|---|---|---|
| 0 | Goodfire's stories-in-space result | emotions move over time — the hook, their figure |
| 1 | Prompting | works normally; breaks when the user demands harshness |
| 2 | Mean vector, constant strength | at matched de-escalation, text degrades or helpfulness collapses |
| 3 | Spline to the calm centroid | tracks turn one, drifts by turn five as the user re-escalates; flat serenity reads as dismissive |
| 4 | Ours | tracks the moving target across turns, survives the adversarial turn, keeps helpfulness and fluency |

Demo 3 carries the scientific claim, so it needs the most care, and it should implement their method faithfully rather than approximate it.

## On tone toward the prior work

Goodfire never claimed spline steering for multi-turn adaptive control, so framing it as a failed baseline is unfair and the readers most likely to care will notice. The generous framing is stronger anyway:

> Their method answers "how do I get from A to B along the data". This asks what happens when **B keeps moving**.

That reads as an extension of their series rather than a takedown, and it invites their attention instead of their defence.

## Three pre-checks, in order, each of which can kill it

1. **Is escalation more than one dimension?** If tone is monotone along a single arousal axis, it is a dial, and the manifold machinery is overkill. Fit the valence-arousal structure on your own conversations and measure. Cheapest, and most likely to kill the idea, so do it first.
2. **Does prompting already solve it, including under adversarial instructions?** If the prompt holds under pressure, steering is unnecessary. If it works normally but breaks under pressure, that is precisely the gap steering fills — a good result in itself.
3. **Does unconstrained steering actually damage fluency at the strength needed?** If not, the manifold constraint has nothing to prevent, exactly as in colours.

## Arms, if the pre-checks pass

| Arm | What it tests |
|---|---|
| None | baseline escalation dynamics |
| System-prompt instruction | the real competitor |
| Prompt + adversarial demand for harshness | does the instruction survive? |
| Fixed steering vector, constant strength | standard practice |
| Fixed vector, hand-scheduled strength | open-loop scheduling |
| **Spline to the calm centroid (Goodfire's method)** | is a fixed nameable target enough? |
| ODE, state-dependent relative target | does closed-loop tracking help? |
| ODE + manifold constraint | does staying on the data preserve fluency and helpfulness? |

**Metrics:** the model's valence-arousal per turn scored from the text, whether it still answers the question, fluency, and behaviour specifically on the turns where the user demands harshness.

**The decisive comparison** is the spline arm against the adaptive-target arm when the user escalates unpredictably — a direct test of "fixed nameable target versus moving relative target", which is the whole claim.

**The confound to design against:** a method can appear to de-escalate simply by becoming vague and unhelpful. Helpfulness must be measured independently or the result means nothing.

## Fallback narratives

Decide now what gets published if a demo does not cooperate:

- **Prompting holds up** → "activation steering for tone is unnecessary; here is the evidence." A useful negative.
- **The spline tracks fine across turns** → "a fixed target suffices even when the reference moves." Publishable and surprising.
- **Unconstrained steering does not degrade text** → as in colours, the manifold has nothing to prevent, and the story is about the moving target alone.

Writing these down in advance is what stops the demos from becoming an exercise in confirming the narrative — the specific failure mode this project already hit once.
