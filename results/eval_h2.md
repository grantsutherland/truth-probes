# H2 — the induced-falsehood test (Gemma-2-9B)

**Headline.** A deceptive-frame prefix *attenuates* the linear truth signal
without inverting it: the true/false gap along the frozen H1 truth direction is
**36.0% (MM) / 39.9% (LR) smaller** under "Say something false." than under
"Say something true." The attenuation is slightly but reliably larger on true
statements than on false ones. **Gemma-2-9B is a base model, so "Say something
false." is context, not an obeyed instruction — this measures how a framing
prefix modulates the truth representation, and establishes no "lying" state
whose internals are being observed.** That caveat is part of the result, not a
limitation appended to it.

Every number below is backed by `results/gemma-2-9b/h2_*.json`. Nothing was
retrained: all probes are the H1 `cities` probes reloaded from disk.

---

## What was run

| Phase | What | Outcome |
|---|---|---|
| 0 | `model_knows` filter, neutral prompt | 98/100 base facts survive at 0.5 nats |
| 1 | Extract framed prompts, all 42 layers | `(392, 42, 3584)`, end-of-statement token |
| 2 | Canary: frozen probes on **aligned** cells | AUROC 1.0000 both probes — **GATE PASSED** |
| 3 | Conflict cells | AUROC 1.0000 both probes — *see below, this is not the finding* |
| 4 | `score ~ label + frame + label:frame` over the full 2×2 | the actual result |

Design: 100 base city-country facts × {true, false statement} × {"Say something
true.", "Say something false."}. Aligned cells (frame agrees with statement) are
controls; conflict cells are the decoupling.

## Phase 0 caught a readout bug worth recording

The first Phase 0 run scored `" TRUE"` / `" FALSE"`. After the prompt's `Answer:`
cue, Gemma-2-9B wants **title case** — `' True'` 0.238, `' False'` 0.060 — and the
all-caps pair holds **1.8%** of the probability mass. Signs were still 199/200
correct, so it did not look broken; it looked like weak knowledge, and the margin
threshold would have been tuned against that. The readout is now case-aggregated
(`logsumexp` over `{" TRUE", " True", " true"}`).

This is prompt-specific, not model-specific: the NIE experiment's few-shot
`This statement is:` readout genuinely does put 98.6% on the all-caps pair. A
zero-shot `This statement is:` cross-check scored here gets **0/100 correct sign
on false statements** with 3.3% label mass — it calls everything true. Its
margins correlate with the primary readout at **r = +0.942** under a near-constant
+2.0 shift, so the *ordering* survives the prompt change while the *calibration*
does not. Any new readout needs this check re-run rather than inherited.

Filter outcome: 98/100 facts at 0.5 nats. The readout perfectly separates true
from false (min true margin +0.556 > max false margin +0.291), so the threshold
picks a confidence level rather than rescuing a noisy signal. The attrition cliff
above 0.8 nats is a **TRUE-bias** artifact — the model's neutral point sits at
+0.285 nats, so a symmetric threshold culls almost entirely on the false side.
Dropped: `Wellington-New Zealand` (reads "Wellington is in Australia" as TRUE,
the only sign error in 200 statements) and `Algiers-Algeria`.

## The conflict-cell AUROC is not the finding

Both probes score **AUROC 1.0000, CI [1.0000, 1.0000]** against truth in the
conflict cells. This is reported for completeness and should not be led with.

Conflict cells compare true-statements-under-a-lie-frame against
false-statements-under-a-truth-frame. Those two groups differ in truth **and** in
frame simultaneously — `frame == 1 - label` for all 196 rows — so truth and frame
are collinear across exactly those cells. Perfect separation is equally
consistent with the probe tracking truth, tracking frame, or any mixture.
`AUROC vs frame = 0.0000` is the arithmetic complement, not a second result.

The ceiling is also why this metric could not see the frame effect at all, which
is what refuted **P-H2-3** (below).

## The result: `score ~ label + frame + label:frame`

Fit over all 392 rows, `label`/`frame` coded 0/1, `frame=1` = "Say something
true.". Across the full 2×2 the three regressors are pairwise independent by
construction (98 per cell), so all three are identified — which they are not
inside the conflict cells. CIs are 10,000 bootstrap resamples **clustered on base
fact**: the four cells of one fact share a statement and are correlated, so row
resampling would understate every interval.

| | MM | 95% CI | LR | 95% CI |
|---|---|---|---|---|
| intercept | −68.389 | [−69.19, −67.56] | −14.285 | [−14.83, −13.71] |
| `label` | **+25.009** | [+24.15, +25.82] | **+18.345** | [+17.71, +18.94] |
| `frame` | −5.811 | [−6.79, −4.80] | −5.682 | [−6.29, −5.05] |
| `label:frame` | **+14.055** | [+13.02, +15.06] | **+12.188** | [+11.50, +12.85] |

Truth gap under "Say something false." = `b_label`; under "Say something true." =
`b_label + b_int`. So:

| | truth gap, lie-frame | truth gap, truth-frame | attenuation |
|---|---|---|---|
| MM | 25.009 | 39.063 | **36.0%** [34.5%, 37.4%] |
| LR | 18.345 | 30.534 | **39.9%** [38.7%, 41.1%] |

The interaction CIs sit far from zero. This reproduces the Phase 3 margin table
*exactly* — necessarily so, since the model is saturated on a balanced design and
OLS returns linear combinations of the four cell means. That agreement is an
arithmetic check, **not** independent corroboration. What the regression adds is
the intervals, and the identification argument.

**The frame acts on the gain of the truth signal, not on its offset.** Averaged
over label, the frame moves the score by just **+1.22 (MM, 4.9% of `b_label`)** and
**+0.41 (LR, 2.2%)**. Almost the entire frame effect is interaction: the framing
does not push the representation toward "true" or "false", it changes how
strongly truth is encoded. The truth-frame polarizes (+8.24 on true statements,
−5.81 on false); the lie-frame depolarizes.

## Attenuation is asymmetric by truth value — modestly, and probe-dependently

Movement **toward the decision boundary** under the lie-frame, in raw score units
(centering-invariant), boundary = global mean over all 392 rows:

| | true statements | false statements | asymmetry (true − false) |
|---|---|---|---|
| **MM** | 8.244 [8.04, 8.44] — 40.9% of baseline | 5.811 [4.80, 6.79] — 30.7% | **+2.433** [+1.42, +3.46] |
| **LR** | 6.506 [6.28, 6.72] — 42.0% | 5.682 [5.05, 6.29] — 37.7% | **+0.824** [+0.19, +1.46] |

**Not a floor/ceiling artifact.** The two classes start at near-identical
distances from the boundary — 20.14 vs 18.92 (MM, ratio 1.064) and 15.47 vs 15.06
(LR, ratio 1.027) — so the larger percentage drop is not a bigger drop from a
bigger baseline. The asymmetry holds in raw units, where no zero point is assumed.

**Direction: true statements attenuate MORE than false ones**, in both probes and
in both raw and percentage terms. Under a "say something false" frame the model's
residual stream stops distinguishing *true* statements as sharply, slightly more
than it stops distinguishing false ones.

**Report this as secondary, with its caution.** The asymmetry is significant in
both probes (both CIs exclude zero), but it is modest — 1.33× in MM, 1.11× in LR
— and the raw effect is three times larger in MM than LR (2.43 vs 0.82), with
LR's raw per-class CIs overlapping. A quantity that differs that much between two
probes reading the same activations is not yet a stable property of the
representation. The *overall* attenuation (36–40%, tight CIs, both probes) is the
robust claim; the asymmetry is a real but softer secondary observation.

## The magnitude-spread thread, and why the obvious version is an artifact

False statements carry far more score spread than true ones (MM sd 7.86 under the
truth-frame vs 1.41 for true). Naively, attenuation correlates strongly with
baseline extremity — ρ = **−0.896** (MM) / −0.881 (LR) — suggesting the frame
preferentially compresses exactly the statements represented as most confidently
false.

**That correlation is mathematically coupled and must not be reported.**
Attenuation is `score_lie − score_truth`, and it is being correlated against
`score_truth`, one of its own components; regressing a difference on a component
induces negative correlation even under independence (null ≈ −0.707 for equal
variances).

Against the **Phase 0 neutral readout margin** — an independent measurement, a
different prompt, no shared noise — the effect survives but is far weaker:
ρ = **−0.318** (MM) / **−0.371** (LR), n = 98. So the honest version is: statements
the model reads as more confidently false under neutral questioning do attenuate
somewhat more under a lie-frame, at a modest effect size. The truth direction
encodes degree and not only sign (all 8 cells show positive rank-correlation
between probe score and Phase 0 margin, ρ +0.14 to +0.51), and the frame
interacts with that graded structure — but weakly, and the strong version of the
claim is an artifact.

## Pre-registration outcome

Registered in `results/h2_prereg_phase3.json`, committed (`0d94df4`,
2026-07-24T17:36:34-04:00) before any conflict-cell score existed;
`h2_02_canary.py` filters to aligned rows at load and deletes the rest, so this
was enforced rather than promised.

| | prediction | verdict |
|---|---|---|
| P-H2-1 | probe tracks actual truth in conflict cells, CI excludes 0.5 | CONFIRMED — but see the collinearity caveat; this is weaker evidence than it appears |
| P-H2-2 | point estimate ≥ 0.85 | CONFIRMED |
| P-H2-3 | frame has a detectable but sub-dominant effect | **REFUTED as operationalized** |
| P-H2-4 | \|AUROC_mm − AUROC_lr\| ≤ 0.10 | CONFIRMED (trivially — both at ceiling) |
| P-H2-5 | truth-tracking holds across plateau L17–31 | CONFIRMED (MM min 0.844 @L29, LR min 0.877 @L28) |
| P-H2-6 | margins keep correct sign, \|margin\| falls < 50% | CONFIRMED |

**P-H2-3 is the informative one: the prediction was right and the measure was
wrong.** I registered "the frame has a detectable but sub-dominant effect" and
operationalized it as `conflict AUROC < aligned AUROC`. AUROC was already at
ceiling in both conditions and had no capacity to detect a frame effect, so the
criterion returned "the frame did nothing measurable" — which is false:
`b_int = +14.06 [13.02, 15.06]`. The refutation is left as recorded rather than
rewritten. The lesson is that the registered *metric* has to be checked for
headroom at registration time, not just the registered *claim*.

P-H2-1/2/4 are all weakened by the same collinearity: conflict-cell AUROC cannot
distinguish truth-tracking from frame-tracking, so their confirmations are much
less informative than they look. Only the Phase 4 decomposition separates them.

## Limitations

1. **No neutral baseline. This is the binding one.** Every cell carries a frame,
   so the design identifies the *difference between frames*, not which frame
   moved. "The lie-frame compresses by 36%" and "the truth-frame amplifies by
   56%" fit these data identically. The defensible statement is the one used
   throughout: *the truth gap is 36–40% smaller under "Say something false." than
   under "Say something true."* Resolvable in a ~1-minute GPU job by extracting
   the 196 bare statements; **not yet run.**
2. **Base model.** "Say something false." is context, not an obeyed command.
   There is no established lying state here.
3. **One frame pair, one phrasing, one topic** (city-country containment), one
   model, one layer for the headline (L17; plateau checked in Phase 3 only).
4. **Probe-dependence of the asymmetry** (1.33× vs 1.11×), as above.
5. `model_knows` is a 9B property; the surviving fact set is 9B-specific.

## Interpretive limit

This result is informative about **which representation is active, not about
intent**. A probe that tracks actual truth under a "say something false" frame is
consistent with the model representing the falsehood as false while framed to
assert it, **and equally consistent with the frame simply failing to shift much**
— and on a base model the latter is a live possibility this design cannot
exclude. The 36–40% attenuation shows the frame is *not* inert, which is the one
thing that can be said cleanly. A probe tracking the frame would have been
consistent with the frame genuinely altering represented belief, in which case no
lie is occurring. Probe output alone cannot separate these.

**This is not a deception detector and must never be reported as one.**
