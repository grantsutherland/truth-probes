# Evaluating the pre-registered 9B predictions

Registered in `results/prereg_9b_prediction.json` before any 9B forward pass.
Evaluated 2026-07-23 against `results/gemma-2-9b/`.

| | prediction | verdict |
|---|---|---|
| **P1** | combined-MM cross-dataset generalization is monotone increasing in `cos(theta_A, theta_B)` | **REFUTED** |
| **P2** | LR is substantially less sensitive to `cos(theta_A, theta_B)` than MM | **CONFIRMED at 2B, UNTESTABLE at 9B** (ceiling) |
| **P3** | `cos(larger, smaller)` becomes less antipodal with scale; `cos(cities, neg_cities)` stays ~orthogonal | **CONFIRMED** |
| **P4a** (`predictions_for_9b`) | the MM↔LR gap shrinks *iff* `cos(cities, neg_cities)` rises | **REFUTED** |
| **P4b-antipodal** (`REFINED_mechanism`) | combined-MM NIE degrades when constituents are **antipodal** | **UNTESTABLE** at both scales |
| **P4b-orthogonal** (`REFINED_mechanism`) | combined-MM NIE is **preserved** when constituents are orthogonal | **REFUTED** — degrades sharply at equal norm |

The prereg contains two distinct P4s — a classification-gap prediction in
`predictions_for_9b` and the NIE prediction in `REFINED_mechanism`. They are
separated here as P4a/P4b because they have different verdicts.

---

## P3 — CONFIRMED

The one clean out-of-sample success, and the strongest result in the project.
`cos(theta_A, theta_B)`, from `negation_mechanism.json`:

| pair | 2B (L13) | 9B L17 | L19 | L21 |
|---|---|---|---|---|
| larger_than + smaller_than | **−0.617** | **+0.301** | +0.321 | +0.461 |
| sp_en_trans + neg_sp_en_trans | −0.420 | −0.404 | −0.331 | −0.169 |
| cities + neg_cities | −0.023 | −0.077 | −0.022 | −0.007 |

`cos(larger, smaller)` flips sign, antipodal → positively aligned, and keeps
rising with depth. This reproduces M&T's Fig 3 scale trend (13B antipodal → 70B
common direction) in the 2B→9B interval, and it was registered in advance with
falsification criteria. `cos(cities, neg_cities)` stays ~orthogonal at every
layer, also as predicted.

Note this is `cos_AB`, between two *datasets'* truth directions — not `cos(mm, lr)`,
between two *probe types* on one dataset. Both rose at 9B; they are different
quantities and only the former is P3.

## P1 — REFUTED

The prereg's falsification criterion, verbatim: *"If combined-MM cross-dataset
generalization at 9B does NOT track cos(theta_A, theta_B) (e.g. MM generalizes
well cross-topic even when constituents are orthogonal), the averaging mechanism
is wrong."* That is what happened.

Combined-MM cross-topic signed AUROC, sorted by `cos_AB`:

| cos_AB | MM | cell |
|---|---|---|
| −0.607 | 0.999 | 2B/L11 sp_en+neg_sp → neg_cities |
| −0.420 | 1.000 | 2B/L13 sp_en+neg_sp |
| −0.404 | 1.000 | 9B/L17 sp_en+neg_sp |
| −0.348 | 0.984 | 2B/L15 sp_en+neg_sp |
| −0.331 | **0.137** | 2B/L11 cities+neg → neg_sp_en_trans |
| −0.331 | **1.000** | 9B/L19 sp_en+neg_sp |
| −0.169 | 1.000 | 9B/L21 sp_en+neg_sp |
| −0.076 | 0.986 | 9B/L17 cities+neg |
| −0.058 | 0.036 | 2B/L15 cities+neg |
| −0.023 | 0.399 | 2B/L13 cities+neg |
| −0.022 | 1.000 | 9B/L19 cities+neg |
| −0.007 | 1.000 | 9B/L21 cities+neg |

No monotone relationship. The three *most* antipodal cells all score ≈1.0 — the
opposite of the prediction — and at cos ≈ −0.02 to −0.08 MM takes 0.036, 0.399,
0.986 and 1.000. Two cells at the same cos (−0.331) differ by 0.86 AUROC.

The decisive single comparison: `cities+neg_cities → neg_sp_en_trans` went
**0.399 (2B) → 0.986 (9B)** while its `cos_AB` barely moved (−0.023 → −0.076).
Same geometry, opposite outcome.

### What actually failed — the proxy, not the identity

`aux_score_decomposition.py` separates the two. The identity itself is exact and
verified numerically on this data: for a balanced concat,
`theta_combo = ½(theta_A + theta_B)` reproduces the reported combined AUROC to
three decimals (0.986 reconstructed vs 0.986 reported; 1.000 vs 1.000).

But the identity is about **scores**, and the combined score is a
*norm-weighted* sum of unit-direction scores:

    theta_combo · x  =  ½ ( |theta_A|·(dA·x) + |theta_B|·(dB·x) )

`cos_AB` describes only the angle between `dA` and `dB`. It carries no
information about `|theta_A|` vs `|theta_B|`, nor about each score's spread on
the test set — and it is that product (call it effective weight) which decides
which constituent controls the rank ordering AUROC reads. At 9B/L17:

| combo → test | constituent A | constituent B | dominance A/B | combined |
|---|---|---|---|---|
| cities+neg → neg_sp_en | AUROC 0.003, ‖θ‖ 68.3, sd 11.9, eff 813 | AUROC 1.000, ‖θ‖ 80.7, sd 16.4, eff 1323 | 0.615 → **B wins** | 0.986 |
| sp_en+neg_sp → neg_cities | AUROC 0.821, ‖θ‖ 85.5, sd 4.2, eff 357 | AUROC 1.000, ‖θ‖ 85.8, sd 15.8, eff 1358 | 0.263 → **B wins** | 1.000 |

At 9B the aligned constituent dominates the average, so the inverted one is
outvoted and the combination succeeds. At 2B the same pair landed at 0.399 —
near chance, the signature of two constituents of comparable effective weight
cancelling. So the sign-cancellation *story* is intact; what is refuted is the
claim that `cos_AB` predicts when cancellation happens. It cannot, because it is
blind to the magnitudes that decide it.

**Caveat, stated plainly:** the 2B dominance ratio was not recomputed. The 2B
activation cache is gitignored and was produced on the laptop, not the cluster,
so the 2B row of that table is *inferred* from the observed 0.399, not measured.
Re-extracting 2B (cheap, `gpu_devel` handles 2B) would close it. Until then, the
9B decomposition is measurement and the 2B account is interpretation.

**Second caveat on the P1 test design:** only `cities+neg_cities` varies across
these cells; `sp_en+neg_sp` is at ceiling (MM ≈ 1.0) in all six. The effective
sample for testing monotonicity is one varying pair per model, which is thin
regardless of the verdict.

### What a corrected predictor would look like

This is the informative kind of refutation: it says exactly where the prediction
was mis-specified. `cos_AB` is an angle, and the quantity that actually governs
the outcome is a norm-weighted combination — so the natural next formulation
predicts combined-MM transfer from *direction together with relative magnitude*
(something like the effective-weight ratio above), not from angle alone. Noted
as the obvious next hypothesis; deliberately **not** pursued here, since fitting
a new predictor to the same twelve cells that refuted the old one would be
post-hoc curve-fitting, not a test.

## P2 — CONFIRMED at 2B, UNTESTABLE at 9B

LR − MM on the cross-topic cells:

| | cities+neg | sp_en+neg_sp |
|---|---|---|
| 2B L11 | **+0.701** | −0.002 |
| 2B L13 | **+0.425** | +0.000 |
| 2B L15 | **+0.618** | +0.016 |
| 9B L17 | +0.014 | +0.000 |
| 9B L19 | 0.000 | 0.000 |
| 9B L21 | 0.000 | 0.000 |

At 2B the prediction holds strongly and survives the layer spot-check: where MM
fails cross-topic negation, LR succeeds (this was Task C's real finding). At 9B
both probes saturate at 1.000, so there is no variance left for either to be
differentially sensitive *to*. That is a ceiling, not a confirmation — record it
as untestable rather than claiming P2 passed at scale.

## P4a — REFUTED

Registered: *"The MM↔LR gap on hard cross-topic negation
(cities+neg_cities → neg_sp_en_trans) will shrink at 9B IF
cos(theta_cities, theta_neg_cities) rises; it will persist if that cos stays
near 0."*

That cos stayed near 0 (−0.023 → −0.076, and −0.007 by L21) and the gap closed
completely (+0.425 → +0.014 → 0.000). The stated conditional is false in the
direction it explicitly ruled out. Same root cause as P1: the gap closed because
the aligned constituent came to dominate the average, which `cos_AB` does not see.

## P4b (NIE) — UNTESTABLE, and the comparison is confounded

### Every NIE must be read next to its raw shift

NIE divides the intervention's logit shift by the baseline gap `PD+ − PD−`. That
gap is a property of the **model's readout**, not of theta, and it grew 8.0×
from 2B (0.741) to 9B (5.932). So NIE fell at 9B for every condition while the
actual causal effect *grew* for every condition:

| train set | raw shift f→t 2B | 9B | ratio | NIE 2B | 9B | ratio |
|---|---|---|---|---|---|---|
| larger_than | +0.021 | +0.107 | **5.1×** | 0.028 | 0.018 | 0.63× |
| larger+smaller | +0.029 | +0.118 | **4.1×** | 0.039 | 0.020 | 0.52× |
| cities | +0.222 | +0.593 | **2.7×** | 0.299 | 0.100 | 0.33× |
| cities+neg | +0.140 | +0.401 | **2.9×** | 0.189 | 0.068 | 0.36× |

**No cross-scale NIE comparison in this project is valid without the gap printed
beside it.** In particular "cities became less causal at 9B" is false — its raw
logit shift nearly tripled. `aux_nie_intervention.py` now records
`shift_false_to_true`, `shift_true_to_false` and `gap` for exactly this reason.

### Error bars (`nie_bootstrap.json`, 10k stratified paired resamples)

All four singleton effects are significantly non-zero, `larger_than` included
(NIE f→t 0.0168, 95% CI [0.0152, 0.0184]). **These are not noise** — they are
small effects measured precisely. That distinction matters for what follows.

### Why P4b is still untestable

The antipodal test needs a causal baseline large enough that degradation could
show. `larger_than`'s effect at 9B is **1.7% of the baseline gap**, against the
71–101% M&T report in Table 2 — roughly 40–60× smaller. And per P3 the
constituents are no longer antipodal at 9B (cos −0.617 → +0.301), so the
mechanism now predicts *preservation* rather than degradation.

Observed: `larger+smaller` does not degrade — f→t is marginally *better*
(ΔNIE +0.0018, CI [+0.0007, +0.0029], significant), t→f indistinguishable
(ΔNIE −0.0012, CI [−0.0025, +0.0001], ns). Directionally consistent with the
mechanism, but it is preservation of a negligible effect, which is not evidence
about a mechanism concerning causal direction quality.

**So the sharp prediction was untestable at 2B (no causal signal: larger_than
NIE ~0.03, cos to the sp_en truth direction ~0.002) and untestable at 9B (the
constituents are no longer antipodal). Gemma-2 may simply not offer a scale where
both conditions hold at once.** That is an honest negative about experimental
availability, not a refutation of the mechanism, and it should be reported as
such rather than pursued further. The 2B readout confound is genuinely gone —
the labels hold 98.6% of the probability mass and baseline PD+ is +2.05 (vs
−0.065 at 2B) — so this is a *cleaner* null than 2B's. It remains a null.

### The cities arm degrades sharply — but the comparison is confounded

The prereg predicted orthogonal constituents → preserved NIE. Observed at 9B,
`cities+neg_cities` vs `cities`, all significant:

| | cities | cities+neg | Δ (95% CI) |
|---|---|---|---|
| raw shift f→t | +0.593 | +0.400 | −0.194 [−0.208, −0.179] |
| raw shift t→f | +0.560 | +0.030 | **−0.530 [−0.561, −0.500]** |
| asymmetry (Δt→f − Δf→t) | | | **−0.336 [−0.371, −0.302]** |

t→f is almost entirely destroyed (−95%) while f→t drops by a third. The
asymmetry the point estimates hinted at is real and large. This contradicts both
the prereg (orthogonal → preserved) and M&T's reported direction for this pair
(NIE *improves*, .77→.85 at 13B, .58→.81 at 70B). The same ordering appears at
2B (cities 0.30 → cities+neg 0.19), so it is consistent across our two scales
and consistently opposite to M&T.

**Do not publish that as a directional finding, because the design cannot support
it.** Theta here is the raw training-set mass-mean, and for a balanced concat
`theta_combo = ½(theta_A + theta_B)` has a *smaller norm* than either
constituent whenever they are not aligned. The combined intervention is
therefore a physically weaker push:

| | ‖theta‖ ratio combo/single | cos-to-test ratio |
|---|---|---|
| larger+smaller / larger_than | 0.780 | 1.106 |
| cities+neg / cities | 0.744 | 0.771 |

For the cities pair, direction quality (0.771) and push magnitude (0.744) fall by
almost the same factor, and this design cannot separate them. Note this is *not*
an implementation error: adding the raw mass-mean is exactly M&T's normalization
(`p(mu− + theta) = p(mu+)` holds identically when theta = mu+ − mu−). But that
normalization is *per-probe* — it equalizes each probe against its own training
distribution, not against another probe. Singleton-vs-combo comparisons are
confounded regardless.

One thing argues part of the effect is not magnitude: a pure norm reduction is
symmetric and cannot produce a 95%-vs-32% split, and the asymmetry CI excludes
zero decisively. A saturation account (PD+ +2.05 and PD− −3.87 sit at different
points of the readout curve, so the two push directions could saturate
differently) is available but weak: PD is a difference of log probabilities, and
logits shift approximately linearly under a small residual-stream perturbation.
Saturation bites in probability space, much less in log space. It is not zero —
LayerNorm and downstream nonlinearities are in the path — but magnitude is the
stronger confound, and it is the one the rescaling addresses.

### THE SCOPE OF THIS CONFOUND — causal arm only

**Every classification result in this project is norm-invariant and untouched.**
AUROC depends only on the ranking of scores, and scaling a direction by a
positive constant leaves rankings identical. Magnitude enters *only* where a
vector is added to the residual stream, i.e. the NIE experiment alone. The
transfer sharpening (0.84/0.88 → 0.988/0.994), the banded bimodality, P1, P2,
P4a and P3 are all rank- or angle-based and are unaffected. A reader should not
generalize this confound beyond the causal arm.

### The rescaling test — the degradation is REAL, and asymmetric

`RESCALE_TO` in `aux_nie_intervention.py` rescales each `theta_combo` to
‖theta_single‖, holding push magnitude fixed so only direction varies (job
2159250). Raw logit shifts, signed so positive = intervention worked as intended:

| condition | ‖theta‖ | shift f→t | shift t→f |
|---|---|---|---|
| `cities` | 68.48 | +0.593 | +0.558 |
| `cities+neg_cities` (raw) | 50.95 | +0.400 | +0.030 |
| `cities+neg_cities` @ norm=cities | 68.44 | +0.523 | **+0.076** |

Contrasts vs `cities`, 10k paired resamples, **all significant**:

| | raw combo | norm-matched | share of degradation that was magnitude |
|---|---|---|---|
| Δ shift f→t | −0.194 [−0.208, −0.179] | −0.071 [−0.085, −0.057] | 64% |
| Δ shift t→f | −0.527 [−0.558, −0.497] | **−0.482 [−0.510, −0.455]** | **9%** |
| asymmetry | −0.333 [−0.368, −0.300] | **−0.411 [−0.443, −0.380]** | — |

**The verdict: magnitude explained most of the f→t degradation and almost none
of the t→f degradation.** At equal push magnitude, `cities+neg_cities` still
loses 86% of the true→false causal effect (0.558 → 0.076) while losing only 12%
of the false→true effect (0.593 → 0.523). The asymmetry is *larger* after
norm-matching, not smaller.

So P4b's orthogonal arm is **refuted**, in the informative direction: orthogonal
constituents degrade the causal effect even though they leave classification
untouched — the same pair classifies at 0.986–1.000 cross-topic (better than
`cities` alone, which is *inverted* at 0.003). The registered prediction said
orthogonal constituents merely rotate the direction and should be preserved
causally. They are not.

Note what this does and does not do to the broader thesis. **It reproduces M&T's
classification-vs-NIE dissociation** — combined training that helps
classification while hurting the causal effect — but attaches it to *orthogonal*
constituents rather than the antipodal ones our mechanism predicted, and with
the opposite sign to M&T's own `cities+neg_cities` row (they report NIE
*improving*, .77→.85 at 13B and .58→.81 at 70B). Their numbers come from a
different model family and a different evaluation set, so this is not a direct
contradiction of their measurement; it is our mechanism's prediction failing on
our data.

The aligned pair behaves as the mechanism expects. `larger+smaller` at equal
norm *improves* over `larger_than` on both directions (Δ shift f→t +0.043
[+0.036, +0.050]; t→f +0.023 [+0.014, +0.032], both significant) — combining
aligned constituents helps. But these remain 1.7–2.5% of the baseline gap, so
this arm confirms a direction of effect, not a magnitude worth leaning on.

**Open and deliberately not chased:** why the collapse is specific to the
true→false direction. Adding `theta_combo` to a false statement still works
nearly as well as `theta_cities`; subtracting it from a true statement does not.
A pure magnitude account is now excluded, and the saturation account is weak in
log space (see above), so this is a real directional asymmetry without an
explanation. Recorded as an observation, not a mechanism.
