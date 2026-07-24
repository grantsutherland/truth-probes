# Linear truth probes in Gemma-2

An interpretability study of the linear "truth direction" in **Gemma-2-2B** and
**Gemma-2-9B**. It reproduces the core result that LLMs linearly represent the
truth of simple factual statements (Marks & Tegmark, 2024), compares two probe
types (logistic regression and mass-mean), and tests a **pre-registered
mechanistic account** of how mass-mean probes behave when trained on a dataset
concatenated with its negation.

Every number below is backed by a JSON artifact in `results/<model>/`.

## Headline

**The truth direction transfers across topics, and it sharpens sharply with
scale.** Cross-topic signed transfer AUROC (train on one dataset, test on
another, no sign oracle):

| | MM | LR |
|---|---|---|
| Gemma-2-2B (L13) | 0.8425 | 0.8781 |
| Gemma-2-9B (L17) | **0.9882** | **0.9939** |

**A pre-registered geometric prediction was confirmed out-of-sample** (P3), and
**three of our own registered predictions were refuted** by the 9B run (P1, P4a,
and the orthogonal arm of P4b). All are reported in full in
[`results/eval_9b.md`](results/eval_9b.md); the refutations are as informative as
the confirmation and are not buried. The causal thread in particular produced one
structural null and one refutation — reported as such rather than pursued until
it produced something flattering.

**And under a deceptive frame, that direction bends without breaking.** Framing
the model with "Say something false." shrinks the true/false gap along the frozen
truth direction by **36.0% (MM) / 39.9% (LR)** relative to "Say something true."
— it never inverts. Gemma-2-9B is a *base* model, so that prefix is context
rather than an obeyed instruction: this measures how framing modulates the truth
representation, and establishes no "lying" state whose internals are being
observed. See [`results/eval_h2.md`](results/eval_h2.md). **Not a deception
detector.**

---

## What was tested

- **H1 (reproduction).** LR and MM probes separate true from false and transfer
  across topically and structurally different datasets, indicating a general
  truth direction rather than a dataset-specific artifact. **Complete at both
  scales.**
- **Mechanism (the original contribution).** For a balanced concatenation the
  mass-mean direction obeys an exact identity,
  `theta_combo = ½(theta_A + theta_B)`, so the combined probe's score is the
  average of its constituents' scores. This was registered as a candidate
  explanation for a dissociation M&T explicitly flag as open (Section 7.1 /
  the paragraph before it): combined training improves *classification* while
  hurting the *causal* effect. **Partly confirmed, partly refuted — see below.**
- **H2 (induced falsehood).** Does the truth direction still track a statement's
  actual truth when the model is framed to assert a falsehood? Run on 9B, with
  the Phase 3 prediction pre-registered and committed before any conflict-cell
  score existed. **Complete.** A deceptive-frame prefix *attenuates* the truth
  signal by 36–40% without inverting it. Full evaluation in
  [`results/eval_h2.md`](results/eval_h2.md).

## Method

**Model and tooling.** Gemma-2 via **TransformerLens**, residual-stream
activations at `hook_resid_post`, gathered at the **end-of-statement token**.
2B ran locally on Apple MPS (verified numerically identical to CPU); 9B ran on
an H200 on the Yale Misha cluster. Probes are small `torch.nn.Module`s over a
shared base (`src/probes.py`): LR is a linear layer trained with Adam on
`BCEWithLogitsLoss`; MM is the difference-in-means direction
`theta = mean(true) − mean(false)`, with an optional covariance correction.

**Data.** Six Marks & Tegmark `geometry-of-truth` datasets: `cities` (1496),
`neg_cities` (1496), `sp_en_trans` (354), `neg_sp_en_trans` (354),
`larger_than` (1980), `smaller_than` (1980). All balanced 50/50. H2 adds
`data/induced_lies.csv` (400 rows, hand-built), kept clearly separate from the
M&T base data and never used to fit anything.

**Layer selection.** All layers are extracted once and cached, then swept for
MM accuracy averaged over datasets; the peak is committed. 2B chose **L13 of
26**; 9B chose **L17 of 42**. The layer is not hardcoded anywhere — it is
derived from the committed sweep by `src/config.py`, which also refuses to run
if the sweep on disk came from a different model.

**Metrics.** AUROC is the headline, because it is threshold-free and
centering-invariant. Two protocol decisions matter and are easy to get wrong:

- **The bias does not transfer; the direction does.** A first pass reported raw
  accuracy and hit exactly 0.500 in 16 cells — a *bias-transfer artifact*
  (constant prediction), not a direction failure. Correct protocol is
  per-dataset self-centering at evaluation time.
- **Signed, never sign-folded.** Folded metrics `max(a, 1−a)` use test labels to
  pick each direction's sign, which is a **sign oracle**. Only signed numbers
  are comparable to M&T's table. Both are recorded, the folded ones tagged
  `_SIGN_ORACLE`.

## Results

### In-distribution (`probe_accuracy.json`)

LR reaches 1.000 held-out accuracy on all six datasets at both scales; MM ranges
0.975–1.000 (2B) and 0.977–1.000 (9B). A leakage audit
(`leakage_check.json`) found **zero exact and zero near-duplicate train/test
overlap** under seed 0, so the 1.000s are genuine. (Caveat recorded there: the
two translation sets each contain 3 duplicate strings that do not straddle the
split under this seed but could under another.)

### Transfer (`transfer.json`) — H1

Cross-topic, affirmative→affirmative, signed, no oracle:

| | MM | LR |
|---|---|---|
| 2B, excluding logical complements | 0.8425 | 0.8781 |
| 2B, including them | 0.7040 | — |
| 9B, excluding complements | 0.9882 | 0.9939 |
| 9B, including them | 0.9875 | 0.9948 |

The complement exclusion was declared a priori (`larger_than` and
`smaller_than` are complements *by construction*), not chosen after seeing the
cells. **At 9B the exclusion stops mattering** — 0.9875 vs 0.9882 — because the
complement pair stops inverting: `larger_than→smaller_than` goes **0.003 (2B) →
0.973 (9B)**, tracking the geometric change described under P3 below.

**The failures are structured, not noisy.** Off-diagonal cells are strongly
bimodal — the direction is either right or exactly backwards, almost never
confused:

| signed AUROC band | 2B (MM) | 9B (MM) |
|---|---|---|
| ≥ 0.9 | 9 | **19** |
| 0.7–0.9 | 6 | 1 |
| 0.3–0.7 | 8 | **1** |
| 0.1–0.3 | 1 | 5 |
| < 0.1 | 6 | 4 |

Bimodality *strengthens* with scale: 28 of 30 cells sit at the extremes at 9B.
The off-diagonal signed mean (0.597 at 2B, 0.700 at 9B) falls in the empty
valley of this distribution and **must not be quoted as a summary** — no real
cell takes that value.

Inversions track two known a-priori causes: **negation**
(`sp_en_trans ↔ neg_sp_en_trans` ≈ 0.001) and **logical complementarity**
(`larger_than ↔ smaller_than` ≈ 0.003 at 2B).

### Negation is a depth-dependent rotation

`negation_by_layer.json` sweeps every layer. Both negation pairs trace the same
trajectory — antipodal early, orthogonal by mid-depth — reproducing M&T's
Appendix C. They differ in *rate*: factual (`cities`) negation rotates earlier
than translation negation, a sustained cosine gap across L4–17.

This killed a more interesting claim of ours. "Negation cleanly inverts
translation but *degrades* cities, as a statement-type property" does **not**
survive the sweep: at L8–11 both pairs invert cleanly. The layer-13 appearance
was a slice through a rotation — and L13 is the *global* cosine extremum for
`cities↔neg_cities` across all 26 layers, i.e. the accuracy-selected layer is
coincidentally the worst possible layer for that one cell. A layer-robustness
re-run (`transfer_by_layer.json`) confirms the headline is not layer-cherry-picked:
cross-topic transfer holds at 0.72–0.90 (MM) across the whole L11–17 plateau.

### The mechanism, and what the 9B run did to it

The identity is exact and verified numerically: reconstructing the combined
direction as `½(theta_A + theta_B)` reproduces the reported combined AUROC to
three decimals (`score_decomposition.json`). The *predictions built on it* fared
unevenly. Full evaluation in [`results/eval_9b.md`](results/eval_9b.md):

| | prediction | verdict |
|---|---|---|
| **P3** | `cos(larger, smaller)` becomes less antipodal with scale; `cos(cities, neg_cities)` stays ~orthogonal | **CONFIRMED** |
| **P1** | combined-MM cross-dataset generalization is monotone in `cos(theta_A, theta_B)` | **REFUTED** |
| **P2** | LR is much less sensitive to `cos(theta_A, theta_B)` than MM | confirmed at 2B, **untestable at 9B** (ceiling) |
| **P4a** | the MM↔LR gap shrinks *iff* `cos(cities, neg_cities)` rises | **REFUTED** |
| **P4b** | combined-MM causal effect degrades when **antipodal**, is preserved when **orthogonal** | antipodal arm **UNTESTABLE**; orthogonal arm **REFUTED** |

**P3 — confirmed out-of-sample.** Registered before any 9B forward pass:
`cos(larger, smaller)` goes **−0.617 (2B) → +0.301 / +0.321 / +0.461** at 9B
L17/19/21 — a sign flip from antipodal to aligned, rising with depth, exactly
M&T's 13B→70B trend reproduced in the 2B→9B interval. `cos(cities, neg_cities)`
stayed ~orthogonal as predicted. This is the strongest result here, and it is
independently visible in the transfer matrix as the complement pair ceasing to
invert.

**P1 — refuted, informatively.** `cities+neg_cities → neg_sp_en_trans` moved
0.399 → 0.986 while its `cos_AB` barely changed (−0.023 → −0.076). Sorted by
`cos_AB` there is no monotone relation at all. The *identity* is not what
failed — the **cosine proxy** is. The combined score is a *norm-weighted* sum,
`theta_combo·x = ½(‖theta_A‖(dA·x) + ‖theta_B‖(dB·x))`, and cosine is blind to
magnitude. At 9B the aligned constituent carries 1.6× the effective weight
(norm × test-set spread) of the inverted one and simply outvotes it. Sign
cancellation survives as a mechanism; "cosine predicts when cancellation
happens" does not. A corrected predictor would combine direction *and* relative
magnitude — noted as the next formulation, deliberately not fitted here.

### The causal arm (`nie_intervention.json`, `nie_bootstrap.json`)

Following M&T Section 6.1: add `theta` to false statements' residual stream at
the chosen layer, subtract from true ones, and read the change in
`P(TRUE) − P(FALSE)`. All numbers below carry 95% CIs from 10,000 stratified,
paired bootstrap resamples over the 352 test statements.

**Report raw logit shifts next to every NIE.** NIE divides by the baseline gap
`PD+ − PD−`, which is a property of the *model's readout*, not of `theta` — and
that gap grew **8.0×** from 2B (0.741) to 9B (5.932). So every NIE fell at 9B
while every underlying causal effect *grew*:

| train set | raw shift f→t 2B → 9B | NIE 2B → 9B |
|---|---|---|
| `cities` | +0.222 → **+0.593** (2.7×) | 0.299 → 0.100 (0.33×) |
| `larger_than` | +0.021 → **+0.107** (5.1×) | 0.028 → 0.018 (0.63×) |

"Cities became less causal at 9B" would be a pure artifact of the denominator.
**No cross-scale NIE comparison here is valid without the gap beside it.**

**The registered causal prediction failed, and the failure is real rather than an
artifact.** The prediction was that *antipodal* constituents degrade the causal
effect while *orthogonal* ones are preserved. Instead:

- The **antipodal arm is untestable at both scales**, for different structural
  reasons: at 2B `larger_than` had no causal effect on `sp_en_trans` to degrade
  from (cos to the test truth direction ≈ 0.002), and at 9B the pair is no
  longer antipodal at all (that *is* the P3 confirmation). Gemma-2 may simply
  offer no scale where both conditions hold. This is an honest negative about
  experimental availability, not evidence about the mechanism.
- The **orthogonal arm is refuted.** `cities+neg_cities` (cos ≈ −0.08) degrades
  sharply. Because `theta_combo = ½(theta_A + theta_B)` is *shorter* than either
  constituent, this was initially confounded with push magnitude — so the
  intervention was re-run with `theta_combo` rescaled to `‖theta_single‖`,
  holding magnitude fixed so only direction varies:

| `cities+neg` vs `cities` | raw combo | at equal norm | magnitude's share |
|---|---|---|---|
| Δ shift f→t | −0.194 | −0.071 | 64% |
| Δ shift t→f | −0.527 | **−0.482** | **9%** |

At equal push magnitude the combined direction still loses **86% of the
true→false causal effect** while losing only 12% of false→true. Magnitude
explained most of one direction and almost none of the other, and the asymmetry
grows after norm-matching. Meanwhile that same probe *classifies* at 0.986–1.000
cross-topic. So the **classification-vs-causal dissociation M&T flag is
reproduced here — but on orthogonal constituents, which our mechanism predicted
would be safe.**

Why the collapse is specific to the true→false direction is unexplained.
Magnitude is now excluded and readout saturation is weak in log space (PD is a
difference of log probabilities, and logits shift roughly linearly under small
perturbations). It is recorded as an observation, not a mechanism.

**Scope of the norm confound: the causal arm only.** AUROC depends solely on
ranking, and scaling a direction by a positive constant leaves rankings
identical, so magnitude enters nowhere else. Every classification result above —
the transfer sharpening, the bimodality, P1, P2, P3 — is rank- or angle-based
and untouched.

---

## H2 — the induced-falsehood test

**The question.** In the standard setup a statement's truth and the context's
endorsement of it are perfectly confounded: statements are presented neutrally,
so "the statement is true" and "the context treats it as true" always coincide.
H2 decouples them and asks which one the frozen truth direction follows.

**The design.** 100 base city-country facts (`The city of Paris is in France.`),
each paired with a plausible false counterpart, each placed under two instruction
frames — giving a full 2×2 of 400 rows:

| frame | statement | condition |
|---|---|---|
| "Say something true." | true | aligned (control) |
| "Say something true." | false | **conflict** |
| "Say something false." | true | **conflict** |
| "Say something false." | false | aligned (control) |

Aligned cells reproduce the confounded neutral setup. Conflict cells are the
experiment. The model never generates — the prompt is a fixed string and the
residual stream is read over it. **Probes are never retrained**: the H1 `cities`
probes are reloaded from disk, because the whole question is whether the
*pre-existing* direction survives the frame.

### Phase 0 — the `model_knows` filter

Framing the model to negate a fact it does not hold is not inducing a lie, so
each fact is first checked under a neutral, frame-free prompt
(`Is the following statement true or false? {statement} Answer:`), scoring
`logP(true labels) − logP(false labels)` at the final token.

**A readout bug was caught here that would not have looked like one.** After
`Answer:`, Gemma-2-9B wants *title case* — `' True'` 0.238, `' False'` 0.060 —
and the all-caps `" TRUE"`/`" FALSE"` pair holds **1.8%** of the probability
mass. Signs were still 199/200 correct, so the first run looked like weak
knowledge rather than a bug, and the threshold would have been tuned against
noise. The readout is now case-aggregated (`logsumexp` over case variants),
raising label mass to 33%. This is prompt-specific, not model-specific: the NIE
experiment's few-shot `This statement is:` readout genuinely does hold 98.6%.

**Outcome: 98/100 facts survive at 0.5 nats.** The readout separates true from
false *perfectly* (min true margin +0.556 > max false margin +0.291), so the
threshold picks a confidence level rather than rescuing a noisy signal. The
attrition cliff above 0.8 nats is a **TRUE-bias** artifact — the model's neutral
point sits at +0.285 nats, so a symmetric threshold culls almost entirely on the
false side. Dropped: `Wellington-New Zealand` (reads "Wellington is in Australia"
as TRUE — the only sign error in 200 statements) and `Algiers-Algeria`.

### Phase 1 — extraction

The framed prompts run through the same path as H1: `verify_model` first,
`blocks.{L}.hook_resid_post`, last real token. The framed prompt still *ends*
with the statement, so the gathered token is the same end-of-statement position
with a prefix in front of it — verified in the log (`[' is', ' in', ' France',
'.'] -> gathering '.'`). All 42 layers in one forward pass: `(392, 42, 3584)`,
236 MB, 59 seconds. Row identity (`base_fact`, `condition`, `frame`, `label`)
travels *inside* the cache, so row *i* of the tensor is row *i* of the CSV by
construction rather than by a re-read that would shift if Phase 0 were retuned.

Acceptance: shape, finiteness, and **392/392 distinct rows** — an all-identical
tensor is what a broken gather reading pad positions looks like, and it would
pass the first two checks.

### Phase 2 — the canary (a gate)

The frozen probes applied to the **aligned cells only**. If those don't behave
like H1, either extraction is broken in the frame format or the frame shifts the
representation even when it *agrees* — and conflict cells would be
uninterpretable either way.

**AUROC vs truth = 1.0000 for both probes** (H1 in-distribution reference: 0.9997
MM / 1.0000 LR), holding across L17–31. **Gate passed.** Note the gate cannot
distinguish truth-tracking from frame-tracking — in aligned cells `frame == label`
by construction — which is exactly why the conflict cells are the experiment.

### Phase 3 — conflict cells, and why the AUROC is *not* the finding

Both probes score **AUROC 1.0000, CI [1.0000, 1.0000]** against truth. This is
reported for completeness and should not be led with.

Conflict cells compare true-statements-under-a-lie-frame against
false-statements-under-a-truth-frame. Those groups differ in truth **and** frame
at once — `frame == 1 − label` for all 196 rows — so the two are collinear across
exactly those cells, and perfect separation is equally consistent with tracking
truth, tracking frame, or any mixture. `AUROC vs frame = 0.0000` is the
arithmetic complement, not a second result. The ceiling is also why this metric
could not detect the frame effect at all.

### Phase 4 — the decomposition (where the effect is actually estimable)

Over the full 2×2, `label`, `frame` and their product are pairwise independent by
construction (98 per cell), so all three are identified — which they are not
inside the conflict cells. Dependent variable is the continuous signed projection
onto the frozen direction. CIs are 10,000 bootstrap resamples **clustered on base
fact**: the four cells of one fact share a statement and are correlated, so row
resampling would understate every interval.

`score ~ label + frame + label:frame` (0/1 coding, `frame=1` = "Say something true."):

| | MM | 95% CI | LR | 95% CI |
|---|---|---|---|---|
| `label` | **+25.009** | [+24.15, +25.82] | **+18.345** | [+17.71, +18.94] |
| `frame` | −5.811 | [−6.79, −4.80] | −5.682 | [−6.29, −5.05] |
| `label:frame` | **+14.055** | [+13.02, +15.06] | **+12.188** | [+11.50, +12.85] |

| | truth gap, lie-frame | truth gap, truth-frame | attenuation |
|---|---|---|---|
| MM | 25.009 | 39.063 | **36.0%** [34.5%, 37.4%] |
| LR | 18.345 | 30.534 | **39.9%** [38.7%, 41.1%] |

**The frame acts on the gain of the truth signal, not its offset.** Averaged over
label the frame moves the score by just **+1.22 (MM, 4.9% of `b_label`)** and
**+0.41 (LR, 2.2%)**. Almost the entire effect is interaction: framing does not
push the representation toward "true" or "false", it changes how strongly truth
is encoded. The truth-frame polarizes (+8.24 on true statements, −5.81 on false);
the lie-frame depolarizes.

This reproduces the Phase 3 cell means *exactly* — necessarily, since the model
is saturated on a balanced design. That agreement is an arithmetic check, **not**
independent corroboration; what the regression adds is the intervals and the
identification argument.

**Attenuation is asymmetric by truth value — modestly.** Movement toward the
boundary under the lie-frame, in raw (centering-invariant) units:

| | true statements | false statements | asymmetry |
|---|---|---|---|
| MM | 8.244 [8.04, 8.44] — 40.9% | 5.811 [4.80, 6.79] — 30.7% | **+2.433** [+1.42, +3.46] |
| LR | 6.506 [6.28, 6.72] — 42.0% | 5.682 [5.05, 6.29] — 37.7% | **+0.824** [+0.19, +1.46] |

Not a floor/ceiling artifact: the classes start at near-identical distances from
the boundary (20.14 vs 18.92 MM, ratio 1.064; 15.47 vs 15.06 LR, ratio 1.027), and
the asymmetry holds in raw units where no zero point is assumed. **Reported as
secondary**: it is significant in both probes but modest (1.33× MM, 1.11× LR) and
three times larger in MM than LR, and a quantity that probe-dependent is not yet
a stable property of the representation. The 36–40% overall attenuation is the
robust claim.

**One artifact caught and not reported as a finding.** Attenuation appears to
concentrate in the most-extreme statements at ρ = −0.896 — but attenuation is
`score_lie − score_truth` correlated against `score_truth`, one of its own
components, which induces negative correlation even under independence (null
≈ −0.707). Against the *independent* Phase 0 margin the effect survives at
ρ = **−0.318 / −0.371**: real, modest, one sentence rather than a mechanism.

### Pre-registration

`results/h2_prereg_phase3.json`, committed before any conflict-cell score
existed; `h2_02_canary.py` filters to aligned rows at load and deletes the rest,
so this was enforced rather than promised. Five of six predictions confirmed.

**P-H2-3 is the informative one: the prediction was right and the measure was
wrong.** "The frame has a detectable but sub-dominant effect" was operationalized
as `conflict AUROC < aligned AUROC`. AUROC was already at ceiling in both
conditions and had no headroom to detect anything, so the criterion returned "the
frame did nothing measurable" — which is false, as `b_int = +14.06 [13.02, 15.06]`
shows. The refutation is left as recorded rather than rewritten. The lesson: a
pre-registration has to check the *metric* for headroom, not only the *claim*.

The confirmations of P-H2-1/2/4 are all weakened by the same collinearity that
makes the conflict-cell AUROC uninformative. Only Phase 4 separates the effects.

### Limitations, and the interpretive limit

1. **No neutral baseline — the binding one.** Every cell carries a frame, so the
   design identifies the *difference between frames*, not which one moved. "The
   lie-frame compresses 36%" and "the truth-frame amplifies 56%" fit these data
   identically. The defensible statement is the one used throughout: *the truth
   gap is 36–40% smaller under "Say something false." than under "Say something
   true."* Resolvable by extracting the 196 bare statements (~1 min GPU);
   **not yet run.**
2. **Base model.** "Say something false." is context, not an obeyed command.
3. One frame pair, one phrasing, one topic, one model, one headline layer.
4. Probe-dependence of the asymmetry, above.

> **Interpretive limit.** This is informative about **which representation is
> active, not about intent**. A probe tracking actual truth under a "say something
> false" frame is consistent with the model representing the falsehood as false
> while framed to assert it, **and equally consistent with the frame simply
> failing to shift much** — on a base model the latter is a live possibility this
> design cannot exclude. The 36–40% attenuation shows the frame is not inert,
> which is the one thing that can be said cleanly. Probe output alone cannot
> separate these readings. **This is not a deception detector.**

---

## Reproducibility

- Fixed seed (0) for all splits; recorded in every artifact.
- `src/config.py` is the single source of model choice and derives every layer
  from the committed sweep. `TRUTH_PROBES_MODEL=gemma-2-2b python scripts/...`
  re-runs any analysis against the other model.
- Results and activation caches are **scoped by model** (`results/<model>/`).
  Before this, the 9B run would have silently overwritten every 2B number the
  write-up cites — and a stale 2B cache would have satisfied the "already
  extracted" check, computing 9B results from 2304-dim 2B activations.
- Activation caches (`results/activations/`) and probes are gitignored;
  regenerate with `cluster/04_extract.sbatch` + `05_sweep.sbatch` (~7 min), and
  `cluster/09_h2_extract.sbatch` (~1 min) for the H2 cache.
- **H2 uses the H1 probes frozen off disk and never retrains.** Every H2 script
  asserts the reloaded probe's `model_name` and `layer` against `config`, so a
  stale or wrong-layer probe fails loudly instead of producing a plausible
  number. Phases 2–5 never load the model at all; if one needs to, the
  no-retraining rule has been broken.
- H2 bootstraps resample **base facts, not rows** — the four cells of one fact
  share a statement and are correlated, so row resampling would understate every
  interval.
- Cluster runbook in [`cluster/README.md`](cluster/README.md). Note that
  TransformerLens upcasts the state dict to fp32 during weight processing, so
  `dtype=bfloat16` does **not** bound peak memory: 9B needs ≥176G host RAM
  (measured peak 151 GiB), and VRAM was never the constraint.

## Repository layout

```
src/         config.py (model + layer resolution), activations.py, probes.py,
             data.py, lastping.py
scripts/     H1:  01_extract_activations, 02_train_probes, 03_transfer_eval,
                  03b_combined_negation, then aux_* analyses
             H2:  h2_00_model_knows, h2_01_extract, h2_02_canary,
                  h2_03_conflict, h2_04_decompose
cluster/     SLURM runbook + sbatch scripts (01-07 H1, 08-09 H2)
data/        Marks & Tegmark CSVs + induced_lies.csv (H2, hand-built)
results/     <model>/*.json findings, eval_9b.md, eval_h2.md,
             prereg_9b_prediction.json, h2_prereg_phase3.json
```

H2 reruns in order: `08_h2_model_knows.sbatch` (GPU) →
`h2_00_model_knows.py --phase filter` (CPU, retunes the threshold without
re-queuing the GPU job) → `09_h2_extract.sbatch` (GPU) → phases 2–4 all CPU,
seconds each, since they only apply frozen probes to the cached activations.

## Next steps

### The no-frame baseline (H2's binding gap, and the cheapest thing here)

Every H2 cell carries a frame, so the design identifies the *difference between*
frames, not which frame moved. Extracting the 196 bare statements — the same
sentences with no instruction prefix — places a neutral baseline against both and
settles whether the lie-frame compresses, the truth-frame amplifies, or both. It
is a ~1-minute GPU job reusing `h2_01_extract.py`'s path, and it is the one
result that would change how H2's headline is phrased. Not run.

### H2 extensions, in rough order of value

- **Phase 5 (layer robustness).** Phase 3 checked conflict-cell AUROC across the
  L17–31 plateau, but the *attenuation* was only estimated at L17. All 42 layers
  are already cached, so re-running Phase 4's decomposition per layer is CPU-only
  and free — it would say whether the frame's gain effect is localized or global.
- **A second frame phrasing.** One frame pair is one phrasing; "Lie about the
  following." or a few-shot deceptive context would separate the effect from the
  particular wording.
- **An instruction-tuned model.** The base-model caveat is load-bearing precisely
  because `gemma-2-9b` has no reason to obey "Say something false." Running the
  identical design on `gemma-2-9b-it` is the clean way to ask whether an obeyed
  instruction moves the representation differently — and it is the only version
  of this experiment where "the model is complying" is a defensible premise.
- **The asymmetry, on more data.** True statements attenuate more than false, but
  3× more strongly in MM than LR. More facts or more topics would say whether
  that is real structure or probe idiosyncrasy.

### Smaller open threads

- **The 2B score decomposition** was not recomputed: 2B activations were
  generated on the laptop and are gitignored, so the 2B half of the P1 analysis
  is inferred from recorded outcomes rather than measured. Re-extracting 2B is
  cheap and would close it.
- **A corrected P1 predictor** combining direction and relative magnitude —
  needs new cells to be a real test, not the twelve that refuted the old one.
- **Re-seeding robustness**: dedupe the two translation sets first (see the
  leakage caveat above).

## References

- Marks & Tegmark (2024), *The Geometry of Truth* (arXiv:2310.06824, COLM 2024).
- Burns et al. (2023), *Discovering Latent Knowledge Without Supervision*.
- Azaria & Mitchell (2023); Li et al. (2023) — truth probing.
- The deception-evaluation literature on the belief-vs-intent problem.
