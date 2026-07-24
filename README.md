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
- **H2 (induced falsehood).** Not run. Still the natural next step; see
  [Next steps](#next-steps).

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
`larger_than` (1980), `smaller_than` (1980). All balanced 50/50.

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
  regenerate with `cluster/04_extract.sbatch` + `05_sweep.sbatch` (~7 min).
- Cluster runbook in [`cluster/README.md`](cluster/README.md). Note that
  TransformerLens upcasts the state dict to fp32 during weight processing, so
  `dtype=bfloat16` does **not** bound peak memory: 9B needs ≥176G host RAM
  (measured peak 151 GiB), and VRAM was never the constraint.

## Repository layout

```
src/         config.py (model + layer resolution), activations.py, probes.py,
             data.py, lastping.py
scripts/     01_extract_activations, 02_train_probes, 03_transfer_eval,
             03b_combined_negation, then aux_* analyses
cluster/     SLURM runbook + sbatch scripts for the 9B run
data/        Marks & Tegmark CSVs
results/     <model>/*.json findings, eval_9b.md, prereg_9b_prediction.json
```

## Next steps

### H2 — the induced-falsehood test (the original planned contribution)

**Not run, and still the natural next step.** The question: *does a probe
trained on passively presented true/false statements still track truth when the
model is prompted to assert a falsehood?*

> **The belief-vs-intent caveat, up front.** This would not detect "lying" or
> "deception" in any strong sense. Whether a probe firing on a model-asserted
> falsehood means the model *knows* the statement is false, or means the prompt
> has *shifted what the model represents as true*, is not resolvable from this
> setup. Results must be reported as "the probe tracks actual truth vs. asserted
> position," with that ambiguity stated — never as a deception detector.

The hard part is `data/induced_lies.csv`, which does not exist and **must be
hand-built**: prompts leading the model to assert a statement regardless of its
truth, with columns for `statement`, `actual_truth`, `asserted_position`,
`prompt`, `source_dataset`. It is by far the most contestable artifact in the
project and should be designed unhurried. Then `scripts/04_induced_lie_test.py`
would reuse the probes saved by `02` (no retraining), extract activations at the
same layer/token while the model asserts each statement, and record signed
margins — not hard labels — so magnitude collapse stays visible.

Three outcomes are all publishable: the probe tracks actual truth, it tracks the
asserted position, or its margin collapses toward zero. Frame the result against
H1's MM-vs-LR split (`cos(mm, lr)` ≈ 0.77 at 2B, 0.77–0.95 at 9B) rather than as
a standalone deception probe.

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
