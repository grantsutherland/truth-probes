# Linear truth directions in Gemma-2

Probing whether Gemma-2 linearly represents the truth of simple factual
statements, reproducing and extending Marks & Tegmark's *Geometry of Truth*
with two probe types (logistic regression and mass-mean) on residual-stream
activations. Run at two scales, 2B for development and 9B for the reported
results.

## What this found

A general truth direction exists and sharpens sharply with scale. Cross-topic
transfer (train on one dataset, test on an unrelated one) rises from signed
AUROC 0.84/0.88 at 2B to 0.99 at 9B for MM/LR. A probe trained on
city-country facts ranks true Spanish-English translations above false ones
almost perfectly, so the direction is not topic-specific.

Combined-probe training has an averaging failure mode that explains a
dissociation Marks & Tegmark flag but leave open. For balanced concatenation,
the mass-mean direction is exactly the average of its two constituent
directions, so the combined score on any point is the average of constituent
projections. When the constituents are antipodal the average nearly cancels
and leaves a non-causal residual; when orthogonal it only rotates. This fits
all three of their Table 2 MM rows, including the flip that appears with
scale.

A deceptive-frame prefix attenuates the truth signal without inverting it.
Prompting the model to "say something false" compresses the linear truth
signal but does not reverse it, and the compression is uniform across true
and false statements. This is a 9B, base-model result, and it is not a
deception detector (see the limits below).

## Setup

Statements come from the six Marks & Tegmark datasets (cities, neg_cities,
sp_en_trans, neg_sp_en_trans, larger_than, smaller_than). Probes are trained
on residual-stream activations at the end-of-statement token. The chosen layer
is selected by a hyperparameter-free mass-mean sweep averaged over all six
datasets, then reused for both probes: layer 13 of 26 at 2B, layer 17 of 42
at 9B.

Two probe types. Logistic regression optimizes class separation directly.
Mass-mean uses the difference of class means, which is a worse classifier but
a more causally implicated direction, following the paper. Data loading,
activation caching, seeded splits, and a model-scoped cache format are in
`src/`; the pipeline runs as numbered scripts under `scripts/`.

## H1: does a truth direction transfer

In-distribution accuracy saturates near 1.0 for both probes, so it cannot
distinguish them and is not the interesting quantity. Transfer is.

Raw transfer accuracy first looked like a collapse to chance. That was a
threshold artifact: the decision boundary is fit to the training
distribution's projection scale and does not carry across datasets. AUROC,
which is threshold-free, shows the direction generalizes, and per-dataset
mean-centering (label-free, the paper's protocol) recovers accuracy. AUROC
diagnosed it, centering fixed it, and the two agree.

Transfer is bimodal. Cells either transfer cleanly (correct sign, no oracle)
or invert cleanly, and the inversions track two known causes: negation, and
logical complementarity between larger_than and smaller_than. Excluding the
complement pairs by construction, cross-topic transfer is 0.84/0.88 at 2B and
0.99 at 9B.

Negation rotates the truth direction with depth, from antipodal in early
layers to orthogonal in later ones, matching the paper's Appendix C. An
apparent difference between factual and translation negation at one layer did
not survive a full-depth sweep: both dataset pairs rotate the same way, just
at different rates. Combined training on a dataset plus its negation improves
cross-topic generalization, reproducing their page-7 result.

## The averaging mechanism

Combined mass-mean training gives theta_combo = (theta_A + theta_B) / 2
exactly, so the combined probe's score is the mean of the two constituent
scores. In-distribution is protected because the native constituent dominates.
Cross-topic transfer fails by sign-cancellation when the constituents project
onto a third dataset with opposite signs.

An earlier sharper version of this ("antipodal constituents collapse
in-distribution too") was tested and falsified at 2B, because the datasets are
not perfectly antipodal and rank-based AUROC saturates. The surviving claim is
the exact identity plus its empirical consequence, kept verbally separate,
since AUROC is a rank statistic and monotonicity does not follow from the
algebra.

This is a candidate answer to the dissociation in their Section 6: combined
training helps cities but hurts larger_than causally. With their reported
constituent geometry (larger/smaller antipodal at 13B, aligned at 70B;
negations approximately orthogonal), the averaging identity predicts the sign
of every combined-MM NIE change in Table 2.

## The causal arm, and where it dead-ends

Testing the mechanism causally needs a scale where the constituents are
antipodal and cross-dataset causal transfer exists at the same time. Neither
model provides it. At 2B, larger_than has no causal signal for sp_en_trans to
begin with (NIE 0.03, cosine 0.002 to the truth direction), so there is
nothing to degrade. At 9B the constituents are no longer antipodal, so the
mechanism predicts no degradation. The antipodal prediction is therefore
untestable in Gemma-2, and the write-up says so rather than reporting the null
as confirmation.

One methodological point falls out. NIE is not comparable across models
without the baseline gap, because the gap grew 8x from 2B to 9B and swamps
every numerator. Every absolute intervention effect grew with scale while
every normalized NIE fell. The paper's own cross-scale NIE comparisons do not
report the gap, so a reader cannot separate the direction from the readout's
confidence.

## H2: the truth signal under a deceptive frame

A 2x2 over 98 city facts that the model verifiably knows: frame (say-true /
say-false) crossed with statement truth (true / false), with the frozen H1
cities probe applied to the framed prompts. Statements use the exact M&T
template so the frame prefix is the only novel element.

Conflict-cell AUROC is at ceiling and carries no frame information, because
truth and asserted position are collinear across those two cells. The finding
is in the regression score ~ label + frame + label:frame over the full design,
where the three targets are pairwise independent and estimable. The frame
attenuates the truth signal by a large fraction without inverting it, and the
attenuation is uniform across true and false statements (the per-cell
compressions overlap in CI). Bootstrap is clustered by base fact, since the
four cells of a fact are not independent.

Limits, which are load-bearing here. Gemma-2-9b is a base model, so "say
something false" is context, not an obeyed command. A probe tracking truth
under a deceptive frame is consistent with the model representing the
falsehood as false while framed to assert it, and equally consistent with the
frame largely failing to move the representation. The result shows the frame
is not inert (it compresses the signal by a large fraction) but it is not
evidence of deception, and this is not a deception detector.

## Pre-registered predictions

Predictions were logged in `results/prereg_9b_prediction.json` before any 9B
forward pass. P3 (larger/smaller geometry aligning with scale) confirmed out
of sample: cosine went from -0.62 at 2B to +0.46 at 9B, rising with depth,
reproducing the paper's 13B-to-70B trend across a 2B-to-9B gap; cities and its
negation stayed orthogonal as predicted. P1 was refuted at its own
falsification criterion, because the combined score is norm-weighted and
cosine is blind to norm. The refutations are reported as refutations.

## Reproducing

Datasets are the public geometry-of-truth release and are not redistributed
here. Probes are scikit-style and trivially cheap; the cost is activation
extraction, which needs a GPU. `requirements-cuda.txt` pins the CUDA build.
Chosen layers, thresholds, and per-run numbers are written to `results/` so
every figure is backed by an artifact. Infrastructure notes (host-RAM
requirements at 9B, the TransformerLens fp32 load) are in `NOTES.md`.