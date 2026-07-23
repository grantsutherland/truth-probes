# Linear truth probes and their behavior under induced falsehood

An interpretability experiment on **Gemma-2** (2B for development, 9B for final
numbers). It reproduces the core result
that LLMs linearly represent the truth of simple factual statements (Marks &
Tegmark, 2024), compares two probe types (logistic regression and mass-mean), and
then extends the setup with an original test: **does a probe trained on passively
presented true/false statements still track truth when the model is prompted to
assert a falsehood?**

> ### Scope and the belief-vs-intent caveat, up front
> This studies whether a linear "truth direction" exists in the model's
> activations and how it behaves when the model states something false. It does
> **not** claim to detect "lying" or "deception" in a strong sense. Whether a
> probe firing on a model-asserted falsehood means the model *knows* it is false
> (Workaround 1 of Goldowsky-Dill et al. / the deception-evaluation literature) or
> means the prompt has *shifted what the model represents as true* is not
> resolvable from this setup alone. Results are reported as "the probe tracks
> actual truth vs. asserted position," with that ambiguity stated, not as a
> deception detector.

## Background and hypotheses

Prior work (Azaria & Mitchell 2023; Burns et al. 2023; Marks & Tegmark 2024) finds
that the truth of a simple factual statement is often linearly decodable from a
transformer's residual stream. Marks & Tegmark show that a plain
difference-in-means ("mass-mean", MM) direction is about as accurate a classifier
as logistic regression (LR) at scale, while being *more causally implicated* in
the model's outputs (interventions along the MM direction change the model's
true/false judgments more effectively than along the LR direction).

This experiment tests two things:

- **H1 (reproduction).** LR and MM probes trained on clean true/false statements
  separate true from false, and transfer across topically and structurally
  different datasets, indicating a general truth direction rather than a
  dataset-specific artifact.
- **H2 (extension, the actual contribution).** Under prompts that induce the model
  to assert a false statement, LR and MM probes may diverge. Because the MM
  direction is more causally tied to the model's own truth computation, it may
  continue to track the statement's *actual* truth value even as the model asserts
  the opposite, whereas LR (optimized for classification, and more contaminated by
  correlated features) may behave differently. Whether they diverge is the finding;
  either outcome is reportable.

## Method

### Model and tooling
- **Model:** Gemma-2 (Google), loaded via **TransformerLens**, which handles
  residual-stream activation caching. Development and debugging run on
  **Gemma-2-2B** (fast, cheap iteration; it reliably knows the simple facts);
  final numbers run on **Gemma-2-9B** (a cleaner, more defensible truth
  representation). Running the small model first means the slow model only runs
  once the pipeline is known to work, and reporting both mirrors Marks &
  Tegmark's finding that the effect sharpens with scale. Both fit on a single
  GPU. (Note: Gemma-2 requires accepting the license and a HuggingFace token to
  download the weights.)
- **Probes:** both implemented as small `torch.nn.Module`s over a shared base
  (`src/probes.py`). LR is a single linear layer trained with Adam on
  `BCEWithLogitsLoss`; MM is a direct difference-in-means computation
  (`theta = mean(true_acts) - mean(false_acts)`), with the optional IID
  covariance correction (`theta @ inv(Sigma)`, equivalent to LDA) noted where used.
- **Stack:** TransformerLens + PyTorch + numpy. No training of the model, no
  heavy infrastructure.

### Data
- **Base true/false statements:** the curated datasets released by Marks & Tegmark
  (`saprmarks/geometry-of-truth`) — e.g. `cities`, `neg_cities`, `sp_en_trans`,
  `larger_than`, `smaller_than`, and their negations/conjunctions. These are
  simple, unambiguous facts the model reliably knows, which separates the
  interpretability question from a capability question.
- **Induced-falsehood prompts (this project's contribution):** a small,
  hand-constructed set placing known-true and known-false statements in a context
  that instructs or leads the model to assert the statement (regardless of its
  truth). Held clearly separate from the base data.

### Activation extraction
1. Run the model (Gemma-2-2B or -9B) over each statement via TransformerLens.
2. Cache the residual-stream activation at a **middle-to-late layer**, over the
   **end-of-statement token** (following the localization in Marks & Tegmark, where
   statement-level truth information is summarized over the final/punctuation token).
3. Sweep layers to select the one where the truth direction is cleanest on a
   validation split; record the chosen layer.
4. **Cache activations to disk once**, then run all probing/analysis off the cache
   (forward passes are the slow step; probes are instant).

### Probing and transfer (tests H1)
1. Train LR and MM probes on an 80% split of a base dataset (e.g. `cities`).
2. Evaluate in-distribution accuracy on the held-out 20%.
3. Evaluate **transfer**: accuracy on the *other* datasets (never seen in training).
   High transfer is the evidence that the probes found a general truth direction,
   not a surface artifact.
4. Repeat training on statements-plus-negations (e.g. `cities + neg_cities`) to
   check the known result that including negations improves generalization on
   negated statements.
5. Report a small table: probe type x train set x test set accuracy.

### Induced-falsehood test (tests H2, the contribution)
1. Using the probes from above (no retraining), take the induced-falsehood prompts.
2. For each item, extract the activation at the same layer/token position while the
   model is asserting the statement.
3. Record, for both LR and MM, whether the probe's output tracks the statement's
   **actual truth value** or the model's **asserted position**.
4. Compare LR vs MM behavior. Report the split explicitly.
5. **Interpret with the caveat above:** a probe tracking actual truth is consistent
   with the model "knowing" the falsehood; a probe tracking the asserted position is
   consistent with the prompt having shifted the model's representation. This setup
   cannot fully distinguish the two, and the write-up says so.

## Reproducibility
- Fixed random seed for all train/test splits (recorded).
- One config block (model name, layer, token position, split ratio) at the top of
  each script; no scattered magic numbers.
- All reported numbers written to `results/` as JSON/CSV, so every figure in this
  README is backed by a saved artifact.
- Base datasets are Marks & Tegmark's public release; the induced-falsehood prompt
  set is included in `data/`.

## Repository layout
```
src/         activations.py (TransformerLens caching), probes.py (LR + MM),
             data.py (dataset loading), experiments.py (train/transfer/induced-lie)
scripts/     01_extract_activations, 02_train_probes, 03_transfer_eval,
             04_induced_lie_test  (run in order)
data/        Marks & Tegmark datasets + induced_lies.csv (this project's data)
results/     cached activations, probe accuracies, plots
notebooks/   exploration + figures
```

## Results
*(To be filled in once the experiment is run. Will report: chosen layer; LR and MM
in-distribution and transfer accuracies; and the induced-falsehood outcome for each
probe type, with the belief-vs-intent interpretation stated explicitly. Hypothesis
language above will be revised to reflect what was actually found — including if H2
shows no divergence, which is itself a valid result.)*

## References
- Marks & Tegmark (2024), *The Geometry of Truth*.
- Burns et al. (2023), *Discovering Latent Knowledge Without Supervision* (CCS).
- Azaria & Mitchell (2023); Li et al. (2023) — truth probing.
- The deception-evaluation literature on the belief-vs-intent problem (Workaround 1:
  measuring known-falsehood rather than deceptive intent).