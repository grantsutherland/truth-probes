"""H2 PHASE 0 — the model_knows filter.

Before anything is framed, establish which base facts Gemma-2-9B reliably knows
under NEUTRAL questioning. The point is definitional, not hygienic: framing the
model to assert the negation of a fact it does not hold is not inducing a lie,
it is asking an open question, and such a fact would contribute noise to the
conflict cells while looking exactly like a real trial.

Readout only — no probe, no frame, no generation. For each of the 200 unique
statements, score

    margin = logP(" TRUE") - logP(" FALSE")

at the final token of

    "Is the following statement true or false? {statement} Answer:"

A base fact passes iff its TRUE statement reads TRUE and its FALSE statement
reads FALSE, both by at least MARGIN_THRESHOLD.

TWO PHASES, deliberately split (same idiom as 01_extract_activations):

    --phase score   GPU. 200 forward passes; writes per-statement margins.
    --phase filter  CPU. Applies the threshold, writes model_knows into the CSV.

The threshold is meant to be "set from the distribution", which is a decision you
cannot make before seeing the distribution. Splitting the phases means that
decision — and any later retune — costs a CPU second rather than another
9B GPU job. The margins file is the source of truth; the CSV's model_knows column
is derived and can always be regenerated from it.

THE TOKENIZATION CHECK IS NOT OPTIONAL, AND IT ALREADY PAID FOR ITSELF. The
leading space matters — the prompt ends at "Answer:", so the continuation carries
one. But CASE matters just as much, and the first run of this script found that
it does not carry over from the rest of the project. After "This statement is:"
(the NIE readout) Gemma-2-9B puts 98.6% of its mass on " TRUE"/" FALSE". After
"Answer:" it wants title case: ' True' 0.238, ' False' 0.060, with the ALL-CAPS
pair holding just 1.8% between them. Scoring the all-caps ids alone would have
measured a continuation the model essentially never produces — the signs still
came out 199/200 correct, so it would not have looked broken, it would have
looked like weak knowledge, and the margin threshold would then have been tuned
against noise.

So the readout is CASE-AGGREGATED: logsumexp over {" TRUE", " True", " true"}
against the same for false. This keeps the plan's neutral prompt verbatim while
scoring what the model actually emits.

Because the readout turned out to be prompt-sensitive, the score phase records a
SECOND readout — the H1/NIE-style "{statement} This statement is:" — over the same
statements in the same job (the model is already loaded; it costs one more pass).
If the two prompts agree on which facts pass, model_knows is a property of the
model rather than of a prompt choice, which is worth knowing before 100 facts get
excluded on the strength of one template.
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                                            # noqa: E402
import data as dataio                                    # noqa: E402
from activations import load_model, verify_model         # noqa: E402

DATASET_CSV = os.path.join(dataio.DATA_DIR, "induced_lies.csv")

# Per-statement readouts from the GPU phase. Kept separate from the summary so a
# retune reads this and never touches the model.
MARGINS_JSON = os.path.join(config.RESULTS_DIR, "h2_model_knows_margins.json")
# The filter decision: threshold, attrition ladder, surviving fact set.
SUMMARY_JSON = os.path.join(config.RESULTS_DIR, "h2_model_knows.json")

# Neutral, frame-free. No few-shot exemplars: they would import their own truth
# framing into a filter whose whole job is to be unframed.
#
# PRIMARY is the plan's prompt. ALT is the readout the rest of the project uses
# (aux_nie_intervention), carried here as a cross-check because the first run
# showed the label distribution is prompt-sensitive: the same model, asked the
# same thing two ways, puts its mass on different casings.
PROMPTS = {
    "primary": "Is the following statement true or false? {statement} Answer:",
    "alt": "{statement} This statement is:",
}

# Case variants, aggregated by logsumexp. See the module docstring: after
# "Answer:" the all-caps pair holds 1.8% of the mass and title case holds ~30%,
# so scoring ALL-CAPS alone would score a continuation the model never emits.
LABEL_FORMS = {
    "true": [" TRUE", " True", " true"],
    "false": [" FALSE", " False", " false"],
}

# Minimum |margin|, in nats, for a statement to count as confidently read.
# SET FROM THE OBSERVED 9B DISTRIBUTION — see the ladder the filter phase prints
# and the recorded rationale in results/<model>/h2_model_knows.json. A margin of
# 1.0 nat is roughly 73/27 between the two labels.
MARGIN_THRESHOLD = 1.0

# Two prompts whose tokenization is printed in full before any scoring.
_TOKENIZATION_PROBES = 2


def label_ids(model):
    """{"true": [ids...], "false": [ids...]} for the case variants in
    LABEL_FORMS, asserting each surface form is a single token.

    The single-token assertion is lifted from aux_nie_intervention (importing it
    would build a whole NIE experiment at module scope). It matters for the same
    reason there: a multi-token label silently scores its first fragment, which
    reads as a weak-but-plausible signal rather than as an error.
    """
    out, seen = {}, {}
    for side, forms in LABEL_FORMS.items():
        ids = []
        for s in forms:
            toks = model.to_tokens(s, prepend_bos=False)[0]
            if len(toks) != 1:
                pieces = [model.to_string(t.unsqueeze(0)) for t in toks]
                raise ValueError(
                    f"{s!r} tokenizes to {len(toks)} tokens {pieces} under "
                    f"{model.cfg.model_name}, not 1. The margin readout assumes "
                    f"single-token continuations; every model_knows decision "
                    f"would be computed against the wrong logit."
                )
            tid = toks[0].item()
            # A form appearing on both sides would make the margin partly a
            # difference of a quantity with itself. Cheap to check, impossible
            # to spot in the output.
            if tid in seen:
                raise ValueError(
                    f"{s!r} and {seen[tid]!r} share token id {tid}.")
            seen[tid] = s
            ids.append(tid)
        out[side] = ids
    return out


def _margin(logp, ids):
    """logsumexp over the true-label ids minus the same over the false ones."""
    lp_t = torch.logsumexp(logp[ids["true"]], dim=0)
    lp_f = torch.logsumexp(logp[ids["false"]], dim=0)
    return lp_t.item(), lp_f.item()


def load_rows():
    """The induced-lie CSV, with `frame` kept as the strings 'true'/'false'.

    Without dtype=str + keep_default_na=False pandas reads that column as
    booleans, which then compare unequal to the 'true'/'false' the `condition`
    column was derived from — a mismatch that would silently mislabel cells
    downstream.
    """
    df = pd.read_csv(DATASET_CSV, dtype={"frame": str, "model_knows": str},
                     keep_default_na=False)
    expected = {"base_fact", "city", "statement", "label", "frame",
                "condition", "prompt", "model_knows"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{DATASET_CSV} missing column(s) {sorted(missing)}.")
    if set(df["frame"].unique()) != {"true", "false"}:
        raise ValueError(f"unexpected frame values {df['frame'].unique()}")
    return df


# --------------------------------------------------------------------------- #
# Phase: score  (GPU)
# --------------------------------------------------------------------------- #
def phase_score():
    df = load_rows()
    # 400 rows = 200 unique statements x 2 frames. The filter is frame-free, so
    # score each statement once and let both its rows inherit the result.
    stmts = (df[["base_fact", "city", "statement", "label"]]
             .drop_duplicates(subset="statement")
             .sort_values(["base_fact", "label"], ascending=[True, False])
             .reset_index(drop=True))
    print(f"{len(df)} rows -> {len(stmts)} unique statements "
          f"across {stmts.base_fact.nunique()} base facts")

    model = load_model(config.MODEL_NAME)
    print("verify_model:", json.dumps(verify_model(model), indent=2, default=str))

    ids = label_ids(model)
    print("\nlabel token ids:")
    for side, forms in LABEL_FORMS.items():
        print(f"  {side:5} " + "  ".join(f"{s!r}={i}"
                                         for s, i in zip(forms, ids[side])))

    # What does the model ACTUALLY want to say? Printed per prompt template,
    # because the answer differs between them — which is the whole reason the
    # readout is case-aggregated and the alt template is carried at all.
    for name, template in PROMPTS.items():
        print(f"\n--- readout check: {name} ---")
        for stmt in stmts["statement"].head(_TOKENIZATION_PROBES):
            prompt = template.format(statement=stmt)
            with torch.no_grad():
                logits = model(model.to_tokens(prompt))
            probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
            top = torch.topk(probs, 5)
            shown = " | ".join(f"{model.to_string(i.unsqueeze(0))!r} {p:.3f}"
                               for p, i in zip(top.values, top.indices))
            caps = (probs[ids["true"][0]] + probs[ids["false"][0]]).item()
            agg = (probs[ids["true"]].sum() + probs[ids["false"]].sum()).item()
            print(f"  prompt: {prompt!r}\n  top-5: {shown}"
                  f"\n  mass: ALL-CAPS only {caps:.3f}  case-aggregated {agg:.3f}")

    records = []
    with torch.no_grad():
        for i, row in stmts.iterrows():
            rec = {
                "base_fact": row["base_fact"],
                "city": row["city"],
                "statement": row["statement"],
                "label": int(row["label"]),
            }
            for name, template in PROMPTS.items():
                logits = model(model.to_tokens(
                    template.format(statement=row["statement"])))
                logp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
                lp_t, lp_f = _margin(logp, ids)
                # Signed: positive = reads TRUE. A correct read is a POSITIVE
                # margin on a true statement and a NEGATIVE one on a false
                # statement, so sign carries the direction and |margin| the
                # confidence. Keep them separate; never store |margin| alone.
                rec[f"{name}_logp_true"] = lp_t
                rec[f"{name}_logp_false"] = lp_f
                rec[f"{name}_margin"] = lp_t - lp_f
                rec[f"{name}_label_mass"] = float(
                    torch.exp(logp[ids["true"]]).sum()
                    + torch.exp(logp[ids["false"]]).sum())
            # The primary readout is what model_knows is decided on; `margin`
            # is the key every downstream consumer reads.
            rec["margin"] = rec["primary_margin"]
            rec["label_mass"] = rec["primary_label_mass"]
            records.append(rec)
            if (i + 1) % 50 == 0:
                print(f"  scored {i + 1}/{len(stmts)}")

    payload = {
        "model": config.MODEL_NAME,
        "prompt_templates": PROMPTS,
        "primary_readout": "primary",
        "label_forms": LABEL_FORMS,
        "label_token_ids": ids,
        "aggregation": "logsumexp over case variants",
        "n_statements": len(records),
        "statements": records,
    }
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(MARGINS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved -> {MARGINS_JSON}")

    _describe(records)


def _describe(records):
    """Distribution diagnostics — the acceptance criterion is that the true and
    false margins are clearly BIMODAL, not everything hovering near zero."""

    def q(x, p):
        return torch.quantile(x, torch.tensor(p)).tolist()

    for name in PROMPTS:
        key = f"{name}_margin"
        if key not in records[0]:
            continue
        t = torch.tensor([r[key] for r in records if r["label"] == 1])
        f = torch.tensor([r[key] for r in records if r["label"] == 0])
        mass = torch.tensor([r[f"{name}_label_mass"] for r in records])
        tag = " (PRIMARY — model_knows is decided on this)" if name == "primary" else ""
        print(f"\nMARGIN DISTRIBUTION [{name}]{tag}"
              f"\n  {PROMPTS[name]!r}"
              f"\n  logP(true labels) - logP(false labels), case-aggregated; "
              f"correct read = positive on true statements, negative on false")
        for lbl, x in (("true  statements", t), ("false statements", f)):
            p5, p25, p50, p75, p95 = q(x, [0.05, 0.25, 0.5, 0.75, 0.95])
            print(f"  {lbl}  n={len(x):3d}  mean {x.mean():+.2f}  "
                  f"p5 {p5:+.2f}  p25 {p25:+.2f}  median {p50:+.2f}  "
                  f"p75 {p75:+.2f}  p95 {p95:+.2f}")
        print(f"  correct sign: true {int((t > 0).sum())}/{len(t)}  "
              f"false {int((f < 0).sum())}/{len(f)}")
        # Below ~0.10 the readout is weak and margins are a proxy for a
        # continuation the model rarely emits — the condition that made the
        # all-caps-only first run untrustworthy.
        print(f"  label probability mass: mean {mass.mean():.3f}  "
              f"min {mass.min():.3f}"
              f"{'   <-- WEAK READOUT' if mass.mean() < 0.10 else ''}")

    # Separation: does EVERY true statement outscore EVERY false one? If so the
    # readout is not merely bimodal, it is perfectly ordered, and the threshold
    # is choosing a confidence level rather than rescuing a noisy signal.
    t = torch.tensor([r["primary_margin"] for r in records if r["label"] == 1])
    f = torch.tensor([r["primary_margin"] for r in records if r["label"] == 0])
    gap = (t.min() - f.max()).item()
    print(f"\nSEPARATION [primary]: min(true) {t.min():+.3f}  "
          f"max(false) {f.max():+.3f}  gap {gap:+.3f}  "
          f"({'PERFECTLY SEPARATED' if gap > 0 else 'OVERLAPPING'})")
    # The neutral point is NOT zero: the model has a prior toward TRUE, so a
    # symmetric threshold in raw margin cuts harder on the false side than the
    # true one. Reported because it is the reason false margins are smaller, and
    # a reader would otherwise read that as "false statements are less known".
    bias = ((t.mean() + f.mean()) / 2).item()
    print(f"  TRUE-bias offset (midpoint of class means): {bias:+.3f} nats — "
          f"a symmetric threshold is {'' if abs(bias) < 0.05 else 'NOT '}"
          f"neutral between the two sides")

    _readout_agreement(records)
    print("\nRun --phase filter to pick a threshold against this.")


def _readout_agreement(records):
    """Do the two prompts agree about which facts the model knows?

    If they do, model_knows is a property of the model. If they disagree, it is
    partly a property of the template, and excluding facts on the strength of
    one prompt is a choice that has to be defended rather than assumed.
    """
    if f"alt_margin" not in records[0]:
        return
    p = torch.tensor([r["primary_margin"] for r in records])
    a = torch.tensor([r["alt_margin"] for r in records])
    lab = torch.tensor([r["label"] for r in records])
    # Correct-sign agreement, which is what the filter actually consumes.
    ok_p = torch.where(lab == 1, p > 0, p < 0)
    ok_a = torch.where(lab == 1, a > 0, a < 0)
    both = int((ok_p & ok_a).sum())
    corr = float(torch.corrcoef(torch.stack([p, a]))[0, 1])
    print(f"\nREADOUT AGREEMENT (primary vs alt, {len(records)} statements)"
          f"\n  correct sign under both: {both}  "
          f"primary only: {int((ok_p & ~ok_a).sum())}  "
          f"alt only: {int((ok_a & ~ok_p).sum())}  "
          f"neither: {int((~ok_p & ~ok_a).sum())}"
          f"\n  Pearson r between margins: {corr:+.3f}")

    # Diagnose an alt readout that is merely OFFSET from the primary rather than
    # uninformative. Bare "This statement is:" does not offer the model a binary
    # choice, so it defaults toward TRUE and toward non-label continuations
    # (' a', '\n\n'); the RANKING can still be intact under a large constant
    # shift. Saying which of those two failures occurred matters: an offset
    # readout corroborates the primary's ordering, an uninformative one does not.
    n_alt_wrong = int((~ok_a).sum())
    if n_alt_wrong > 0.4 * len(records):
        shift = (a - p).mean().item()
        print(f"  ALT READOUT UNUSABLE AS AN ABSOLUTE CRITERION: "
              f"{n_alt_wrong}/{len(records)} wrong-signed, mean shift vs "
              f"primary {shift:+.3f} nats."
              f"\n  r={corr:+.3f} says the ORDERING survives the prompt change "
              f"while the CALIBRATION does not — so this corroborates the "
              f"primary's ranking and cannot corroborate its zero point.")


# --------------------------------------------------------------------------- #
# Phase: filter  (CPU)
# --------------------------------------------------------------------------- #
def phase_filter(threshold):
    if not os.path.exists(MARGINS_JSON):
        raise FileNotFoundError(
            f"No margins at {MARGINS_JSON}. Run --phase score (GPU) first.")
    with open(MARGINS_JSON) as f:
        payload = json.load(f)
    if payload["model"] != config.MODEL_NAME:
        raise ValueError(
            f"{MARGINS_JSON} was scored with {payload['model']!r} but "
            f"config.MODEL_NAME is {config.MODEL_NAME!r}.")
    # Fail on a stale schema HERE, before anything is written. The first version
    # of this file scored ALL-CAPS labels only; consuming it would apply a filter
    # built on a readout carrying 1.8% of the probability mass.
    if "prompt_templates" not in payload:
        raise ValueError(
            f"{MARGINS_JSON} predates the case-aggregated readout (no "
            f"'prompt_templates' key). Re-run --phase score; the all-caps-only "
            f"margins it holds are not a usable basis for model_knows.")
    records = payload["statements"]
    print(f"{len(records)} scored statements from {payload['model']}")

    _describe(records)

    # Attrition ladder: how many facts survive at each candidate threshold. The
    # threshold is a judgement call, so make the whole curve visible rather than
    # reporting only the chosen point.
    ladder = {}
    for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        ladder[t] = len(_surviving_facts(records, t))
    print("\nATTRITION LADDER  (base facts surviving vs threshold)")
    for t, n in ladder.items():
        mark = "  <-- chosen" if t == threshold else ""
        print(f"  |margin| >= {t:4.1f} nats : {n:3d}/100 facts{mark}")

    kept = _surviving_facts(records, threshold)
    print(f"\nTHRESHOLD {threshold} nats -> {len(kept)} base facts kept "
          f"({4 * len(kept)} rows, {2 * len(kept)} conflict rows, "
          f"{len(kept)} per conflict cell)")

    # Which statement side did the losers fail on? A fact failing because its
    # FALSE statement reads TRUE is a different problem from one whose TRUE
    # statement is unknown, and the split is worth seeing.
    by_fact = {}
    for r in records:
        by_fact.setdefault(r["base_fact"], {})[r["label"]] = r["margin"]
    fail_true = [k for k, m in by_fact.items()
                 if k not in kept and m.get(1, 0.0) < threshold]
    fail_false = [k for k, m in by_fact.items()
                  if k not in kept and m.get(0, 0.0) > -threshold]
    print(f"  failed on the TRUE statement:  {len(fail_true)}")
    print(f"  failed on the FALSE statement: {len(fail_false)}")
    print(f"  failed on both:                "
          f"{len(set(fail_true) & set(fail_false))}")
    if fail_true or fail_false:
        dropped = sorted(set(fail_true) | set(fail_false))
        print(f"  dropped facts: {', '.join(dropped[:20])}"
              f"{' ...' if len(dropped) > 20 else ''}")

    # Would the OTHER prompt have kept the same facts? Reported at the chosen
    # threshold, so the prompt-sensitivity that the all-caps bug exposed stays
    # visible in the filter decision rather than only in the score log.
    alt_kept = None
    if "alt_margin" in records[0]:
        alt_kept = _surviving_facts(records, threshold, key="alt_margin")
        print(f"\nCROSS-READOUT at the same threshold: alt keeps "
              f"{len(alt_kept)}, primary keeps {len(kept)}, "
              f"both {len(kept & alt_kept)}, primary-only "
              f"{len(kept - alt_kept)}, alt-only {len(alt_kept - kept)}")

    df = load_rows()
    df["model_knows"] = df["base_fact"].isin(kept).astype(int)

    summary = {
        "model": config.MODEL_NAME,
        "prompt_templates": payload["prompt_templates"],
        "primary_readout": payload.get("primary_readout", "primary"),
        "label_forms": payload.get("label_forms"),
        "margin_threshold_nats": threshold,
        "rule": ("a base fact passes iff margin(true stmt) >= +T and "
                 "margin(false stmt) <= -T, where margin = "
                 "logsumexp(logP over true-label case variants) - "
                 "logsumexp(same for false) at the final token of the primary "
                 "neutral prompt"),
        "n_base_facts_kept_alt_readout": None if alt_kept is None else len(alt_kept),
        "n_base_facts_kept_both_readouts": (
            None if alt_kept is None else len(kept & alt_kept)),
        "n_base_facts_total": len(by_fact),
        "n_base_facts_kept": len(kept),
        "n_rows_kept": int(df.model_knows.sum()),
        "attrition_ladder": {str(k): v for k, v in ladder.items()},
        "kept_base_facts": sorted(kept),
        "dropped_base_facts": sorted(set(by_fact) - set(kept)),
        "failed_on_true_statement": sorted(fail_true),
        "failed_on_false_statement": sorted(fail_false),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {SUMMARY_JSON}")

    # CSV last: it is the one DERIVED artifact this script mutates in place, so
    # nothing should half-apply a filter and then fail. The margins file remains
    # the source of truth — this column is regenerable at any threshold.
    df.to_csv(DATASET_CSV, index=False)
    print(f"wrote model_knows -> {DATASET_CSV} "
          f"({int(df.model_knows.sum())}/{len(df)} rows kept)")


def _surviving_facts(records, threshold, key="margin"):
    """Base facts whose true statement reads TRUE and false statement reads
    FALSE, both by at least `threshold` nats."""
    by_fact = {}
    for r in records:
        by_fact.setdefault(r["base_fact"], {})[r["label"]] = r[key]
    return {k for k, m in by_fact.items()
            if m.get(1, 0.0) >= threshold and m.get(0, 0.0) <= -threshold}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["score", "filter", "all"], default="all")
    ap.add_argument("--threshold", type=float, default=MARGIN_THRESHOLD,
                    help="minimum |margin| in nats (filter phase only)")
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.phase in ("score", "all"):
        phase_score()
    if args.phase in ("filter", "all"):
        phase_filter(args.threshold)


if __name__ == "__main__":
    main()
