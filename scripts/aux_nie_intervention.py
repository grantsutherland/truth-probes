"""AUX — Causal intervention (NIE), replicating Marks & Tegmark Section 6.1.

Tests the refined mechanism: combined-MM NIE degrades when constituents are
ANTIPODAL (near-cancellation -> noise-dominated residual DIRECTION), not merely
orthogonal. Registered prediction: antipodal -> degrades, orthogonal -> preserved.

STATUS after both runs (see results/eval_9b.md):
  * The ANTIPODAL arm is UNTESTABLE at both scales, for different structural
    reasons. At 2B larger_than had no causal effect on sp_en_trans to degrade
    from (NIE ~0.03, cos to the test truth direction ~0.002). At 9B the pair is
    no longer antipodal at all (cos -0.617 -> +0.301, the P3 confirmation), so
    the mechanism now predicts preservation rather than degradation.
  * The ORTHOGONAL arm went the WRONG WAY: cities+neg_cities degrades sharply
    vs cities (t->f raw shift +0.560 -> +0.030), where preservation was
    predicted. Whether that is direction or magnitude is what RESCALE_TO tests.

Runs on the model named by src/config.py (9B on the cluster; 2B/MPS locally),
forward passes only. Set N_TEST small first to validate.

Protocol (M&T Sec 6.1), MM probes, theta = raw mass-mean (mu_true - mu_false) at
the chosen layer computed from the TRAINING set's cached activations:
  * few-shot prompt: 2 labeled sp_en_trans examples, then 'stmt This statement is:'
  * read P(TRUE) - P(FALSE) on the next token.
  * intervene at the end-of-statement token, layer L, hook_resid_post:
      false statements: ADD theta   -> should push toward TRUE
      true  statements: SUBTRACT theta -> should push toward FALSE
  * NIE_{f->t} = (PD-_star - PD-) / (PD+ - PD-)
    NIE_{t->f} = (PD+_star - PD+) / (PD- - PD+)

Diagnostics also reported so failure can be attributed to DIRECTION vs norm:
theta norm, cos(theta_combo, theta of the test set's own truth direction), the
raw logit shifts, and the baseline gap (see the shift_* keys below on why NIE
alone is not interpretable across models).
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                                 # noqa: E402
import data                                   # noqa: E402
from activations import load_model, resid_hook_name  # noqa: E402

LAYER = config.chosen_layer()
TEST_SET = "sp_en_trans"
TRAIN_SETS = ["larger_than", "larger_than+smaller_than", "cities", "cities+neg_cities"]

# Combo -> the singleton whose norm it is rescaled to match.
#
# theta_combo = 1/2(theta_A + theta_B) exactly, so whenever the constituents are
# not aligned the combined vector is SHORTER than either — at 9B, 0.744x for
# cities+neg_cities. The combined intervention is then a physically weaker push,
# and singleton-vs-combo NIE confounds direction quality with push magnitude:
# for the cities pair, cos-to-test falls 0.771x and the norm falls 0.744x, which
# this design cannot separate. (Adding the raw mass-mean IS M&T's normalization
# --- p(mu- + theta) = p(mu+) holds identically when theta = mu+ - mu-. But that
# is a PER-PROBE normalization against each probe's own training distribution;
# it does not equalize two different probes against each other.)
#
# Rescaling to the singleton's norm holds magnitude fixed so only DIRECTION
# varies. If the degradation survives, orthogonal constituents degrade causally
# while leaving classification intact, and the registered P4 is wrong in an
# interesting way; if it vanishes, the arm was a magnitude artifact.
RESCALE_TO = {"larger_than+smaller_than": "larger_than",
              "cities+neg_cities": "cities"}
N_TEST = None            # None = all; set an int to subsample for validation
_RESULTS = config.RESULTS_DIR
OUT_JSON = os.path.join(_RESULTS, "nie_intervention.json")
# Per-statement readouts, kept in a separate file so the findings JSON stays
# small. NIE is a ratio of MEANS, so the aggregate file cannot support a
# bootstrap; without this, every error bar costs another 8-minute GPU job.
PER_ITEM_JSON = os.path.join(_RESULTS, "nie_per_item.json")

# Two fixed few-shot exemplars (excluded from the test loop).
FEWSHOT = [
    ("The Spanish word 'fruta' means 'goat'.", "FALSE"),
    ("The Spanish word 'carne' means 'meat'.", "TRUE"),
]


def raw_theta(train_set, layer):
    """mu_true - mu_false at `layer`, from cached all-layer sweep activations.
    For a combo 'A+B', concatenate A and B first (balanced -> average of thetas)."""
    parts = train_set.split("+")
    acts_list, lab_list = [], []
    for p in parts:
        a, l, meta = data.load_sweep(p)
        idx = meta["layers"].index(layer)
        acts_list.append(a[:, idx, :])
        lab_list.append(l)
    acts = torch.cat(acts_list, 0)
    labels = torch.cat(lab_list, 0)
    return acts[labels == 1].mean(0) - acts[labels == 0].mean(0)


def build_prompt(stmt):
    lines = [f"{s} This statement is: {lab}" for s, lab in FEWSHOT]
    prefix = "\n".join(lines) + "\n" + stmt          # ends at end-of-statement token
    full = prefix + " This statement is:"
    return prefix, full


def true_false_ids(model):
    """Token ids for the readout labels, as they appear after "This statement is:".

    The leading space matters: the prompt ends at "is:", so the continuation the
    model actually emits is " TRUE"/" FALSE", and the unspaced ids would score a
    continuation the model never produces.

    Taking [0, 0] is only correct if each is a SINGLE token. If the tokenizer
    splits " TRUE" into pieces, [0, 0] silently returns the first fragment and
    every NIE number is computed against the wrong logit — a corruption that
    looks like a weak-but-plausible causal effect rather than an error. Assert
    it instead, so a vocab difference at 9B surfaces here rather than in the
    results.
    """
    ids = {}
    for s in (" TRUE", " FALSE"):
        toks = model.to_tokens(s, prepend_bos=False)[0]
        if len(toks) != 1:
            pieces = [model.to_string(t.unsqueeze(0)) for t in toks]
            raise ValueError(
                f"{s!r} tokenizes to {len(toks)} tokens {pieces} under "
                f"{model.cfg.model_name}, not 1. The P(TRUE)-P(FALSE) readout "
                f"assumes a single-token continuation; pick label words that are "
                f"single tokens for this vocab before trusting any NIE number."
            )
        ids[s] = toks[0].item()
    return ids[" TRUE"], ids[" FALSE"]


def main():
    torch.manual_seed(0)
    model = load_model(config.MODEL_NAME)
    hook_name = resid_hook_name(LAYER)
    TRUE_ID, FALSE_ID = true_false_ids(model)
    print(f"token ids: ' TRUE'={TRUE_ID}, ' FALSE'={FALSE_ID}")

    texts, labels = data.load_statements(TEST_SET)
    fewshot_stmts = {s for s, _ in FEWSHOT}
    items = [(t, int(l)) for t, l in zip(texts, labels.tolist()) if t not in fewshot_stmts]
    if N_TEST:
        items = items[:N_TEST]
    print(f"test items: {len(items)} from {TEST_SET}")

    thetas = {ts: raw_theta(ts, LAYER).to(next(model.parameters()).device,
                                          next(model.parameters()).dtype)
              for ts in TRAIN_SETS}
    # Norm-matched variants: same direction as the combo, same length as the
    # singleton it is compared against.
    for combo, single in RESCALE_TO.items():
        scale = thetas[single].norm() / thetas[combo].norm()
        thetas[f"{combo}|norm={single}"] = thetas[combo] * scale
    conditions = TRAIN_SETS + [f"{c}|norm={s}" for c, s in RESCALE_TO.items()]
    theta_test = raw_theta(TEST_SET, LAYER)  # test set's own truth direction (diagnostic)

    def readout(full_prompt, intervention=None):
        """P(TRUE)-P(FALSE) at the last position. intervention=(theta, sign, pos)."""
        tokens = model.to_tokens(full_prompt)
        handles = []
        if intervention is not None:
            theta, sign, pos = intervention
            def hook(act, hook):
                act[:, pos, :] = act[:, pos, :] + sign * theta
                return act
            handles.append((hook_name, hook))
        with torch.no_grad():
            if handles:
                logits = model.run_with_hooks(tokens, fwd_hooks=handles)
            else:
                logits = model(tokens)
        logp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
        return (logp[TRUE_ID] - logp[FALSE_ID]).item()

    # end-of-statement token position per item (last token of prefix)
    def eos_pos(prefix):
        return model.to_tokens(prefix).shape[1] - 1

    # ---- baseline PD+ / PD- (no intervention), computed once ----
    pd_plus, pd_minus = [], []
    prompts = {}
    pd_base = []                       # per item, aligned with `items`
    for stmt, lab in items:
        prefix, full = build_prompt(stmt)
        prompts[stmt] = (prefix, full)
        d = readout(full)
        pd_base.append(d)
        (pd_plus if lab == 1 else pd_minus).append(d)
    PDp = sum(pd_plus) / len(pd_plus)
    PDm = sum(pd_minus) / len(pd_minus)
    print(f"\nbaseline: PD+ (true) = {PDp:.3f}  PD- (false) = {PDm:.3f}  "
          f"gap = {PDp - PDm:.3f}")

    out = {"layer": LAYER, "test_set": TEST_SET, "n_test": len(items),
           "baseline": {"PD_plus": PDp, "PD_minus": PDm,
                        "gap": PDp - PDm}, "train_sets": {}}
    per_item = {"layer": LAYER, "test_set": TEST_SET,
                "model": config.MODEL_NAME,
                "labels": [lab for _, lab in items],
                "statements": [stmt for stmt, _ in items],
                "pd_baseline": pd_base, "pd_star": {}}

    for ts in conditions:
        theta = thetas[ts]
        pdp_star, pdm_star = [], []
        star = []                      # per item, aligned with `items`
        for stmt, lab in items:
            prefix, full = prompts[stmt]
            pos = eos_pos(prefix)
            sign = -1.0 if lab == 1 else +1.0   # subtract for true, add for false
            d = readout(full, intervention=(theta, sign, pos))
            star.append(d)
            (pdp_star if lab == 1 else pdm_star).append(d)
        per_item["pd_star"][ts] = star
        PDp_s = sum(pdp_star) / len(pdp_star)
        PDm_s = sum(pdm_star) / len(pdm_star)
        nie_ft = (PDm_s - PDm) / (PDp - PDm)
        nie_tf = (PDp_s - PDp) / (PDm - PDp)
        theta_cpu = theta.float().cpu()
        cos_test = float(torch.dot(theta_cpu / theta_cpu.norm(),
                                   theta_test / theta_test.norm()))
        out["train_sets"][ts] = {
            "NIE_false_to_true": nie_ft, "NIE_true_to_false": nie_tf,
            # Raw logit shifts = the NIE numerators, BEFORE dividing by the
            # baseline gap. The gap is a property of the MODEL's readout, not of
            # theta, and it grew ~8x from 2B to 9B — so NIE alone cannot say
            # whether an effect shrank or the denominator grew. Report both.
            "shift_false_to_true": PDm_s - PDm,
            "shift_true_to_false": PDp_s - PDp,
            "PD_plus_star": PDp_s, "PD_minus_star": PDm_s,
            "theta_norm": float(theta_cpu.norm()),
            "cos_theta_to_test_truth_dir": cos_test,
        }
        print(f"  {ts:26} NIE f->t {nie_ft:+.3f}  t->f {nie_tf:+.3f}  "
              f"| raw shift f->t {PDm_s - PDm:+.3f}  t->f {PDp_s - PDp:+.3f}  "
              f"|theta| {theta_cpu.norm():.2f}  cos->test {cos_test:+.3f}")

    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    with open(PER_ITEM_JSON, "w") as f:
        json.dump(per_item, f)
    print(f"\nsaved -> {OUT_JSON}\nsaved -> {PER_ITEM_JSON}")


if __name__ == "__main__":
    main()
