"""AUX — Causal intervention (NIE), replicating Marks & Tegmark Section 6.1.

Tests the refined mechanism: combined-MM NIE degrades when constituents are
ANTIPODAL (near-cancellation -> noise-dominated residual DIRECTION), not merely
orthogonal. Prediction at 2B (geometry measured, not inferred):
  cos(larger,smaller) = -0.62 (antipodal-ish) -> larger+smaller combined NIE
    DEGRADES vs larger_than alone.
  cos(cities,neg_cities) = -0.02 (orthogonal)  -> cities+neg_cities combined NIE
    does NOT degrade vs cities alone.
Classification is already known to survive both (combined MM ~0.99 own-held-out),
so a NIE drop for larger+smaller only = M&T's classification-vs-NIE dissociation
reproduced within 2B.

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
theta norm and cos(theta_combo, theta of the test set's own truth direction).

Runs on 2B/MPS, forward passes only. Set N_TEST small first to validate.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data                                   # noqa: E402
from activations import load_model, resid_hook_name  # noqa: E402

LAYER = 13
TEST_SET = "sp_en_trans"
TRAIN_SETS = ["larger_than", "larger_than+smaller_than", "cities", "cities+neg_cities"]
N_TEST = None            # None = all; set an int to subsample for validation
_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_JSON = os.path.join(_RESULTS, "nie_intervention.json")

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
    tid = lambda s: model.to_tokens(s, prepend_bos=False)[0, 0].item()
    return tid(" TRUE"), tid(" FALSE")


def main():
    torch.manual_seed(0)
    model = load_model("gemma-2-2b")
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
    for stmt, lab in items:
        prefix, full = build_prompt(stmt)
        prompts[stmt] = (prefix, full)
        d = readout(full)
        (pd_plus if lab == 1 else pd_minus).append(d)
    PDp = sum(pd_plus) / len(pd_plus)
    PDm = sum(pd_minus) / len(pd_minus)
    print(f"\nbaseline: PD+ (true) = {PDp:.3f}  PD- (false) = {PDm:.3f}  "
          f"gap = {PDp - PDm:.3f}")

    out = {"layer": LAYER, "test_set": TEST_SET, "n_test": len(items),
           "baseline": {"PD_plus": PDp, "PD_minus": PDm}, "train_sets": {}}

    for ts in TRAIN_SETS:
        theta = thetas[ts]
        pdp_star, pdm_star = [], []
        for stmt, lab in items:
            prefix, full = prompts[stmt]
            pos = eos_pos(prefix)
            sign = -1.0 if lab == 1 else +1.0   # subtract for true, add for false
            d = readout(full, intervention=(theta, sign, pos))
            (pdp_star if lab == 1 else pdm_star).append(d)
        PDp_s = sum(pdp_star) / len(pdp_star)
        PDm_s = sum(pdm_star) / len(pdm_star)
        nie_ft = (PDm_s - PDm) / (PDp - PDm)
        nie_tf = (PDp_s - PDp) / (PDm - PDp)
        theta_cpu = theta.float().cpu()
        cos_test = float(torch.dot(theta_cpu / theta_cpu.norm(),
                                   theta_test / theta_test.norm()))
        out["train_sets"][ts] = {
            "NIE_false_to_true": nie_ft, "NIE_true_to_false": nie_tf,
            "PD_plus_star": PDp_s, "PD_minus_star": PDm_s,
            "theta_norm": float(theta_cpu.norm()),
            "cos_theta_to_test_truth_dir": cos_test,
        }
        print(f"  {ts:26} NIE f->t {nie_ft:+.3f}  t->f {nie_tf:+.3f}  "
              f"|theta| {theta_cpu.norm():.2f}  cos->test {cos_test:+.3f}")

    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
