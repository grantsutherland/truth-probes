"""AUX — Bootstrap CIs for the NIE intervention. CPU only, no model.

Reads results/<model>/nie_per_item.json (written by aux_nie_intervention.py) and
resamples statements to put error bars on every quantity the NIE table reports.
Without this the table invites comparisons it cannot support: whether 0.018
differs from 0.020, or 0.100 from 0.068, is not readable off point estimates
from 352 items.

TWO QUANTITIES, ALWAYS REPORTED TOGETHER:
  raw shift = the NIE numerator, in log-odds. A property of theta's causal effect.
  NIE       = raw shift / (PD+ - PD-). The denominator is a property of the
              MODEL's readout and grew ~8x from 2B to 9B, so NIE alone cannot
              distinguish "the effect shrank" from "the gap grew".

Both are signed so POSITIVE MEANS THE INTERVENTION WORKED AS INTENDED:
  f->t: adding theta to false statements should raise P(TRUE)-P(FALSE).
  t->f: subtracting theta from true statements should LOWER it, so the reported
        effect is negated. A negative number here means the intervention pushed
        the readout the wrong way.

RESAMPLING is stratified and PAIRED: true and false items are resampled
separately (preserving class balance, since PD+ and PD- are separate means), and
every train set is evaluated on the SAME resampled indices. Pairing is what makes
the between-train-set differences — the actual questions — narrower than the CIs
on the individual estimates would suggest.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                                   # noqa: E402

N_BOOT = 10000
SEED = 0
ALPHA = 5.0                                     # -> 95% percentile interval
_RESULTS = config.RESULTS_DIR
IN_JSON = os.path.join(_RESULTS, "nie_per_item.json")
OUT_JSON = os.path.join(_RESULTS, "nie_bootstrap.json")

# (a, b) -> the CI is on a - b. Contrasts absent from the input are skipped, so
# this list covers both the pre- and post-rescaling runs.
CONTRASTS = [
    # The mechanism's own predictions: does combining degrade the causal effect?
    ("larger_than+smaller_than", "larger_than"),
    ("cities+neg_cities", "cities"),
    # Same question with push MAGNITUDE held fixed. If the degradation above
    # survives here, it is a property of the combined DIRECTION.
    ("larger_than+smaller_than|norm=larger_than", "larger_than"),
    ("cities+neg_cities|norm=cities", "cities"),
    # How much of the raw degradation was magnitude alone (rescaled - raw combo).
    ("larger_than+smaller_than|norm=larger_than", "larger_than+smaller_than"),
    ("cities+neg_cities|norm=cities", "cities+neg_cities"),
]


def effects(pd_base, pd_star, is_true, idx_t, idx_f):
    """(shift_f2t, shift_t2f, nie_f2t, nie_t2f) for one resample.

    Signed so positive = intervention worked as intended (see module docstring).
    """
    PDp = pd_base[idx_t].mean()
    PDm = pd_base[idx_f].mean()
    gap = PDp - PDm
    PDp_s = pd_star[idx_t].mean()
    PDm_s = pd_star[idx_f].mean()
    shift_ft = PDm_s - PDm
    shift_tf = PDp - PDp_s                      # negated: positive = pushed toward FALSE
    return shift_ft, shift_tf, shift_ft / gap, shift_tf / gap


def ci(samples):
    lo, hi = np.percentile(samples, [ALPHA / 2, 100 - ALPHA / 2])
    return {"mean": float(np.mean(samples)), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    with open(IN_JSON) as f:
        d = json.load(f)

    labels = np.array(d["labels"])
    pd_base = np.array(d["pd_baseline"])
    pd_star = {ts: np.array(v) for ts, v in d["pd_star"].items()}
    where_t = np.flatnonzero(labels == 1)
    where_f = np.flatnonzero(labels == 0)
    print(f"{d['model']} layer {d['layer']} test={d['test_set']}: "
          f"{len(where_t)} true / {len(where_f)} false items, {N_BOOT} resamples")

    rng = np.random.default_rng(SEED)
    # Draw once, reuse across train sets -> paired.
    boot_t = rng.choice(where_t, size=(N_BOOT, len(where_t)), replace=True)
    boot_f = rng.choice(where_f, size=(N_BOOT, len(where_f)), replace=True)

    # samples[ts] = (N_BOOT, 4) array of [shift_ft, shift_tf, nie_ft, nie_tf]
    samples = {}
    for ts, star in pd_star.items():
        s = np.empty((N_BOOT, 4))
        for b in range(N_BOOT):
            s[b] = effects(pd_base, star, labels, boot_t[b], boot_f[b])
        samples[ts] = s

    point = {ts: effects(pd_base, star, labels, where_t, where_f)
             for ts, star in pd_star.items()}

    names = ["shift_f2t", "shift_t2f", "NIE_f2t", "NIE_t2f"]
    out = {"model": d["model"], "layer": d["layer"], "test_set": d["test_set"],
           "n_boot": N_BOOT, "seed": SEED,
           "n_true": len(where_t), "n_false": len(where_f),
           "gap": float(pd_base[where_t].mean() - pd_base[where_f].mean()),
           "per_train_set": {}, "contrasts": {}, "asymmetry": {}}

    print(f"\nbaseline gap (PD+ - PD-) = {out['gap']:.3f}")
    print("\n=== per train set: point [95% CI] ===")
    for ts in pd_star:
        out["per_train_set"][ts] = {
            n: {"point": float(point[ts][i]), **ci(samples[ts][:, i])}
            for i, n in enumerate(names)}
        r = out["per_train_set"][ts]
        print(f"\n  {ts}")
        for n in names:
            v = r[n]
            star = "  *" if v["excludes_zero"] else "  ns"
            print(f"    {n:10} {v['point']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]{star}")

    # ---- the actual questions: combined MINUS singleton, paired ----
    print("\n=== contrasts (combined - singleton), paired resamples ===")
    print("    negative => combined DEGRADED the effect")
    for comb, single in CONTRASTS:
        if comb not in samples or single not in samples:
            continue
        diff = samples[comb] - samples[single]
        key = f"{comb} - {single}"
        out["contrasts"][key] = {
            n: {"point": float(point[comb][i] - point[single][i]), **ci(diff[:, i])}
            for i, n in enumerate(names)}
        print(f"\n  {key}")
        for i, n in enumerate(names):
            v = out["contrasts"][key][n]
            star = "  SIGNIFICANT" if v["excludes_zero"] else "  ns"
            print(f"    d{n:10} {v['point']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]{star}")

        # Is the degradation WORSE for t->f than f->t? Difference-of-differences
        # on the raw shifts (same denominator, so NIE would give the same answer).
        dod = diff[:, 1] - diff[:, 0]
        pt = (point[comb][1] - point[single][1]) - (point[comb][0] - point[single][0])
        out["asymmetry"][key] = {"point": float(pt), **ci(dod)}
        v = out["asymmetry"][key]
        star = "  SIGNIFICANT" if v["excludes_zero"] else "  ns"
        print(f"    asymmetry (d_t2f - d_f2t) {pt:+.4f}  "
              f"[{v['lo']:+.4f}, {v['hi']:+.4f}]{star}")

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
