"""H2 PHASE 3 — the actual test: the conflict cells.

Applies the same FROZEN H1 cities probes to the rows where the frame and the
statement DISAGREE, and asks which one the truth direction follows.

Registered prediction: results/h2_prereg_phase3.json, committed before any
conflict-cell score existed. This script evaluates P-H2-1 .. P-H2-6 mechanically
against the criteria recorded there, so the verdicts are not written after
looking at the numbers.

ONE NUMBER, NOT TWO. In the conflict cells frame == 1 - label for all 196 rows,
so AUROC(score vs label) + AUROC(score vs frame) = 1 identically. Both are
reported because the plan asks for both, but the second carries no information
the first does not, and they must never be presented as independent evidence.
This is asserted at runtime rather than assumed.

PER-CELL AUROC DOES NOT EXIST. Within a single conflict cell the label is
constant (frame=true cells are all false statements; frame=false cells are all
true statements), so there is no positive/negative pair to rank and AUROC is
undefined. The per-cell readout is therefore the mean SIGNED MARGIN, which is
what the H2 design notes asked for anyway: a probe can keep its ranking while its
margins collapse toward zero, and AUROC alone would hide that.

MARGINS USE ONE COMMON CENTERING. Aligned and conflict scores are centered by the
same global mean over all 392 rows, label-free, so the two conditions sit on a
comparable scale. Centering each set by its own mean would remove exactly the
between-condition shift that P-H2-6 is about.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                                            # noqa: E402
import data                                              # noqa: E402

PROBE_TRAIN_SET = "cities"
PROBE_TYPES = ["mm", "lr"]
N_BOOTSTRAP = 10000
SEED = 0

PREREG = os.path.join(os.path.dirname(__file__), "..", "results",
                      "h2_prereg_phase3.json")
OUT_JSON = os.path.join(config.RESULTS_DIR, "h2_conflict.json")


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney). Identical to 03_transfer_eval's."""
    labels = labels.long()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_auroc(scores, labels, n=N_BOOTSTRAP, seed=SEED):
    """Stratified percentile CI, resampling each class separately."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.where(labels == 1)[0]
    neg = torch.where(labels == 0)[0]
    vals = []
    for _ in range(n):
        i = pos[torch.randint(len(pos), (len(pos),), generator=g)]
        j = neg[torch.randint(len(neg), (len(neg),), generator=g)]
        idx = torch.cat([i, j])
        vals.append(auroc(scores[idx], labels[idx]))
    v = torch.tensor(vals, dtype=torch.double)
    return float(torch.quantile(v, 0.025)), float(torch.quantile(v, 0.975))


def main():
    torch.manual_seed(SEED)
    with open(PREREG) as f:
        prereg = json.load(f)
    print(f"prereg registered {prereg['registered_utc']} "
          f"(conflict_cells_read={prereg['state_at_registration']['conflict_cells_read']})")

    acts, labels, frame, meta = data.load_induced_lies()
    layers = meta["layers"]
    committed = config.chosen_layer()
    idx = layers.index(committed)
    cond = meta["condition"]

    is_conf = torch.tensor([c == "conflict" for c in cond])
    is_align = ~is_conf
    lab_c, frm_c = labels[is_conf], frame[is_conf]
    lab_a = labels[is_align]

    # The complementarity that makes this one number, checked not assumed.
    assert torch.equal(frm_c, 1 - lab_c), \
        "conflict cells must satisfy frame == 1 - label"
    print(f"conflict rows: {int(is_conf.sum())}  "
          f"({int((lab_c == 1).sum())} true statements under a 'false' frame, "
          f"{int((lab_c == 0).sum())} false statements under a 'true' frame)")

    results, summary = {}, {}
    for p in PROBE_TYPES:
        probe, pmeta = data.load_probe(p, PROBE_TRAIN_SET)
        assert pmeta["model_name"] == config.MODEL_NAME
        assert pmeta["layer"] == committed
        direction = probe.direction

        all_scores = acts[:, idx, :] @ direction
        # One common, label-free centering constant for both conditions.
        centered = all_scores - all_scores.mean()
        s_c, s_a = centered[is_conf], centered[is_align]

        a_truth = auroc(s_c, lab_c)
        lo, hi = bootstrap_auroc(s_c, lab_c)
        a_frame = auroc(s_c, frm_c)

        # Per-cell mean signed margin (AUROC is undefined per cell — see module
        # docstring). Cells keyed by the frame they carry.
        cells = {}
        for cname, mask in (
            ("conflict/frame=true(false stmt)", is_conf & (frame == 1)),
            ("conflict/frame=false(true stmt)", is_conf & (frame == 0)),
            ("aligned/frame=true(true stmt)", is_align & (frame == 1)),
            ("aligned/frame=false(false stmt)", is_align & (frame == 0)),
        ):
            v = centered[mask]
            cells[cname] = {"n": int(mask.sum()), "mean_margin": float(v.mean()),
                            "sd": float(v.std())}

        absmean_c, absmean_a = float(s_c.abs().mean()), float(s_a.abs().mean())
        retention = absmean_c / absmean_a if absmean_a else float("nan")

        by_layer = {}
        for L in config.plateau_layers():
            sl = acts[:, layers.index(L), :] @ direction
            by_layer[L] = auroc(sl[is_conf], lab_c)

        print(f"\n[{p}] CONFLICT-CELL AUROC vs TRUTH: {a_truth:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"     AUROC vs FRAME: {a_frame:.4f}  "
              f"(= 1 - the above by construction; carries no extra information)")
        print(f"     mean |margin|: conflict {absmean_c:.4f} vs aligned "
              f"{absmean_a:.4f}  -> retention {retention:.3f}")
        for cname, c in cells.items():
            print(f"       {cname:34} n={c['n']:3d}  "
                  f"mean margin {c['mean_margin']:+.4f}  sd {c['sd']:.4f}")
        lo_l = min(by_layer, key=by_layer.get)
        print(f"     plateau L{config.plateau_layers()[0]}-"
              f"{config.plateau_layers()[-1]}: min {by_layer[lo_l]:.4f} @L{lo_l}"
              f"  max {max(by_layer.values()):.4f}")

        results[p] = {
            "auroc_vs_truth": a_truth,
            "auroc_vs_truth_ci95": [lo, hi],
            "auroc_vs_frame": a_frame,
            "cells": cells,
            "mean_abs_margin_conflict": absmean_c,
            "mean_abs_margin_aligned": absmean_a,
            "margin_retention": retention,
            "auroc_by_plateau_layer": {str(k): v for k, v in by_layer.items()},
        }
        summary[p] = (a_truth, lo, hi, retention, cells, by_layer)

    # ---- mechanical evaluation against the registered criteria ----
    a_mm, a_lr = summary["mm"][0], summary["lr"][0]
    ci_excl = {p: (summary[p][1] > 0.5 or summary[p][2] < 0.5) for p in PROBE_TYPES}
    verdicts = {}

    verdicts["P-H2-1"] = (
        "CONFIRMED" if all(summary[p][0] > 0.5 and ci_excl[p] for p in PROBE_TYPES)
        else ("REFUTED (tracks frame)"
              if any(summary[p][0] < 0.5 for p in PROBE_TYPES)
              else "REFUTED (null: CI spans 0.5)"))
    verdicts["P-H2-2"] = ("CONFIRMED" if all(summary[p][0] >= 0.85 for p in PROBE_TYPES)
                          else "REFUTED")
    verdicts["P-H2-3"] = ("REFUTED (frame had no measurable effect)"
                          if all(summary[p][0] == 1.0 for p in PROBE_TYPES)
                          else "CONFIRMED")
    verdicts["P-H2-4"] = "CONFIRMED" if abs(a_mm - a_lr) <= 0.10 else "REFUTED"
    verdicts["P-H2-5"] = ("CONFIRMED"
                          if all(v > 0.5 for p in PROBE_TYPES
                                 for v in summary[p][5].values())
                          else "REFUTED")
    signs_ok = all(
        c["mean_margin"] > 0 if "true stmt" in name else c["mean_margin"] < 0
        for p in PROBE_TYPES for name, c in summary[p][4].items())
    verdicts["P-H2-6"] = ("CONFIRMED"
                          if signs_ok and all(summary[p][3] >= 0.5 for p in PROBE_TYPES)
                          else "REFUTED")

    print(f"\n{'=' * 70}\nPREREG VERDICTS")
    for k, v in verdicts.items():
        claim = next(x["claim"] for x in prereg["predictions"] if x["id"] == k)
        print(f"  {k}: {v}\n      {claim[:110]}")
    print("=" * 70)
    print("\nINTERPRETIVE LIMIT (ships with every report of this result):")
    print("  " + prereg["interpretive_limit_shipped_regardless"][:400] + " ...")

    out = {
        "model": config.MODEL_NAME,
        "layer": committed,
        "probe_train_set": PROBE_TRAIN_SET,
        "retrained": False,
        "n_conflict_rows": int(is_conf.sum()),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "prereg_file": "results/h2_prereg_phase3.json",
        "prereg_verdicts": verdicts,
        "auroc_vs_frame_is_complement": True,
        "probes": results,
        "interpretive_limit": prereg["interpretive_limit_shipped_regardless"],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
