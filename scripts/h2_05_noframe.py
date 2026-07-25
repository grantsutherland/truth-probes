"""H2 — THE NO-FRAME BASELINE. Closes the design's binding gap.

Every H2 cell carries a frame, so Phase 4 identifies the DIFFERENCE between the
two frames and not which one moved. "The lie-frame compresses the truth gap by
36%" and "the truth-frame amplifies it by 56%" fit those data identically.

This script scores the same 196 statements with NO instruction prefix — the bare
sentence, exactly the form the H1 cities probes were fitted on — and places that
neutral gap against the two framed gaps.

    bare gap ~ truth-frame gap   =>  the LIE-frame compresses
    bare gap ~ lie-frame gap     =>  the TRUTH-frame amplifies
    bare gap in between          =>  both, and the split is quantified
    bare gap ABOVE both          =>  both frames compress (what actually happened)
    bare gap BELOW both          =>  both frames amplify

The last two were not anticipated when this was written: the first version's
verdict logic assumed the neutral gap falls between the two framed gaps, and
reported `pos > 0.75` as "the lie-frame compresses, neutral sits at the
truth-frame end". The observed `pos` is 2.75, i.e. neutral sits far ABOVE the
truth-frame, and that branch described it wrongly. Both frames turn out to
compress; the two-option framing in the write-ups was a false dichotomy.

ANCHOR. A bare-vs-framed comparison is only meaningful if the bare number is
itself trustworthy, so it is reported against the H1 `cities` gap under the same
frozen probe — 1496 independent statements the probe was actually fitted on. If
the bare gap were far from that, "both frames compress" would be indistinguishable
from the H2 statement set simply being unusual. It is not: bare is ~0.9x H1.

THE GAP IS THE RIGHT STATISTIC because it is centering-invariant. A truth gap is
a difference of two class means, so any constant offset cancels — which matters
here more than anywhere else in H2, because the bare set has a different global
mean from the framed set and there is no shared, principled zero between them.
Per-class numbers are also reported, but only against an explicitly stated common
boundary, and they are flagged as depending on that choice while the gap does not.

Bootstrap resamples BASE FACTS (each contributes its true and its false
statement), matching the clustering used everywhere else in H2.

    --phase extract   GPU. 196 short forward passes.
    --phase analyze   CPU. The comparison; rerunnable without the GPU.
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
from activations import load_model, verify_model, extract_activations  # noqa: E402

DATASET_CSV = os.path.join(dataio.DATA_DIR, "induced_lies.csv")
DECOMP_JSON = os.path.join(config.RESULTS_DIR, "h2_decompose.json")
OUT_JSON = os.path.join(config.RESULTS_DIR, "h2_noframe.json")
PROBE_TRAIN_SET = "cities"
PROBE_TYPES = ["mm", "lr"]
N_BOOTSTRAP = 10000
SEED = 0


def surviving_statements():
    """The 196 unique statements of the 98 facts Phase 0 kept, deduplicated."""
    df = pd.read_csv(DATASET_CSV, dtype={"frame": str, "model_knows": str},
                     keep_default_na=False)
    if (df["model_knows"] == "").any():
        raise ValueError("model_knows unpopulated; run h2_00 --phase filter.")
    kept = df[df["model_knows"].astype(int) == 1]
    uniq = (kept[["base_fact", "statement", "label"]]
            .drop_duplicates(subset="statement")
            .sort_values(["base_fact", "label"], ascending=[True, False])
            .reset_index(drop=True))
    per_fact = uniq.groupby("base_fact").size()
    if set(per_fact.unique()) != {2}:
        raise ValueError(f"expected 2 statements per fact, got {per_fact.unique()}")
    return uniq


def phase_extract():
    uniq = surviving_statements()
    print(f"{len(uniq)} bare statements across {uniq.base_fact.nunique()} facts")

    model = load_model(config.MODEL_NAME)
    print("verify_model:", json.dumps(verify_model(model), indent=2, default=str))
    layers = list(range(model.cfg.n_layers))

    # Same token convention as Phase 1, re-checked rather than assumed: with no
    # prefix the last real token must still be the statement's period.
    for s in uniq["statement"].head(2):
        toks = model.to_tokens(s)[0]
        pieces = [model.to_string(t.unsqueeze(0)) for t in toks[-4:]]
        print(f"  {s!r}\n    last 4 tokens: {pieces} -> gathering {pieces[-1]!r}")

    acts = extract_activations(model, uniq["statement"].tolist(), layers=layers,
                               token_pos=-1, batch_size=16)
    expected = (len(uniq), len(layers), model.cfg.d_model)
    if tuple(acts.shape) != expected:
        raise RuntimeError(f"shape {tuple(acts.shape)} != {expected}")
    if not torch.isfinite(acts).all():
        raise RuntimeError("non-finite activations")
    n_uniq = len({tuple(r[:8].tolist()) for r in acts[:, layers.index(
        config.chosen_layer()), :]})
    print(f"acts {tuple(acts.shape)}  finite OK  {n_uniq}/{len(uniq)} distinct rows")
    if n_uniq < len(uniq):
        raise RuntimeError("duplicate activation rows — bad token gather")

    dataio.save_bare_statements(acts, uniq["statement"].tolist(),
                                uniq["label"].astype(int).tolist(),
                                uniq["base_fact"].tolist(), layers,
                                token_pos=-1, model_name=config.MODEL_NAME)
    print(f"saved -> {dataio._cache_path(dataio.BARE_STATEMENTS)}")


def h1_anchor():
    """The H1 `cities` activations, for the sanity anchor. None if uncached.

    Optional by design: the anchor makes the bare number interpretable, but the
    H1 cache is gitignored, and a missing cache should not stop the comparison
    from running. Absence is recorded in the JSON rather than silently skipped.
    """
    try:
        acts, labels, meta = dataio.load_activations(PROBE_TRAIN_SET)
    except FileNotFoundError:
        return None
    if meta["layer"] != config.chosen_layer():
        return None
    return acts, labels


def phase_analyze():
    acts, labels, meta = dataio.load_bare_statements()
    if meta["model_name"] != config.MODEL_NAME:
        raise ValueError(f"cache is {meta['model_name']!r}")
    idx = meta["layers"].index(config.chosen_layer())
    facts = meta["base_fact"]

    # (n_facts, 2) row indices: column 0 = true statement, column 1 = false.
    order = {}
    for i, (f, l) in enumerate(zip(facts, labels.tolist())):
        order.setdefault(f, {})[int(l)] = i
    fact_ids = sorted(order)
    grid = torch.tensor([[order[f][1], order[f][0]] for f in fact_ids])
    n_facts = len(fact_ids)

    with open(DECOMP_JSON) as f:
        decomp = json.load(f)

    anchor = h1_anchor()
    print(f"{n_facts} facts x 2 statements = {grid.numel()} bare rows")
    print("H1 anchor: " + ("cities cache present"
                           if anchor else "cities cache ABSENT — anchor skipped") + "\n")
    results = {}
    for p in PROBE_TYPES:
        probe, pmeta = dataio.load_probe(p, PROBE_TRAIN_SET)
        assert pmeta["model_name"] == config.MODEL_NAME
        assert pmeta["layer"] == config.chosen_layer()
        s = (acts[:, idx, :] @ probe.direction).double()

        t_bare, f_bare = s[grid[:, 0]], s[grid[:, 1]]
        gap_bare = float(t_bare.mean() - f_bare.mean())

        d = decomp["probes"][p]
        gap_lie = d["truth_gap_lie_frame"]
        gap_truth = d["truth_gap_truth_frame"]
        # Where the neutral gap sits on the lie->truth axis. 0 = at the
        # lie-frame gap, 1 = at the truth-frame gap.
        pos = (gap_bare - gap_lie) / (gap_truth - gap_lie)
        lie_effect = gap_lie - gap_bare        # negative => lie-frame compresses
        truth_effect = gap_truth - gap_bare    # positive => truth-frame amplifies

        g = torch.Generator().manual_seed(SEED)
        gb, pd_, le, te = [], [], [], []
        for _ in range(N_BOOTSTRAP):
            k = torch.randint(n_facts, (n_facts,), generator=g)
            gv = (s[grid[k, 0]].mean() - s[grid[k, 1]].mean()).item()
            gb.append(gv)
            pd_.append((gv - gap_lie) / (gap_truth - gap_lie))
            le.append(gap_lie - gv)
            te.append(gap_truth - gv)

        def ci(v):
            v = torch.tensor(v, dtype=torch.double)
            return [float(torch.quantile(v, 0.025)), float(torch.quantile(v, 0.975))]

        ci_g, ci_p, ci_l, ci_t = ci(gb), ci(pd_), ci(le), ci(te)

        gap_h1, bare_over_h1 = None, None
        if anchor is not None:
            ha, hl = anchor
            hs = (ha @ probe.direction).double()
            gap_h1 = float(hs[hl == 1].mean() - hs[hl == 0].mean())
            bare_over_h1 = gap_bare / gap_h1

        print(f"[{p}] truth gap along the frozen direction")
        print(f"   'Say something false.' (lie-frame)   {gap_lie:8.3f}")
        print(f"    NO FRAME (neutral baseline)         {gap_bare:8.3f}   "
              f"95% CI [{ci_g[0]:.3f}, {ci_g[1]:.3f}]")
        print(f"   'Say something true.'  (truth-frame) {gap_truth:8.3f}")
        if gap_h1 is not None:
            print(f"    [anchor] H1 {PROBE_TRAIN_SET} gap, same probe   {gap_h1:8.3f}"
                  f"   bare/H1 = {bare_over_h1:.3f}")
        print(f"    position on lie->truth axis: {pos:.3f}  "
              f"CI [{ci_p[0]:.3f}, {ci_p[1]:.3f}]  "
              f"(0 = at lie-frame, 1 = at truth-frame)")
        print(f"    lie-frame effect vs neutral:   {lie_effect:+8.3f}  "
              f"CI [{ci_l[0]:+.3f}, {ci_l[1]:+.3f}]  "
              f"({100 * lie_effect / gap_bare:+.1f}%)")
        print(f"    truth-frame effect vs neutral: {truth_effect:+8.3f}  "
              f"CI [{ci_t[0]:+.3f}, {ci_t[1]:+.3f}]  "
              f"({100 * truth_effect / gap_bare:+.1f}%)")
        # The first three branches assume neutral falls BETWEEN the two framed
        # gaps. It does not — see the module docstring — so the out-of-interval
        # cases are handled first rather than being folded into the nearest
        # in-interval branch, which is what produced a wrong verdict string.
        if pos > 1:
            verdict = ("BOTH FRAMES COMPRESS: the neutral gap exceeds both framed "
                       "gaps, so neither frame amplifies; the lie-frame compresses "
                       f"{lie_effect / truth_effect:.2f}x as much as the truth-frame")
        elif pos < 0:
            verdict = ("BOTH FRAMES AMPLIFY: the neutral gap is below both framed "
                       "gaps, so neither frame compresses")
        elif pos > 0.75:
            verdict = "the LIE-frame compresses (neutral sits at the truth-frame end)"
        elif pos < 0.25:
            verdict = "the TRUTH-frame amplifies (neutral sits at the lie-frame end)"
        else:
            verdict = "BOTH: the lie-frame compresses and the truth-frame amplifies"
        print(f"    => {verdict}\n")

        results[p] = {
            "gap_bare": gap_bare, "gap_bare_ci95": ci_g,
            "gap_lie_frame": gap_lie, "gap_truth_frame": gap_truth,
            "position_on_lie_to_truth_axis": pos,
            "position_ci95": ci_p,
            "lie_frame_effect_vs_neutral": lie_effect,
            "lie_frame_effect_ci95": ci_l,
            "lie_frame_effect_pct": 100 * lie_effect / gap_bare,
            "truth_frame_effect_vs_neutral": truth_effect,
            "truth_frame_effect_ci95": ci_t,
            "truth_frame_effect_pct": 100 * truth_effect / gap_bare,
            "h1_anchor_gap": gap_h1,
            "bare_over_h1_anchor": bare_over_h1,
            "verdict": verdict,
        }

    out = {
        "model": config.MODEL_NAME,
        "layer": config.chosen_layer(),
        "n_statements": int(grid.numel()),
        "n_facts": n_facts,
        "probe_train_set": PROBE_TRAIN_SET,
        "retrained": False,
        "bootstrap": {"n": N_BOOTSTRAP, "unit": "base_fact", "seed": SEED},
        "statistic_note": ("The truth GAP is centering-invariant (a difference of "
                           "class means), which is why it carries the comparison: "
                           "the bare set has a different global mean from the "
                           "framed set and there is no shared principled zero."),
        "anchor_note": ("h1_anchor_gap is the same frozen probe's true/false gap on "
                        "the 1496 H1 `cities` statements — the distribution it was "
                        "fitted on. It is a sanity check on the bare number, not a "
                        "condition of the design: bare/H1 near 1 means the neutral "
                        "gap is normal for this probe, so 'both frames compress' "
                        "cannot be explained by the H2 statement set being unusual."),
        "confound_note": ("The bare condition is in-distribution for the probe and "
                          "both framed conditions are not, so 'both frames compress' "
                          "is not cleanly separable from 'any prefix moves activations "
                          "off the probe's fitted distribution'. The lie-vs-truth "
                          "frame contrast is unaffected — both carry a prefix — and "
                          "remains the design's clean comparison."),
        "probes": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["extract", "analyze", "all"], default="all")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    if args.phase in ("extract", "all"):
        phase_extract()
    if args.phase in ("analyze", "all"):
        phase_analyze()


if __name__ == "__main__":
    main()
