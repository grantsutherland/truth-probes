"""H2 PHASE 4 — separating truth from frame over the full 2x2.

The conflict cells alone cannot do this. True-under-lie-frame and
false-under-truth-frame differ in truth AND in frame at once, so those two groups
are collinear and their AUROC — 1.0000 for both probes — is consistent with the
probe tracking truth, tracking frame, or any mixture of the two. That ceiling is
also exactly why the conflict-cell AUROC carries no information about the frame
effect. It is not the finding and is not reported as one.

The full 2x2 IS estimable. Across all 392 rows label, frame and their product are
pairwise independent by construction (98 per cell), so

    score ~ label + frame + label:frame        (label, frame coded 0/1)

identifies all three. The dependent variable is the CONTINUOUS signed projection
onto the frozen probe direction — not AUROC, not a binary prediction, because the
question is about magnitude.

READING THE COEFFICIENTS (0/1 coding, so "main effects" are conditional):
    b_label  truth signal at frame=0, i.e. the truth gap under "Say something
             false."  — the lie-frame gap.
    b_frame  effect of the frame at label=0, i.e. on FALSE statements only. With
             a nonzero interaction there is no single frame main effect; the
             average across label values is reported separately.
    b_int    the quantity of interest: how much the frame changes the MAGNITUDE
             of the truth signal. Truth gap under "Say something true." is
             b_label + b_int, so compression = b_int / (b_label + b_int).

THE REGRESSION AND THE MARGIN TABLE AGREE ALGEBRAICALLY, NOT INDEPENDENTLY. The
model is saturated (4 parameters, 4 cells) on a balanced design, so OLS returns
exact linear combinations of the cell means and the implied compression must
equal the margin-table compression to floating point. Agreement is a correctness
check on the arithmetic, not corroborating evidence. What the regression actually
adds is the confidence intervals.

BOOTSTRAP BY BASE FACT, NOT BY ROW. The four cells of one fact share a statement
and a city and are strongly correlated; resampling rows would treat 392
correlated observations as independent and understate every CI. The 98 facts are
resampled with replacement, all four of their cells travelling together, and the
model is refit on each resample. Same clustering issue as the NIE bootstrap.
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

MARGINS_JSON = os.path.join(config.RESULTS_DIR, "h2_model_knows_margins.json")
CONFLICT_JSON = os.path.join(config.RESULTS_DIR, "h2_conflict.json")
OUT_JSON = os.path.join(config.RESULTS_DIR, "h2_decompose.json")

# Cell order within a fact, fixed so the design matrix is constant across
# bootstrap resamples: (label, frame).
CELLS = [(0, 0), (1, 0), (0, 1), (1, 1)]
CELL_NAMES = ["false stmt / lie frame (aligned)",
              "true  stmt / lie frame (CONFLICT)",
              "false stmt / truth frame (CONFLICT)",
              "true  stmt / truth frame (aligned)"]


def design_matrix(n_facts):
    """[1, label, frame, label*frame], fact-major with CELLS repeating."""
    rows = []
    for _ in range(n_facts):
        for lab, frm in CELLS:
            rows.append([1.0, float(lab), float(frm), float(lab * frm)])
    return torch.tensor(rows, dtype=torch.double)


def spearman(x, y):
    """Rank correlation, for the variance thread (monotone, outlier-tolerant)."""
    def rank(v):
        order = torch.argsort(v)
        r = torch.empty(len(v), dtype=torch.double)
        r[order] = torch.arange(1, len(v) + 1, dtype=torch.double)
        return r
    rx, ry = rank(x.double()), rank(y.double())
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx @ ry) / (rx.norm() * ry.norm()))


def main():
    torch.manual_seed(SEED)
    acts, labels, frame, meta = data.load_induced_lies()
    layers = meta["layers"]
    idx = layers.index(config.chosen_layer())
    facts = meta["base_fact"]
    statements = meta["statement"]

    # Index rows into a (n_facts, 4) grid in the fixed CELLS order. Errors if any
    # fact is missing a cell — the whole identification argument rests on the
    # design being complete and balanced.
    order = {}
    for i, (f, l, fr) in enumerate(zip(facts, labels.tolist(), frame.tolist())):
        order.setdefault(f, {})[(int(l), int(fr))] = i
    fact_ids = sorted(order)
    grid = []
    for f in fact_ids:
        if set(order[f]) != set(CELLS):
            raise ValueError(f"fact {f!r} is missing cells: has {sorted(order[f])}")
        grid.append([order[f][c] for c in CELLS])
    grid = torch.tensor(grid)                      # (n_facts, 4) row indices
    n_facts = len(fact_ids)
    print(f"{n_facts} base facts x 4 cells = {grid.numel()} rows "
          f"(design complete and balanced)")

    X = design_matrix(n_facts)
    # OLS projector. X is IDENTICAL for every fact-level resample (each fact
    # contributes the same four design rows), so the pseudo-inverse is computed
    # once and the bootstrap is a matrix multiply.
    P = torch.linalg.pinv(X)                       # (4, 4*n_facts)

    with open(MARGINS_JSON) as f:
        phase0 = {r["statement"]: r["primary_margin"]
                  for r in json.load(f)["statements"]}
    with open(CONFLICT_JSON) as f:
        conflict = json.load(f)

    results = {}
    for p in PROBE_TYPES:
        probe, pmeta = data.load_probe(p, PROBE_TRAIN_SET)
        assert pmeta["model_name"] == config.MODEL_NAME
        assert pmeta["layer"] == config.chosen_layer()
        scores = (acts[:, idx, :] @ probe.direction).double()

        y = scores[grid.reshape(-1)]               # fact-major, CELLS order
        beta = P @ y
        b0, b_lab, b_frm, b_int = [float(v) for v in beta]

        gap_lie = b_lab                            # truth gap at frame=0
        gap_truth = b_lab + b_int                  # truth gap at frame=1
        compression = b_int / gap_truth
        avg_frame = b_frm + b_int / 2              # frame effect averaged over label

        # Fact-clustered bootstrap: resample facts, carry all four cells.
        g = torch.Generator().manual_seed(SEED)
        draws = []
        yg = scores[grid]                          # (n_facts, 4)
        for _ in range(N_BOOTSTRAP):
            sel = torch.randint(n_facts, (n_facts,), generator=g)
            draws.append(P @ yg[sel].reshape(-1).double())
        B = torch.stack(draws)                     # (n_boot, 4)
        comp_draws = B[:, 3] / (B[:, 1] + B[:, 3])

        def ci(v):
            return [float(torch.quantile(v, 0.025)), float(torch.quantile(v, 0.975))]

        ci_b = {name: ci(B[:, k]) for k, name in
                enumerate(["intercept", "label", "frame", "label:frame"])}
        ci_comp = ci(comp_draws)

        # Cross-check against Phase 3's margin table. Saturated model on a
        # balanced design => this must agree to floating point, so a mismatch
        # means an arithmetic or indexing error, not a substantive disagreement.
        c3 = conflict["probes"][p]["cells"]
        gt = (c3["aligned/frame=true(true stmt)"]["mean_margin"]
              - c3["conflict/frame=true(false stmt)"]["mean_margin"])
        gl = (c3["conflict/frame=false(true stmt)"]["mean_margin"]
              - c3["aligned/frame=false(false stmt)"]["mean_margin"])
        margin_comp = 1 - gl / gt
        # Tolerance is 1e-6, not 0: Phase 3 accumulated its cell means in
        # float32 while this fit runs in float64, which moves the scores by
        # ~5e-5 and the compression by ~1e-7. An actual indexing error would
        # shift this by order 1e-2 or more, so the check still has teeth.
        agree = abs(margin_comp - compression) < 1e-6

        print(f"\n{'=' * 72}\n[{p}]  score ~ label + frame + label:frame   "
              f"(n={grid.numel()}, clustered on {n_facts} facts)")
        print(f"  {'coefficient':<14}{'estimate':>10}  {'95% CI (fact-clustered)':>26}"
              f"   {'as % of b_label':>16}")
        for name, est in (("intercept", b0), ("label", b_lab),
                          ("frame", b_frm), ("label:frame", b_int)):
            lo, hi = ci_b[name]
            pct = "" if name in ("intercept", "label") else f"{100 * est / b_lab:+.1f}%"
            print(f"  {name:<14}{est:>10.3f}  [{lo:>10.3f}, {hi:>10.3f}]"
                  f"   {pct:>16}")
        print(f"\n  truth gap under 'Say something false.' (b_label)         "
              f"{gap_lie:8.3f}")
        print(f"  truth gap under 'Say something true.'  (b_label+b_int)    "
              f"{gap_truth:8.3f}")
        print(f"  COMPRESSION b_int/(b_label+b_int)   {100 * compression:6.2f}%   "
              f"95% CI [{100 * ci_comp[0]:.2f}%, {100 * ci_comp[1]:.2f}%]")
        print(f"  margin-table compression {100 * margin_comp:.2f}%  -> "
              f"{'AGREES (algebraic, saturated model)' if agree else 'MISMATCH — BUG'}")
        if not agree:
            raise RuntimeError("regression and margin table disagree; check indexing")
        print(f"  frame effect on FALSE statements (b_frame)   {b_frm:+.3f}")
        print(f"  frame effect on TRUE  statements (b_frame+b_int) "
              f"{b_frm + b_int:+.3f}")
        print(f"  frame effect averaged over label             {avg_frame:+.3f}"
              f"   ({100 * avg_frame / b_lab:+.1f}% of b_label)")

        # ---- attenuation split by truth value, IN RAW SCORE UNITS ----
        #
        # A percentage compression needs a zero point, and percentages are the
        # place a floor/ceiling effect hides: a bigger relative drop from a
        # bigger baseline can be trivial. So the asymmetry is established in RAW
        # units first (centering-invariant differences of cell means), and the
        # percentages are reported afterwards against an explicit boundary — the
        # global mean over all 392 rows, which is label-balanced.
        #
        # Movement is measured TOWARD THE BOUNDARY, so both classes are positive
        # and directly comparable. By construction they sum to b_int.
        centered = scores - scores.mean()
        cg = centered[grid]                        # (n_facts, 4) in CELLS order
        f_lie, t_lie, f_true, t_true = cg[:, 0], cg[:, 1], cg[:, 2], cg[:, 3]

        toward_true = t_true - t_lie                # true stmts pulled down
        toward_false = f_lie - f_true               # false stmts pulled up
        base_true, base_false = t_true, -f_true     # distance from boundary

        def pack(v):
            return float(v.mean())

        asym_raw = pack(toward_true) - pack(toward_false)
        pct_true = pack(toward_true) / pack(base_true)
        pct_false = pack(toward_false) / pack(base_false)

        g2 = torch.Generator().manual_seed(SEED)
        tt_d, tf_d, a_draws, pt_draws, pf_draws = [], [], [], [], []
        for _ in range(N_BOOTSTRAP):
            s = torch.randint(n_facts, (n_facts,), generator=g2)
            tt, tf = toward_true[s].mean(), toward_false[s].mean()
            tt_d.append(tt)
            tf_d.append(tf)
            a_draws.append(tt - tf)
            pt_draws.append(tt / base_true[s].mean())
            pf_draws.append(tf / base_false[s].mean())

        def ci2(v):
            v = torch.stack(v).double() if isinstance(v, list) else v.double()
            return [float(torch.quantile(v, 0.025)), float(torch.quantile(v, 0.975))]

        ci_tt, ci_tf, ci_a = ci2(tt_d), ci2(tf_d), ci2(a_draws)
        ci_pt, ci_pf = ci2(pt_draws), ci2(pf_draws)

        print(f"\n  ATTENUATION BY TRUTH VALUE "
              f"(movement TOWARD the boundary under the lie-frame)")
        print(f"    {'':18}{'raw':>9}{'95% CI (raw)':>22}"
              f"{'baseline':>11}{'% of baseline':>16}{'95% CI (%)':>20}")
        print(f"    {'true statements':18}{pack(toward_true):>9.3f}"
              f"{f'[{ci_tt[0]:.3f}, {ci_tt[1]:.3f}]':>22}{pack(base_true):>11.3f}"
              f"{100 * pct_true:>15.1f}%"
              f"{f'[{100*ci_pt[0]:.1f}%, {100*ci_pt[1]:.1f}%]':>20}")
        print(f"    {'false statements':18}{pack(toward_false):>9.3f}"
              f"{f'[{ci_tf[0]:.3f}, {ci_tf[1]:.3f}]':>22}{pack(base_false):>11.3f}"
              f"{100 * pct_false:>15.1f}%"
              f"{f'[{100*ci_pf[0]:.1f}%, {100*ci_pf[1]:.1f}%]':>20}")
        print(f"    asymmetry (true - false), RAW units: {asym_raw:+.3f}  "
              f"95% CI [{ci_a[0]:+.3f}, {ci_a[1]:+.3f}]  "
              f"{'SIGNIFICANT' if ci_a[0] * ci_a[1] > 0 else 'NOT SIGNIFICANT'}")
        print(f"    raw CIs overlap: "
              f"{'yes' if ci_tt[0] <= ci_tf[1] and ci_tf[0] <= ci_tt[1] else 'no'}"
              f"   pct CIs overlap: "
              f"{'yes' if ci_pt[0] <= ci_pf[1] and ci_pf[0] <= ci_pt[1] else 'no'}")
        print(f"    floor/ceiling check: baselines are {pack(base_true):.2f} "
              f"(true) vs {pack(base_false):.2f} (false) — "
              f"ratio {pack(base_true) / pack(base_false):.3f}")

        # Does the false-side attenuation concentrate in the MOST-false items?
        # Ties the magnitude-spread observation to the compression if it holds.
        rho_false = spearman(toward_false, f_true)      # baseline (more neg = more false)
        rho_true = spearman(toward_true, t_true)
        m_false = torch.tensor([phase0[statements[i]] for i in grid[:, 2].tolist()])
        rho_false_p0 = spearman(toward_false, m_false)
        print(f"    concentration: rho(false attenuation, its truth-frame "
              f"baseline) {rho_false:+.3f}")
        print(f"                   rho(false attenuation, Phase 0 margin) "
              f"{rho_false_p0:+.3f}")
        print(f"                   rho(true  attenuation, its baseline) "
              f"{rho_true:+.3f}")

        attenuation = {
            "toward_boundary_true_raw": pack(toward_true),
            "toward_boundary_false_raw": pack(toward_false),
            "asymmetry_true_minus_false_raw": asym_raw,
            "asymmetry_ci95": ci2(a_draws),
            "asymmetry_significant": bool(ci2(a_draws)[0] * ci2(a_draws)[1] > 0),
            "baseline_true": pack(base_true),
            "baseline_false": pack(base_false),
            "toward_boundary_true_ci95": ci_tt,
            "toward_boundary_false_ci95": ci_tf,
            "pct_true": pct_true, "pct_true_ci95": ci_pt,
            "pct_false": pct_false, "pct_false_ci95": ci_pf,
            "rho_false_attenuation_vs_baseline": rho_false,
            "rho_false_attenuation_vs_phase0_margin": rho_false_p0,
            "rho_true_attenuation_vs_baseline": rho_true,
        }

        # ---- the variance thread (logged, not chased) ----
        var_rows = {}
        for k, (lab, frm) in enumerate(CELLS):
            sel = grid[:, k]
            s = scores[sel]
            m = torch.tensor([phase0[statements[i]] for i in sel.tolist()])
            var_rows[CELL_NAMES[k]] = {
                "n": len(sel), "sd": float(s.std()),
                "spearman_score_vs_phase0_margin": spearman(s, m),
            }
        print(f"\n  variance structure (score sd by cell, and rank-correlation "
              f"with the Phase 0 neutral readout margin):")
        for name, v in var_rows.items():
            print(f"    {name:36} sd {v['sd']:6.3f}   "
                  f"rho(score, phase0 margin) {v['spearman_score_vs_phase0_margin']:+.3f}")

        results[p] = {
            "coefficients": {"intercept": b0, "label": b_lab,
                             "frame": b_frm, "label:frame": b_int},
            "ci95_fact_clustered": ci_b,
            "truth_gap_lie_frame": gap_lie,
            "truth_gap_truth_frame": gap_truth,
            "compression": compression,
            "compression_ci95": ci_comp,
            "compression_from_margin_table": margin_comp,
            "compression_agrees_algebraically": agree,
            "frame_effect_on_false_statements": b_frm,
            "frame_effect_on_true_statements": b_frm + b_int,
            "frame_effect_averaged_over_label": avg_frame,
            "coefficients_as_fraction_of_label": {
                "frame": b_frm / b_lab, "label:frame": b_int / b_lab,
                "frame_averaged": avg_frame / b_lab},
            "variance_structure": var_rows,
            "attenuation_by_truth_value": attenuation,
        }

    out = {
        "model": config.MODEL_NAME,
        "layer": config.chosen_layer(),
        "probe_train_set": PROBE_TRAIN_SET,
        "retrained": False,
        "n_rows": int(grid.numel()),
        "n_facts": n_facts,
        "coding": "label, frame in {0,1}; frame=1 means 'Say something true.'",
        "bootstrap": {"n": N_BOOTSTRAP, "unit": "base_fact", "seed": SEED,
                      "note": ("resampled by fact, not by row: the four cells of "
                               "a fact share a statement and are correlated, so "
                               "row resampling would understate every CI")},
        "headline": ("The deceptive frame ATTENUATES the truth signal without "
                     "reversing it. Conflict-cell AUROC is NOT the finding: "
                     "truth and frame are collinear across those two cells, so "
                     "its ceiling value is uninformative about the frame."),
        "probes": results,
        "interpretive_limit": conflict["interpretive_limit"],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
