# Cluster run (Gemma-2-9B on Misha)

Ordered so that nothing network-bound or CPU-bound ever holds a GPU.

| # | Step | Where | Script |
|---|------|-------|--------|
| 0 | Accept the Gemma license | **you, by hand** | — |
| 1 | Swap torch to the cu126 build | CPU batch job | `01_setup_env.sbatch` |
| 2 | Pre-download the 18.5 GB of weights | CPU batch job | `02_download_weights.sbatch` |
| 3 | `verify_model` smoke test | **interactive** `gpu_devel` | `03_verify.sh` |
| 4 | Extract activations (all 42 layers) | GPU batch job | `04a_extract_devel.sbatch` (preferred) or `04_extract.sbatch` |
| 5 | Layer sweep + commit chosen layer | CPU batch job | `05_sweep.sbatch` |
| 6 | Analyses (02, 03, 03b + aux) | CPU batch job | `06_analysis.sbatch` |
| 7 | NIE causal experiment | GPU batch job | `07_nie.sbatch` |
| 8 | H2 Phase 0 — `model_knows` filter | GPU batch job | `08_h2_model_knows.sbatch` |

**Queue note.** The `gpu` partition is badly backlogged — 462 pending against 117
running when this was written, projecting a ~2-day wait, and identical for a40,
l40s, a100 and h100 alike (the queue is binding, not the card type). `gpu_devel`
starts in about a minute and permits a 6-hour walltime, which extraction (minutes
of compute) fits inside comfortably. Hence `04a_extract_devel.sbatch` as the
default path; `04_extract.sbatch` is the roomier `gpu`-partition fallback.

Steps 1–2 are deliberately *not* GPU jobs: pip resolution and an 18.5 GB download
are network- and disk-bound, and doing either while holding an accelerator is the
idle-GPU pattern the YCRC monitors flag. Step 3 is deliberately interactive and
short — it is the gate that must pass before step 4 is allowed to queue.

## Step 0 — license (do this first, by hand)

Token auth is already done — `hf auth whoami` returns `gsuth5`, and the token
lives at `~/.cache/huggingface/token`, which compute nodes read over GPFS.

What is **not** done is the license grant. Accept it at
<https://huggingface.co/google/gemma-2-9b> while signed in as `gsuth5`.

Verify with an actual gated file fetch, not `model_info` — `model_info` returns
metadata for a gated repo you have no access to, so it reports success either way:

```
source .venv/bin/activate
HF_HOME=/gpfs/radev/project/krishnaswamy_smita/gss34/hf_cache \
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('google/gemma-2-9b','config.json'); print('LICENSE OK')"
```

A `GatedRepoError` means the grant has not gone through yet. Acceptance is
per-repo: `google/gemma-2-2b` is currently gated for this account too, so accept
that one as well if you ever want to re-run the 2B pipeline on the cluster (the
existing 2B JSON results do not need it).

This is the step that, if skipped, kills the job at model download after it has
already sat in the queue — which is the whole point of doing it first.

## Steps 1–2

```
sbatch cluster/01_setup_env.sbatch
# wait for it to finish, check the log, then:
sbatch cluster/02_download_weights.sbatch
```

## Job monitoring (LastPing)

`cluster/lastping.sh` is sourced by both extract scripts; `src/lastping.py` sends
the per-dataset heartbeats from inside the extraction loop. Put the key at
`~/.lastping_key` (`chmod 600`) — **it is not there yet**, and without it the
instrumentation disables itself and says so in the job log.

Compute-node egress is confirmed: `healthz` returns 200 from `r4516u05n01`.

Two deliberate deviations from `snippets/self_report.sh`, both verified:

- **The canonical trap always reports `exit_code: 0`.** In
  `trap '... \"exit_code\":'"$?"'}"' EXIT`, the `'"$?"'` closes the single quote,
  so `$?` expands when `trap` *runs* and freezes the status of whatever preceded
  it. A script exiting 42 reports 0. Keeping `$?` inside the single-quoted body
  defers expansion to trap-fire time and reports 42. Worth fixing upstream —
  every failed job in the dashboard is currently indistinguishable from a clean one.
- **SIGTERM sends no ping at all.** A shell killed by an untrapped signal skips
  the EXIT trap, so a job that hits its walltime — the case you most want to see —
  reports nothing and looks like a hang. Explicit `TERM`/`INT` traps send
  `143`/`130` and then clear the EXIT trap so the ping fires exactly once.

`|| true` on every call means instrumentation can never fail a run, but it also
means a firewall or bad key fails silently. `lp_init` therefore preflights
`healthz` and warns once in the job log.

## Step 3 — the gate

```
salloc -p gpu_devel -t 0:30:00 --gres=gpu:1 --mem=32G -c 4
bash cluster/03_verify.sh
exit
```

This must report 42 layers, d_model 3584, the `blocks.{n}.hook_resid_post` hook
present at both layer 0 and layer 41, a real all-layer extraction returning
`(3, 42, 3584)`, and peak VRAM comfortably under the card. Do not run step 4
until it does.

It also checks two things that are cheap only while the model is already loaded:

- **Host RAM, not VRAM, is the binding constraint here.** TransformerLens loads
  the HF checkpoint, applies weight processing (LayerNorm folding, centering),
  *then* moves to device — transiently holding close to two copies of 18.5 GB
  against gpu_devel's 32 GB cap. This fails during load, not during extraction,
  so the gate is what catches it. The script prints peak RSS and warns above
  26 GiB. If it OOMs, move to the roomier `gpu` partition and accept the queue.
  Do **not** reach for `from_pretrained_no_processing` — it changes the
  activations and breaks comparability with the 2B numbers.
- **Readout tokenization.** The NIE experiment scores `P(" TRUE") - P(" FALSE")`
  on the token after `This statement is:`. `aux_nie_intervention.true_false_ids`
  took `to_tokens(...)[0, 0]` — the *first* token — which silently returns a
  fragment if those words are multi-token in this vocab, corrupting every causal
  number in a way that reads as a weak-but-plausible effect. That function now
  asserts single-token instead, and the gate prints the tokenization plus the
  top-5 actual continuations and how much probability mass the two labels hold.
  Gemma-2-9B has its own vocab, so this is re-checked at 9B rather than assumed
  from 2B. If the labels hold under 10% of the mass, the readout is weak and the
  NIE ceiling is low for that reason alone — which is the likely story behind the
  2B run's `PD+ = -0.065`, and decides whether a small 9B NIE means "no causal
  effect" or "bad readout".

## Steps 4–5

```
sbatch cluster/04a_extract_devel.sbatch
sbatch --dependency=afterok:<extract_jobid> cluster/05_sweep.sbatch
```

Then read `results/layer_sweep.json` for the 9B `chosen_layer`. Nothing needs
editing afterwards: every downstream script derives its layer from that file via
`src/config.py`.

## Step 8 — H2 Phase 0 (`model_knows`)

```
sbatch cluster/08_h2_model_knows.sbatch          # GPU: 200 forward passes
# then, on the login node, once you have looked at the distribution:
python scripts/h2_00_model_knows.py --phase filter --threshold <T>
```

H2 runs on **9B only** (`h2plan.txt` records why). Phase 0 asks which base facts
the model reliably knows under neutral, frame-free questioning, because framing
the model to assert the negation of a fact it does not hold is not inducing a
lie — such a fact would add noise to the conflict cells while looking like a
real trial.

The two phases are split for the same reason `01` splits extract from sweep: the
margin threshold is supposed to be set *from* the distribution, so it cannot be
chosen before the GPU job runs, and a retune must not re-queue that job. The GPU
phase writes per-statement margins to
`results/<model>/h2_model_knows_margins.json`; the CPU phase applies a threshold,
prints the attrition ladder, writes `model_knows` into `data/induced_lies.csv`
and the decision to `results/<model>/h2_model_knows.json`. The margins file is
the source of truth — the CSV column is derived and regenerable at any threshold.

Note the readout differs from the NIE experiment's: the neutral prompt ends at
`Answer:` rather than `This statement is:`, so the `" TRUE"` / `" FALSE"`
single-token assertion is re-run and the top-5 continuations are printed for two
example prompts. Same failure mode as before — unspaced ids would score a
continuation the model never emits, and every margin would read as "the model
doesn't know this fact" rather than as a bug.
