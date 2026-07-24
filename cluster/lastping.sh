# LastPing job instrumentation — sourced by the extract sbatch scripts.
#
# Plain curl to an HTTPS endpoint; nothing to install. Compute-node egress to
# lastping-grant.fly.dev is confirmed working (healthz 200 from r4516u05n01).
#
# Every call ends in `|| true` so instrumentation can never fail the run. That
# is deliberate but it cuts both ways: a firewall, a bad key, or a typo'd URL
# all fail silently and you simply get no data. Hence the preflight below, which
# reports (without failing) whether the endpoint is actually reachable before
# the real work starts.
#
# Usage:
#   source cluster/lastping.sh
#   lp_init "gemma-9b-extract"      # sets the trap and sends start
#   lp heartbeat "{\"run_id\":\"$LP_RUN\",\"note\":\"...\"}"

LP_URL="${LP_URL:-https://lastping-grant.fly.dev/api/ingest}"
LP_KEYFILE="${LP_KEYFILE:-$HOME/.lastping_key}"
LP_RUN="${SLURM_JOB_ID:-manual-$(date +%s)}"
export LP_URL LP_RUN

if [ -r "$LP_KEYFILE" ]; then
  LP_KEY="$(cat "$LP_KEYFILE")"
else
  LP_KEY=""
  echo "lastping: no key at $LP_KEYFILE — instrumentation disabled for this run" >&2
fi
export LP_KEY

lp() {
  [ -n "$LP_KEY" ] || return 0
  curl -fsS -m 10 -H "Authorization: Bearer $LP_KEY" \
    -H "Content-Type: application/json" "$LP_URL/$1" -d "$2" >/dev/null || true
}

lp_init() {
  local name="${1:-unnamed}"
  if [ -n "$LP_KEY" ]; then
    # Preflight: the || true in lp() means a blocked endpoint is indistinguishable
    # from a successful ping. Say so once, loudly, in the job log.
    if curl -fsS -m 10 "${LP_URL%/api/ingest}/healthz" >/dev/null 2>&1; then
      echo "lastping: endpoint reachable, run_id=$LP_RUN"
    else
      echo "lastping: WARNING endpoint unreachable from $(hostname) — pings will be silently dropped" >&2
    fi
  fi

  # NOTE, deliberate deviation from snippets/self_report.sh:
  #
  #   trap 'lp exit "{... \"exit_code\":'"$?"'}"' EXIT     <-- the canonical form
  #
  # closes the single quote around $?, so $? expands when `trap` RUNS, not when
  # the trap FIRES. It captures the status of whatever preceded lp_init — 0 —
  # and bakes it in as a literal. Every job then reports exit_code 0, including
  # the failures you most want to hear about. Verified: the canonical form
  # reports 0 for a script exiting 42; the form below reports 42.
  #
  # Keeping $? inside the single-quoted body defers expansion to trap time. It
  # is expanded while building lp's argument, before lp runs and clobbers $?.
  trap 'lp exit "{\"run_id\":\"$LP_RUN\",\"exit_code\":$?}"' EXIT

  # SLURM-specific: hitting the walltime, or scancel, sends SIGTERM. A shell
  # killed by an untrapped signal terminates via that signal WITHOUT running the
  # EXIT trap — so the most interesting failure (the job that ran out of time)
  # would send no exit ping at all and be indistinguishable from a hang. Trap
  # the signals explicitly and exit normally, which then runs the EXIT trap.
  # IngestExit accepts run_id and exit_code ONLY — a "reason" field 422s and the
  # whole ping is dropped. 143 = 128+SIGTERM is itself the signal, so no field
  # is needed to say so.
  trap 'lp exit "{\"run_id\":\"$LP_RUN\",\"exit_code\":143}"; trap - EXIT; exit 143' TERM
  trap 'lp exit "{\"run_id\":\"$LP_RUN\",\"exit_code\":130}"; trap - EXIT; exit 130' INT

  lp start "{\"run_id\":\"$LP_RUN\",\"name\":\"$name\",\"external_id\":\"${SLURM_JOB_ID:-}\",\"host\":\"$(hostname)\"}"
}
