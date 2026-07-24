"""Optional LastPing heartbeats from inside the extraction loop.

The sbatch wrapper (cluster/lastping.sh) sends the job-level start and exit
pings. It cannot send per-dataset progress, because the loop over datasets lives
inside 01's extract phase — so this module emits those.

Design constraints, in order of importance:
  1. It must never break a run. Every failure path is swallowed; a monitoring
     call that takes down a GPU job is worse than no monitoring.
  2. It must be a no-op when unconfigured, so the pipeline still runs locally
     and in tests with no key and no network.
  3. No new dependencies — urllib from the stdlib, not requests.

Configuration comes from the environment the sbatch script already exports:
LP_URL, LP_KEY, LP_RUN. If any is unset, every call here does nothing.
"""

import json
import os
import sys
import urllib.error
import urllib.request

TIMEOUT = 10

# Swallowing every error keeps monitoring from breaking a run, but it also hides
# a broken integration completely: the first version of this module sent fields
# the API rejects (IngestHeartbeat permits only run_id/step/metric/text), every
# call 422'd, and the run looked fine while producing no heartbeats at all. So
# failures are still non-fatal, but the FIRST one is reported once to stderr —
# fail-open, not fail-silent.
_warned = False


def enabled():
    return bool(os.environ.get("LP_URL") and os.environ.get("LP_KEY")
                and os.environ.get("LP_RUN"))


def ping(endpoint, **fields):
    """POST one event. Returns True if it landed, False otherwise — never raises.

    The return value is for callers that want to log it; ignoring it is fine and
    is what the extraction loop does.
    """
    if not enabled():
        return False

    url = f"{os.environ['LP_URL'].rstrip('/')}/{endpoint}"
    body = json.dumps({"run_id": os.environ["LP_RUN"], **fields}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {os.environ['LP_KEY']}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        # 422 = the payload does not match the endpoint's schema; 401 = bad key.
        # Both are integration bugs that would otherwise be invisible.
        _warn(f"{endpoint} -> HTTP {e.code}: {e.read()[:200].decode(errors='replace')}")
        return False
    except (urllib.error.URLError, OSError, ValueError) as e:
        # Network down, DNS failure, endpoint moved — non-fatal.
        _warn(f"{endpoint} -> {type(e).__name__}: {e}")
        return False


def _warn(msg):
    global _warned
    if not _warned:
        _warned = True
        print(f"lastping: {msg} (further failures silenced)", file=sys.stderr, flush=True)


def heartbeat(**fields):
    """Progress ping. Accepts only step (int), metric (str), text (str)."""
    return ping("heartbeat", **fields)
