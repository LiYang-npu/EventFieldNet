"""Conservative phase accounting independent of pre-evaluation checkpoints."""

import json
import math
from pathlib import Path


def recovered_seconds(path, identity, checkpoint_seconds, limit):
    checkpoint_seconds = float(checkpoint_seconds)
    if not math.isfinite(checkpoint_seconds) or checkpoint_seconds < 0:
        raise RuntimeError("invalid checkpoint budget")
    path = Path(path)
    if not path.exists():
        if checkpoint_seconds > 0:
            raise RuntimeError("checkpoint has no matching durable budget ledger")
        return 0.0
    saved = json.loads(path.read_text())
    if saved.get("identity") != identity:
        raise RuntimeError("budget ledger identity mismatch")
    used = float(saved["active_seconds"])
    if not math.isfinite(used) or used < 0:
        raise RuntimeError("invalid budget ledger elapsed time")
    # A killed process cannot report its final elapsed time. Never grant that
    # unaccounted interval again: require a newly reviewed phase budget.
    if saved.get("in_flight"):
        if limit <= 0:
            raise RuntimeError("interrupted phase without a finite time budget")
        used = max(used, float(limit))
    return max(float(checkpoint_seconds), used)


def ledger_payload(identity, elapsed, in_flight):
    elapsed = float(elapsed)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("invalid elapsed time")
    return {
        "identity": identity,
        "active_seconds": elapsed,
        "in_flight": bool(in_flight),
    }
