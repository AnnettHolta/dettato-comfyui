import os

import numpy as np
import pytest


def weights_available(stage: str) -> bool:
    """True when the stage's weights can be loaded (DETTATO_WEIGHTS set, or a network / Hub cache)."""
    from dettato import hub

    try:
        hub.fetch(stage, "config.json")
        return True
    except Exception:  # noqa: BLE001
        return False


def tone(seconds=4.0, rate=16000, freqs=(220.0, 330.0), start=0.5):
    """A few seconds of two sine notes starting at `start` s (float32 mono)."""
    t = np.arange(int(seconds * rate)) / rate
    x = np.zeros_like(t)
    for f in freqs:
        x += 0.2 * np.sin(2 * np.pi * f * t) * (t >= start)
    return x.astype(np.float32)


needs = {s: pytest.mark.skipif(not weights_available(s), reason=f"{s} weights not available") for s in
         ("transcriber", "evidence", "grouper", "drums")}
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
