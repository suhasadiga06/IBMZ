"""JSON-safe values: NaN/inf (e.g. a precision when nothing reached a level) become null."""
from __future__ import annotations

import math


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) or type(x).__module__ == "numpy":
        try:
            f = float(x)
        except (TypeError, ValueError):
            return x
        if not math.isfinite(f):
            return None
        return int(f) if type(x).__name__.startswith(("int", "uint")) else f
    return x
