"""Live scoring: one engine, one flight at a time, the way it would run in service."""
import math
import threading

import pandas as pd

from src.config import COLS, K_ALERT, W
from src.features import SEQ_FEATURES, add_lags


class CycleOrderError(ValueError):
    """A flight arrived with a cycle that is not after the engine's last one."""


def _to_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class Scorer:
    def __init__(self, model):
        self.model = model
        self.history = {}   # unit -> its last W rows (dicts of floats), oldest first
        self.seen = {}      # unit -> how many flights it has sent in total
        self.latest = {}    # unit -> the last answer, for GET /engines/{unit}
        # One request at a time through check-predict-store: otherwise two flights for one engine could both
        # pass the cycle check and overwrite each other's history.
        self._lock = threading.Lock()

    def score(self, row):
        """row: dict with unit, cycle, set1-3, s1-s21. Returns the prediction for this flight."""
        missing = [c for c in COLS if c not in row]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        values = {c: _to_float(row[c]) for c in COLS}
        bad = [c for c, v in values.items() if v is None]
        if bad:
            raise ValueError(f"not finite numbers: {bad}")
        unit, cycle = row["unit"], values["cycle"]

        with self._lock:
            rows = self.history.get(unit, [])
            if rows and cycle <= rows[-1]["cycle"]:
                raise CycleOrderError(f"engine {unit}: cycle {row['cycle']} is not after its last cycle "
                                      f"{rows[-1]['cycle']:g}")

            # The lags reach back W - 1 cycles, so the last W rows are all add_lags ever needs.
            rows = (rows + [values])[-W:]
            # The same add_lags as training: build the stored rows' features and keep the newest row.
            x = add_lags(pd.DataFrame(rows, columns=COLS)).iloc[[-1]][SEQ_FEATURES]
            pred = float(self.model.predict(x)[0])

            # State changes only after a successful prediction, so a failed call leaves the engine as it was.
            self.history[unit] = rows
            self.seen[unit] = self.seen.get(unit, 0) + 1
            self.latest[unit] = {"unit": unit, "cycle": row["cycle"], "predicted_rul": pred,
                                 "alert": pred <= K_ALERT, "cycles_seen": self.seen[unit]}
            return self.latest[unit]
