"""Model inputs (notebook sections 7-8). The only feature code: train and test both go through add_lags."""
import pandas as pd

from src.config import SENSORS, W


def add_lags(d):
    """Add one column per sensor per past cycle: {sensor}_lag{l} = the reading l cycles ago, same engine."""
    d = d.drop(columns=[c for c in d.columns if "_lag" in c])   # remove old lags so calling twice is safe
    d = d.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = d.groupby("unit")   # shift within one engine, never across two
    parts = []
    for c in SENSORS:
        lags = pd.DataFrame({f"{c}_lag{l}": g[c].shift(l) for l in range(W)})
        # An engine's first cycles have no reading l cycles ago, so the oldest lags (rightmost columns) are empty.
        # bfill(axis=1) finds nothing to their right and changes nothing; ffill(axis=1) then copies the
        # oldest available reading into them.
        parts.append(lags.bfill(axis=1).ffill(axis=1))
    return pd.concat([d] + parts, axis=1)


LAG_COLS = [f"{c}_lag{l}" for c in SENSORS for l in range(W)]
SEQ_FEATURES = LAG_COLS + ["cycle"]

# Columns deliberately left out (section 8)
CONSTANT = ["set3", "s1", "s5", "s10", "s16", "s18", "s19"]   # never change in train or test
NOISE = ["set1", "set2"]                                       # one flight condition in FD001
MAYBE = ["s6"]                                                 # 2 values, no effect in CV
NEVER = ["unit", "RUL", "cycles_left_capped", "failure_cycle",  # ID, the answers, the failure point
         "RUL_cap100", "RUL_cap150"]

# Run on import, so nothing can train on an answer column.
for col in CONSTANT + NOISE + MAYBE + NEVER:
    assert col not in SEQ_FEATURES, f"{col} must not be an input"
assert len(SEQ_FEATURES) == len(SENSORS) * W + 1 == 421
