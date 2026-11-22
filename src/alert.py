"""Alert-threshold replay (notebook section 15).

Rule: raise a maintenance alert the first cycle an engine's predicted RUL is <= k.
Replayed on the TRAINING engines, using out-of-fold predictions (evaluate.oof_predictions): every prediction
comes from a model that never saw that engine, and train engines run to failure, so we know what would have
happened. The test set is not used here.
"""
import pandas as pd

from src.config import C_CYCLE, C_FAIL, FAIL_COSTS, KS, LEAD


def replay(df, k, lead=LEAD):
    """One row per engine: did it fail in service, and how many cycles of life were thrown away.

    df needs columns unit, cycle, RUL, oof_pred, sorted by unit and cycle (add_lags does this).
    """
    rows = []
    for unit, e in df.groupby("unit"):
        hit = e[e["oof_pred"] <= k]                   # cycles where the alert would fire
        if hit.empty or hit.iloc[0]["RUL"] < lead:    # never fired, or fired with too little time: failure
            rows.append((unit, True, 0))
        else:
            # maintenance happens `lead` cycles after the alert; the life left at that point is wasted
            rows.append((unit, False, int(hit.iloc[0]["RUL"]) - lead))
    return pd.DataFrame(rows, columns=["unit", "failed", "wasted"])


def cost(df, k, c_fail=C_FAIL, c_cycle=C_CYCLE):
    r = replay(df, k)
    return r["failed"].sum(), r["wasted"].sum(), r["failed"].sum() * c_fail + r["wasted"].sum() * c_cycle


def cost_table(df, c_fail=C_FAIL, c_cycle=C_CYCLE):
    return pd.DataFrame([(k, *cost(df, k, c_fail, c_cycle)) for k in KS],
                        columns=["k", "failures", "wasted cycles", "cost"])


def best_k(df):
    """For each assumed failure cost: the k with the lowest total cost, and its (failures, wasted, cost)."""
    out = {}
    for c_fail in FAIL_COSTS:
        k = min(KS, key=lambda k: cost(df, k, c_fail=c_fail)[2])   # ties go to the first (smallest) k, as in the notebook
        out[c_fail] = (k, *[int(v) for v in cost(df, k, c_fail=c_fail)])
    return out
