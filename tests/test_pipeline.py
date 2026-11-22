import numpy as np
import pandas as pd
import pytest

from src.alert import replay
from src.config import CAP, RAW, SENSORS, W
from src.data import add_labels
from src.evaluate import nasa_score, oof_predictions
from src.features import SEQ_FEATURES, add_lags
from src.model import make_baseline
from src.scorer import Scorer


def fake_engines(lives=(5, 8, 140), seed=0):
    """Small made-up training data: one row per cycle, random sensor readings."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, life in enumerate(lives, start=1):
        for cycle in range(1, life + 1):
            row = {"unit": unit, "cycle": cycle}
            row.update({c: rng.normal() for c in SENSORS})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------- labels ----------

def test_labels_count_down_to_zero_and_are_capped():
    df = add_labels(fake_engines())
    e1 = df[df["unit"] == 1]
    assert list(e1["RUL"]) == [4, 3, 2, 1, 0]
    assert (df.groupby("unit")["RUL"].min() == 0).all()
    long_engine = df[df["unit"] == 3]                  # lives 140 cycles
    assert long_engine["RUL"].iloc[0] == 139
    assert long_engine["cycles_left_capped"].iloc[0] == CAP
    assert df["cycles_left_capped"].max() == CAP


# ---------- lag features ----------

def test_lag_l_is_the_reading_l_cycles_earlier():
    df = add_lags(fake_engines())
    e3 = df[df["unit"] == 3].set_index("cycle")
    for c in SENSORS[:3]:
        for cycle in [40, 80, 140]:
            for lag in [0, 1, 5, W - 1]:
                assert e3.loc[cycle, f"{c}_lag{lag}"] == e3.loc[cycle - lag, c]


def assert_no_future_leakage(lag_fn):
    """Delete everything after a row; that row's inputs must not change."""
    full = lag_fn(fake_engines())
    raw = fake_engines()
    for unit, cycle in [(2, 4), (3, 31), (3, 100)]:
        cut = raw[(raw["unit"] != unit) | (raw["cycle"] <= cycle)]
        short = lag_fn(cut)
        a = full[(full["unit"] == unit) & (full["cycle"] == cycle)][SEQ_FEATURES]
        b = short[(short["unit"] == unit) & (short["cycle"] == cycle)][SEQ_FEATURES]
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_lags_never_use_future_rows():
    assert_no_future_leakage(add_lags)


def test_leakage_check_catches_a_future_reading():
    """Negative control: a lag function that peeks one cycle ahead must fail the same check."""
    def leaky_add_lags(d):
        d = add_lags(d)
        for c in SENSORS:
            # shift(-1) = the NEXT cycle's reading; at an engine's last row there is none, so keep its own
            d[f"{c}_lag0"] = d.groupby("unit")[c].shift(-1).fillna(d[c])
        return d

    with pytest.raises(AssertionError):
        assert_no_future_leakage(leaky_add_lags)


def test_lags_stay_inside_one_engine():
    df = add_lags(fake_engines())
    first_of_engine_2 = df[(df["unit"] == 2) & (df["cycle"] == 1)].iloc[0]
    for c in SENSORS:
        # no history yet: every lag repeats its own first reading, never engine 1's
        assert first_of_engine_2[f"{c}_lag1"] == first_of_engine_2[c]
        assert first_of_engine_2[f"{c}_lag{W - 1}"] == first_of_engine_2[c]


def test_early_cycles_are_padded_with_oldest_reading():
    df = add_lags(fake_engines())
    e3 = df[df["unit"] == 3].set_index("cycle")
    c = SENSORS[0]
    for lag in range(3, W):
        assert e3.loc[3, f"{c}_lag{lag}"] == e3.loc[1, c]


def test_add_lags_twice_gives_same_result():
    once = add_lags(fake_engines())
    twice = add_lags(once)
    assert list(once.columns) == list(twice.columns)
    pd.testing.assert_frame_equal(once, twice)


def test_no_missing_values_in_inputs():
    df = add_lags(fake_engines())
    assert not df[SEQ_FEATURES].isna().any().any()


# ---------- the answer is never an input ----------

def test_answer_columns_are_not_inputs():
    for col in ["unit", "RUL", "cycles_left_capped", "failure_cycle", "RUL_cap100", "RUL_cap150"]:
        assert col not in SEQ_FEATURES
    assert len(SEQ_FEATURES) == len(SENSORS) * W + 1


# ---------- split by engine ----------

def test_each_engine_is_in_exactly_one_fold():
    df = add_lags(add_labels(fake_engines(lives=(30, 40, 50, 60, 70, 80, 90, 100, 110, 120))))
    _, fold = oof_predictions(make_baseline, df, SEQ_FEATURES, "cycles_left_capped")
    folds_per_engine = pd.Series(fold).groupby(df["unit"].values).nunique()
    assert (folds_per_engine == 1).all()
    assert (np.asarray(fold) >= 0).all()          # every row was predicted


# ---------- NASA score ----------

def test_nasa_score_values():
    assert nasa_score(np.array([0.0])) == 0
    assert nasa_score(np.array([10.0])) == pytest.approx(np.e - 1)    # 10 cycles late
    assert nasa_score(np.array([-13.0])) == pytest.approx(np.e - 1)   # 13 cycles early
    assert nasa_score(np.array([10.0])) > nasa_score(np.array([-10.0]))  # late is worse


# ---------- alert replay ----------

def test_replay_counts_failures_and_wasted_cycles():
    rows = []
    # engine 1: alert fires with 15 cycles left -> serviced, 15 - 10 = 5 wasted
    # engine 2: alert fires with 5 cycles left -> too late for a 10-cycle lead: failure
    # engine 3: prediction never drops to k -> failure
    for unit, fire_at_rul in [(1, 15), (2, 5), (3, None)]:
        for rul in range(30, -1, -1):
            fired = fire_at_rul is not None and rul <= fire_at_rul
            rows.append({"unit": unit, "cycle": 31 - rul, "RUL": rul,
                         "oof_pred": 10.0 if fired else 100.0})
    df = pd.DataFrame(rows)
    r = replay(df, k=20, lead=10).set_index("unit")
    assert not r.loc[1, "failed"] and r.loc[1, "wasted"] == 5
    assert r.loc[2, "failed"]
    assert r.loc[3, "failed"]


# ---------- live scorer: input checks ----------

def test_scorer_rejects_values_that_are_not_finite_numbers():
    row = fake_engines(lives=(1,)).iloc[0].to_dict()
    row.update({"set1": 0.0, "set2": 0.0, "set3": 100.0, **{f"s{i}": 1.0 for i in range(1, 22)}})
    row.update({"s2": float("nan"), "s3": float("inf"), "s4": "abc"})
    s = Scorer(model=None)   # rejected before the model is ever used
    with pytest.raises(ValueError, match=r"not finite numbers: \['s2', 's3', 's4'\]"):
        s.score(row)
    assert s.history == {} and s.seen == {}   # nothing was stored


# ---------- real numbers: only where the data exists (skipped on GitHub) ----------

@pytest.mark.skipif(not (RAW / "train_FD001.txt").exists(), reason="NASA data not available")
def test_final_model_reproduces_test_score():
    from sklearn.metrics import root_mean_squared_error

    from src.data import load_rul, load_test, load_train
    from src.evaluate import last_cycles
    from src.model import make_mlp

    train = add_lags(add_labels(load_train()))
    test = load_test()
    last = last_cycles(add_lags(test), load_rul(test))
    model = make_mlp(seed=0).fit(train[SEQ_FEATURES], train["cycles_left_capped"])
    rmse = root_mean_squared_error(last["RUL"], model.predict(last[SEQ_FEATURES]))
    assert rmse == pytest.approx(14.94, abs=0.05)
