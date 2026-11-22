"""Training/serving skew: scoring flight by flight must give exactly the batch predictions."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import root_mean_squared_error

from src.config import COLS, MODEL_FILE, RAW
from src.data import add_labels
from src.features import SEQ_FEATURES, add_lags
from src.model import make_mlp
from src.scorer import Scorer


def live_predictions(model, d):
    """Send every row through a fresh Scorer, in cycle order per engine."""
    s = Scorer(model)
    rows = d.sort_values(["unit", "cycle"]).to_dict("records")
    return np.array([s.score(r)["predicted_rul"] for r in rows])


def batch_predictions(model, d):
    return model.predict(add_lags(d)[SEQ_FEATURES])   # add_lags sorts by unit and cycle too


def fake_flights(lives=(5, 31, 60), seed=0):
    """Made-up engines with all 26 columns: shorter and longer than the 30-cycle window."""
    rng = np.random.default_rng(seed)
    rows = [{"unit": u, "cycle": c, **{col: rng.normal() for col in COLS[2:]}}
            for u, life in enumerate(lives, start=1) for c in range(1, life + 1)]
    return pd.DataFrame(rows, columns=COLS)


def test_live_equals_batch_on_fake_engines():
    """Runs in CI: a small model trained on fake data, scored both ways."""
    d = fake_flights()
    train = add_lags(add_labels(d))
    model = make_mlp(seed=0).fit(train[SEQ_FEATURES], train["cycles_left_capped"])
    np.testing.assert_allclose(live_predictions(model, d), batch_predictions(model, d), rtol=0, atol=1e-9)


@pytest.mark.slow   # about 4 minutes: 13,096 flights scored one at a time
@pytest.mark.skipif(not (RAW / "test_FD001.txt").exists(), reason="NASA data not available")
def test_live_equals_batch_on_nasa_test_file():
    from src.data import load_rul, load_test

    model = joblib.load(MODEL_FILE)
    test = load_test()
    live, batch = live_predictions(model, test), batch_predictions(model, test)
    assert len(live) == 13_096
    np.testing.assert_allclose(live, batch, rtol=0, atol=1e-9)

    # at each engine's last cycle, the live predictions give the reported test score
    ordered = test.sort_values(["unit", "cycle"]).reset_index(drop=True)
    last = ordered.groupby("unit").tail(1).index
    truth = ordered.loc[last, ["unit"]].merge(load_rul(test), on="unit")["RUL"]
    assert round(root_mean_squared_error(truth, live[last]), 2) == 14.94
