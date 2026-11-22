"""Scoring: cross-validation by engine (notebook section 9) and the test evaluation (section 11)."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold

from src.config import EVAL_MAX_RUL, N_FOLDS


def oof_predictions(make_model, df, cols, target="cycles_left_capped"):
    """Train one model per fold and predict the engines it never saw.

    Returns (pred, fold): one out-of-fold prediction per row, and which fold that row was validated in.
    The same predictions give the CV score here and drive the alert replay in alert.py.
    """
    pred = np.full(len(df), np.nan)
    fold = np.full(len(df), -1)
    # groups=unit: all rows of an engine land in the same fold, so no engine is in train and validation at once
    for f, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(df, groups=df["unit"])):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        model = make_model().fit(tr[cols], tr[target])
        pred[va_idx] = model.predict(va[cols])
        fold[va_idx] = f
    return pred, fold


def cv_score(df, pred, fold):
    """Mean and spread of the per-fold RMSE, against the TRUE RUL, on rows with RUL <= 150."""
    scores = []
    for f in range(N_FOLDS):
        # the cap is only for training: scoring against the true RUL means the cap can't hide errors
        rows = (fold == f) & (df["RUL"] <= EVAL_MAX_RUL).values
        scores.append(root_mean_squared_error(df["RUL"].values[rows], pred[rows]))
    return np.mean(scores), np.std(scores)


def last_cycles(test_seq, rul):
    """Each test engine's last observed row, with its true RUL: the only rows NASA gives an answer for."""
    return test_seq.groupby("unit").tail(1).merge(rul, on="unit")


def nasa_score(err):
    # err = prediction - truth; late (positive) errors are punished harder than early ones
    return np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1))


def test_metrics(true, pred):
    err = pred - true
    return {
        "rmse": float(root_mean_squared_error(true, pred)),
        "nasa": float(nasa_score(err)),
        "mae": float(mean_absolute_error(true, pred)),
        "bias": float(err.mean()),
        "within_15": int((np.abs(err) <= 15).sum()),
        "late_over_15": int((err > 15).sum()),
    }


def band_table(results):
    """RMSE and mean error by true-RUL band. results needs columns RUL, pred, error."""
    band = pd.cut(results["RUL"], [0, 25, 50, 75, 100, 125, 150]).rename("band")
    return results.groupby(band, observed=True).apply(lambda g: pd.Series({
        "engines": len(g),
        "RMSE": root_mean_squared_error(g["RUL"], g["pred"]),
        "mean error": g["error"].mean(),
    }), include_groups=False)
