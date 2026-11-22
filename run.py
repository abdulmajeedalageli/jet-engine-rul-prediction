"""Run the final pipeline end to end: python run.py [--skip-cv]"""
import argparse
import json

import joblib

from src.alert import best_k, cost_table
from src.config import CAP, MODEL_FILE, MODELS, OUTPUTS, SEEDS
from src.data import add_labels, load_rul, load_test, load_train
from src.evaluate import band_table, cv_score, last_cycles, oof_predictions, test_metrics
from src.features import SEQ_FEATURES, add_lags
from src.model import make_baseline, make_mlp


def main(skip_cv=False):
    OUTPUTS.mkdir(exist_ok=True)
    metrics = {}

    # 1. Data: the loaders assert the checks (duplicates, RUL 0 at failure, RUL file matches test engines)
    df = add_lags(add_labels(load_train()))
    test = load_test()
    rul = load_rul(test)
    print(f"train {len(df)} rows, {df['unit'].nunique()} engines | "
          f"test {len(test)} rows, {test['unit'].nunique()} engines | inputs {len(SEQ_FEATURES)}")
    metrics["data"] = {"train_rows": len(df), "test_rows": len(test), "inputs": len(SEQ_FEATURES)}

    # 2. Cross-validation by engine; the MLP's out-of-fold predictions are kept for the alert replay
    if not skip_cv:
        base = cv_score(df, *oof_predictions(make_baseline, df, SEQ_FEATURES))
        pred, fold = oof_predictions(make_mlp, df, SEQ_FEATURES)
        mlp = cv_score(df, pred, fold)
        print("\nCV RMSE (true RUL <= 150, mean ± std over 5 folds)")
        print(f"  baseline (always the average) {base[0]:.1f} ± {base[1]:.1f}")
        print(f"  MLP, last 30 cycles           {mlp[0]:.1f} ± {mlp[1]:.1f}")
        metrics["cv"] = {"baseline": list(map(float, base)), "mlp": list(map(float, mlp))}

    # 3. Final models on all 100 training engines, scored once at each test engine's last cycle
    last = last_cycles(add_lags(test), rul)
    print("\nTest, 3 seeds")
    models = {}
    metrics["test_seeds"] = {}
    for seed in SEEDS:
        models[seed] = make_mlp(seed).fit(df[SEQ_FEATURES], df["cycles_left_capped"])
        m = test_metrics(last["RUL"].values, models[seed].predict(last[SEQ_FEATURES]))
        print(f"  seed {seed}: RMSE {m['rmse']:.2f}  NASA {m['nasa']:.0f}")
        metrics["test_seeds"][seed] = m

    # 4. Seed 0 is used as the final model. It is not the best of the three on test (seed 2: 14.80 vs 14.94),
    #    so it was not cherry-picked; the seed range is reported alongside it.
    results = last[["unit", "cycle", "RUL"]].copy()
    results["pred"] = models[0].predict(last[SEQ_FEATURES])
    results["error"] = results["pred"] - results["RUL"]
    m = metrics["test_seeds"][0]
    print(f"\nFinal model (seed 0): RMSE {m['rmse']:.2f} | NASA {m['nasa']:.0f} | MAE {m['mae']:.1f} | "
          f"bias {m['bias']:.1f} | within ±15: {m['within_15']} of {len(results)} | late > 15: {m['late_over_15']}")
    bands = band_table(results)
    print("\nError by true-RUL band\n", bands.round(1).to_string(), sep="")
    metrics["final"] = m
    metrics["bands"] = {str(b): row.to_dict() for b, row in bands.iterrows()}

    joblib.dump(models[0], OUTPUTS / "model.joblib")
    MODELS.mkdir(exist_ok=True)
    joblib.dump(models[0], MODEL_FILE)   # the copy the API loads; committed so it ships without the NASA data
    results.to_csv(OUTPUTS / "test_predictions.csv", index=False)

    # 5. Alert threshold, replayed on the out-of-fold predictions (needs step 2)
    if not skip_cv:
        df["oof_pred"] = pred
        costs = cost_table(df)
        print("\nAlert replay (lead 10, failure 100, wasted cycle 1)\n", costs.to_string(index=False), sep="")
        costs.to_csv(OUTPUTS / "alert_costs.csv", index=False)
        best = best_k(df)
        for c_fail, (k, f, w, c) in best.items():
            print(f"  failure cost {c_fail}: best k {k} ({f} failures, {w} wasted, cost {c})")
        metrics["best_k"] = {c_fail: {"k": k, "failures": f, "wasted": w, "cost": c}
                             for c_fail, (k, f, w, c) in best.items()}
    else:
        print("\n--skip-cv: cross-validation and alert replay skipped")
        (OUTPUTS / "alert_costs.csv").unlink(missing_ok=True)   # don't leave a stale table from an earlier run

    metrics["cap"] = CAP
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to {OUTPUTS}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-cv", action="store_true", help="skip cross-validation and the alert replay")
    main(p.parse_args().skip_cv)
