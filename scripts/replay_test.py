"""Send NASA's test file to a running API flight by flight, as engines would in service.

Run from the project folder against a freshly started API (its state must be empty):
    python -m scripts.replay_test [--url http://127.0.0.1:8000]
"""
import argparse

import httpx
import numpy as np

from src.config import COLS, K_ALERT
from src.data import load_rul, load_test


def main(url):
    test = load_test().sort_values(["unit", "cycle"])
    rul = load_rul(test).set_index("unit")["RUL"]   # true RUL at each engine's last cycle
    cut = test.groupby("unit")["cycle"].max()

    with httpx.Client(base_url=url, timeout=10) as client:
        print("API:", client.get("/health").json())
        first_alert, last_pred = {}, {}
        for row in test.to_dict("records"):
            unit = int(row["unit"])
            body = {"cycle": int(row["cycle"]), **{c: float(row[c]) for c in COLS[2:]}}
            r = client.post(f"/engines/{unit}/cycles", json=body)
            if r.status_code != 200:
                raise SystemExit(f"engine {unit} cycle {body['cycle']}: {r.status_code} {r.text}\n"
                                 "Is the API freshly started? It must not hold these engines already.")
            answer = r.json()
            if answer["alert"] and unit not in first_alert:
                first_alert[unit] = answer["cycle"]
            last_pred[unit] = answer["predicted_rul"]

    print(f"\nFirst alert per engine (alert = predicted RUL <= {K_ALERT}); true RUL = cycles really left then")
    print(" unit  first alert at cycle  true RUL then   (NASA cut-off cycle)")
    for unit in cut.index:
        if unit in first_alert:
            c = first_alert[unit]
            print(f"{unit:5d}  {c:20d}  {rul[unit] + cut[unit] - c:13d}   ({cut[unit]})")
        else:
            print(f"{unit:5d}  {'never':>20}  {'':>13}   ({cut[unit]})")

    units = list(cut.index)
    errors = np.array([last_pred[u] for u in units]) - rul[units].values
    print(f"\nEngines that alerted before the cut-off: {len(first_alert)} of {len(units)}")
    print(f"RMSE at each engine's last cycle: {np.sqrt(np.mean(errors ** 2)):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    main(p.parse_args().url)
