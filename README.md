# Remaining useful life of jet engines (NASA C-MAPSS FD001)

[![tests](https://github.com/abdulmajeedalageli/jet-engine-rul-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/abdulmajeedalageli/jet-engine-rul-prediction/actions/workflows/tests.yml)

## What I did and what I learned

I started by exploring the data: which sensors change, which are constant, and how sensors behave as an engine wears out. Seven columns never change and the operating settings are just noise in FD001, so I kept 14 sensors.

I capped RUL at 125. At first I thought the cap was tricking the model, so I tested it, always scoring against the true RUL: in cross-validation the same model scored 27.2 without a cap and 16.1 with it, and 125 beat 100 and 150.

My first model was LightGBM on 10-cycle averages (test RMSE 19.0). In an earlier version of my notebook I looked at where it was wrong: near failure it was late by about 7 cycles, and two engines (37 and 41) with about 20 cycles left were predicted around 67. That's the dangerous direction.

Giving the model the raw readings of the last 30 cycles instead of averages fixed most of that. The best model was a small neural network (MLP): cross-validation 16.1, test RMSE 14.9, and accurate near failure (RMSE about 6 for engines with 50 or fewer cycles left). I also tried linear regression, random forest and XGBoost; the input format mattered more than the model type.

I also learned that splitting rows at random makes results look better than they are (a forest went from 20.2 to a fake 13.0), so everything is split by engine.

Finally I turned the model into an alert rule. With a 10-cycle lead time, alerting at 20 predicted cycles left caught every training engine in time at the lowest cost.

## The project
The model reads an engine's last 30 cycles of sensor readings and predicts its remaining useful life (RUL) in
cycles; a maintenance alert fires when the prediction drops to 20. The exploration and model comparison are in
[notebooks/CMAPSS_FD001.ipynb](notebooks/CMAPSS_FD001.ipynb); this repository holds the final pipeline as plain
Python scripts, with tests, an API and a Docker image.

- **Inputs:** 14 sensors × the last 30 cycles, plus the cycle number: 421 inputs.
- **Label:** RUL capped at 125. Scores always use the true (uncapped) RUL.
- **Model:** StandardScaler + MLP (64, 32), early stopping.
- **Validation:** 5-fold cross-validation grouped by engine, RMSE on rows with RUL ≤ 150.
- **Test:** trained on all 100 training engines (seeds 0, 1, 2; seed 0 is the final model), scored once at each
  test engine's last cycle.

## Data
NASA C-MAPSS turbofan engine degradation simulation, subset **FD001** (one operating condition, one fault mode),
from NASA's Prognostics Center of Excellence data set repository ("Turbofan Engine Degradation Simulation"). The
data is **not included**. Put these files in `data/raw/`:

| File | Contents |
|---|---|
| `train_FD001.txt` | 100 engines, each run until failure: 20,631 rows |
| `test_FD001.txt` | 100 other engines, cut off before failure: 13,096 rows |
| `RUL_FD001.txt` | true RUL of each test engine at its last row |

## Setup and run
Python 3.13:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

```bash
python run.py              # cross-validation, test results, alert replay (about 1 minute)
python run.py --skip-cv    # test results only
```

Outputs go to `outputs/` (not committed): `model.joblib`, `test_predictions.csv`, `metrics.json`,
`alert_costs.csv`. The same final model is also written to `models/mlp_seed0.joblib`, which **is** committed: it
is the model the API serves.

## Results

**Cross-validation by engine** (RMSE in cycles)

| Model | RMSE |
|---|---|
| Baseline: always the average | 45.1 ± 0.0 |
| **MLP, last 30 cycles, cap 125** | **16.1 ± 0.3** |

**Test** (NASA's 100 test engines, last cycle of each)

| Metric | Seed 0 (final) | Seeds 0 / 1 / 2 |
|---|---|---|
| RMSE | **14.94** | 14.94 / 15.48 / 14.80 |
| NASA score | **337** | 337 / 384 / 320 |
| MAE | 11.7 | |
| Mean error (bias) | −0.3 | |
| Within ±15 cycles | 70 of 100 | |
| Late by more than 15 | 16 of 100 | |

Seed 0 is not the best seed on test; the range is reported rather than the best one.

**Error by true RUL** (error = prediction − truth; positive = late)

| True RUL | Engines | RMSE | Mean error |
|---|---|---|---|
| (0, 25] | 19 | 5.9 | +0.3 |
| (25, 50] | 14 | 6.8 | −0.9 |
| (50, 75] | 10 | 16.9 | +3.4 |
| (75, 100] | 24 | 17.8 | +9.8 |
| (100, 125] | 22 | 15.4 | −2.9 |
| (125, 150] | 11 | 22.1 | −21.1 |

Near failure the model is accurate and unbiased; the large early errors above 125 come from the cap.

## Alert decision
An alert fires the first cycle the predicted RUL is ≤ k. Maintenance needs 10 cycles of lead time: an alert with
fewer than 10 cycles left, or none at all, is a failure; otherwise the cycles still left at maintenance are wasted.
Cost = failures × 100 + wasted cycles × 1, replayed on **out-of-fold predictions for the training engines** (they
run to failure, and each prediction comes from a model that never saw that engine). The test set is not used.

| k | Failures | Wasted cycles | Cost |
|---|---|---|---|
| 10 | 31 | 194 | 3294 |
| 15 | 6 | 712 | 1312 |
| **20** | **0** | **1265** | **1265** |
| 25 | 0 | 1845 | 1845 |
| 30 | 0 | 2379 | 2379 |
| 35 | 0 | 2917 | 2917 |
| 40 | 0 | 3440 | 3440 |
| 50 | 0 | 4278 | 4278 |

**k = 20** is cheapest for failure costs of 100, 500 and 1000; only at a failure cost of 50 would k = 15 win.

## API
Live scoring, one flight at a time: the API keeps each engine's last 30 flights and builds the inputs with the
same feature function used in training ([api.py](api.py), [src/scorer.py](src/scorer.py)).

| Endpoint | What it does |
|---|---|
| `POST /engines/{unit}/cycles` | score one new flight. Body: `cycle` (integer), `set1`-`set3`, `s1`-`s21` (numbers) |
| `GET /engines/{unit}` | the engine's latest prediction and alert status; **404** if it has sent no flights |
| `GET /health` | `status` and `model_version` (model file name + start of its SHA-256 fingerprint) |

A missing, non-numeric or non-finite field returns **422**; a cycle that isn't higher than the engine's last one
returns **409**.

**Example:** test engine 1's 31st flight, after its first 30:

```bash
curl -X POST http://127.0.0.1:8000/engines/1/cycles -H "Content-Type: application/json" -d '{"cycle": 31, "set1": -0.0006, "set2": 0.0004, "set3": 100.0, "s1": 518.67, "s2": 642.58, "s3": 1581.22, "s4": 1398.91, "s5": 14.62, "s6": 21.61, "s7": 554.42, "s8": 2388.08, "s9": 9056.4, "s10": 1.3, "s11": 47.23, "s12": 521.79, "s13": 2388.06, "s14": 8130.11, "s15": 8.4024, "s16": 0.03, "s17": 393.0, "s18": 2388.0, "s19": 100.0, "s20": 38.81, "s21": 23.3552}'
```

```json
{"unit": 1, "cycle": 31, "predicted_rul": 125.88504181952905, "alert": false, "cycles_seen": 31}
```

**Run it** locally (docs at http://127.0.0.1:8000/docs) or with Docker:

```bash
uvicorn api:app --port 8000
```

```bash
docker build -t jet-engine-rul-api .
docker run -d --name jet-engine-rul-api -p 127.0.0.1:8000:8000 jet-engine-rul-api
```

**Replay the test file through a running API** (needs the data; start the API fresh):

```bash
python -m scripts.replay_test --url http://127.0.0.1:8000
```

Result, locally and in Docker: 17 of 100 engines alerted before NASA's cut-off, each with 20 to 32 cycles really
left, and RMSE at the last cycles **14.94**, the same as the batch result.

## Tests
```bash
pip install -r requirements-dev.txt
ruff check .
pytest -v            # everything except the slow test
pytest -v -m slow    # the full NASA skew test (about 4 minutes)
```

| What | Why |
|---|---|
| Labels count down to 0 and are capped at 125 | the target the model learns |
| Lag `l` equals the reading `l` cycles earlier; early cycles are padded with the oldest reading | the inputs are what they claim to be |
| Deleting an engine's later rows doesn't change a row's inputs; a deliberately leaky version fails that check | no future readings leak in, and the check can catch a leak |
| Lags never cross engines; answer columns are never inputs | no leak between engines or from the label |
| Each engine is in exactly one CV fold | validation is on unseen engines |
| NASA score values; alert replay counts failures and wasted cycles | the metric and the k = 20 decision are computed correctly |
| The final model reproduces test RMSE 14.94 | end-to-end reproducibility |
| Scoring flight by flight equals batch scoring (fake engines; and all 13,096 NASA test flights within 1e-9) | no training/serving skew |
| API: health, a normal sequence, 422 for bad fields, 409 for a repeated cycle, 404 for an unknown engine | the API behaves as documented |

Tests that need the NASA data only run locally and are skipped on GitHub. GitHub Actions runs ruff and pytest
and builds the Docker image on every push ([.github/workflows/tests.yml](.github/workflows/tests.yml)).

## Limitations
- **Simulated data**, one operating condition and one fault mode (FD001 only).
- **Small test set:** 100 engines; the three seeds range from 14.8 to 15.5.
- **The costs and lead time are assumptions,** not airline figures; the table shows how k depends on them.
- **The test set was used twice during development:** first for LightGBM on summary features (RMSE 19.0, NASA
  765), then for this MLP. The MLP was chosen by cross-validation, not by its test score.
- **Early stopping leaks slightly:** it holds back a random 10% of *rows* from the training engines, which have
  near-identical neighbours in the part it trains on. It only affects when training stops; in a separate check
  the validation error was flat between about 25 and 60 epochs.
- **The API keeps its state in memory:** it is lost on restart, and it allows only one server process.
- **No authentication and no monitoring** (prediction quality, input drift, alert rates) yet.
- **Library versions:** results come from the pinned versions in `requirements.txt`; others can change the MLP's
  numbers slightly.
