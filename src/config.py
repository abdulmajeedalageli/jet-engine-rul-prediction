"""Every setting of the final pipeline, in one place."""
from pathlib import Path

# Paths are built from this file's location, so the scripts work from any folder, on Windows or Linux.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"   # committed: the model the API serves
MODEL_FILE = MODELS / "mlp_seed0.joblib"

# The raw files have no header: unit, cycle, 3 operating settings, 21 sensors.
COLS = ["unit", "cycle", "set1", "set2", "set3", *[f"s{i}" for i in range(1, 22)]]

# The 14 sensors kept (notebook section 8: the others are constant, noise, or s6).
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
           "s13", "s14", "s15", "s17", "s20", "s21"]

W = 30               # how many past cycles the model sees
CAP = 125            # label cap: above this the sensors look the same
EVAL_MAX_RUL = 150   # scores use rows with true RUL <= this

# Model and evaluation
MLP_LAYERS = (64, 32)
MAX_ITER = 300
SEEDS = [0, 1, 2]    # seed 0 is the final model
N_FOLDS = 5

# Alert replay (section 15)
LEAD = 10            # cycles needed to arrange maintenance
C_FAIL = 100         # cost of one failure in service
C_CYCLE = 1          # cost of one wasted cycle of life
KS = [10, 15, 20, 25, 30, 35, 40, 50]
FAIL_COSTS = [50, 100, 500, 1000]
K_ALERT = 20         # the alert decision: best k for failure costs 100 to 1000 (alert.best_k)
