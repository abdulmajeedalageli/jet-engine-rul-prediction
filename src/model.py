"""The final model and the baseline it must beat (notebook sections 9 and 11)."""
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MAX_ITER, MLP_LAYERS


def make_mlp(seed=0):
    # Scaler inside the pipeline: it learns mean/std from the training rows only, in every fold and at test time.
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=MLP_LAYERS, early_stopping=True,
                                      max_iter=MAX_ITER, random_state=seed))


def make_baseline():
    return DummyRegressor(strategy="mean")
