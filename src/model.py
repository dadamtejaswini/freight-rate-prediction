"""The final model, kept in one place so validation and the real run match.

Three gradient boosting models that differ only in which inputs they see, with
their predictions averaged (geometric mean).  In rolling validation the average
was more stable than any single one, mostly because the market_index column is
a shaky guide when the market moves faster than the model has seen before.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES

WITH_WEIGHT = FEATURES + ["weight_k"]
NO_MARKET = [f for f in FEATURES if f not in ("market_index", "day_market_index")]
FEATURE_SETS = [FEATURES, WITH_WEIGHT, NO_MARKET]


def new_model():
    # absolute_error = predict the median, so the odd 5-14x rate-per-mile
    # spikes in the training labels can't drag predictions around
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=96,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=["equipment"],
        random_state=7,
    )


def fit(train_feats):
    """Returns a list of fitted models, one per feature set."""
    y = np.log(train_feats["posted_rate"] / train_feats["distance"])
    return [new_model().fit(train_feats[cols], y) for cols in FEATURE_SETS]


def predict_rate(models, feats):
    log_rpm = np.mean([m.predict(feats[cols]) for m, cols in zip(models, FEATURE_SETS)], axis=0)
    return np.exp(log_rpm) * feats["distance"].values
