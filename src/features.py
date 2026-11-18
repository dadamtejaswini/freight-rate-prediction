"""Cleaning and feature building for the freight rate model.

Everything here works on the *inputs* only (never on posted_rate), so the same
functions are used for training, validation and the December chart.
"""
import numpy as np
import pandas as pd

CATEGORICAL = ["pickup", "delivery", "equipment"]

# Rate-per-mile is what really moves with the market, so the model predicts
# log(rate per mile) and we multiply back by distance at the end.
# City names are left out on purpose: 12% of the validation loads use cities
# that never appear in training, so the model leans on coordinates instead.
NUMERIC = [
    "log_distance", "market_index", "quote_signal",
    "day_market_index", "day_quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]
FEATURES = NUMERIC + ["equipment"]


def load_csv(path):
    return pd.read_csv(path, parse_dates=["date"])


def daily_market(frames):
    """Daily average market_index / quote_signal built from every row we have
    (train + validation inputs). posted_rate is never touched."""
    allrows = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    daily = allrows.groupby("date").agg(
        day_market_index=("market_index", "median"),
        day_quote_signal=("quote_signal", "mean"),
    )
    # a day with no usable value gets filled from its neighbours
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full).interpolate(limit_direction="both")
    daily.index.name = "date"
    return daily.reset_index()


def build_features(df, daily, weight_fill):
    """df: raw loads. daily: output of daily_market. weight_fill: dict of the
    median weight per equipment type, learned from training data only."""
    out = df.copy()

    # negative weights are sign errors (|w| looks like a normal weight)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)
    fill = out["equipment"].map(weight_fill)
    out["weight"] = out["weight"].fillna(fill)
    out["weight_k"] = out["weight"] / 1000.0

    out = out.merge(daily, on="date", how="left")
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out["day_market_index"])

    out["log_distance"] = np.log(out["distance"])
    out["dow"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def weight_medians(train):
    return train.assign(weight=train["weight"].abs()).groupby("equipment")["weight"].median().to_dict()


def set_categories(frames, train_df):
    """Give every frame the same category codes as the training set."""
    for col in CATEGORICAL:
        cats = sorted(set(train_df[col].astype(str)))
        for f in frames:
            f[col] = pd.Categorical(f[col].astype(str), categories=cats)
    return frames
