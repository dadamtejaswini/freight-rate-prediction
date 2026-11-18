"""Validation experiments (run:  python src/validate.py).

Why not a random split?
The real validation file is Nov-Dec 2025, so it is the *future* of the Jan-Oct
training data, and 12% of its loads use cities that never appear in training.
A random split would leak both time and lanes, so I check three things:

  A. Time split: fit on Jan-Aug, score on Sep-Oct.
  B. Rolling origin: fit on everything before a month, score that month
     (Jul, Aug, Sep, Oct).
  C. Unseen cities: like A, but 8 cities are removed from the fit and the
     score is only on Sep-Oct loads touching them.
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silences a harmless Windows warning

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURES, build_features, daily_market, load_csv, weight_medians
from model import fit, predict_rate

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def metrics(y_true, y_pred):
    err = np.asarray(y_pred) - np.asarray(y_true)
    ape = np.abs(err) / np.asarray(y_true)
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "MAPE%": 100 * ape.mean(), "MedAPE%": 100 * np.median(ape)}


# --- simple reference models ------------------------------------------------
def baseline_predict(train, test):
    """Median rate-per-mile for each equipment type x distance.  No ML."""
    rpm = train["posted_rate"] / train["distance"]
    med = rpm.groupby(train["equipment"], observed=True).median()
    return test["equipment"].map(med).astype(float) * test["distance"]


def ridge_predict(train, test):
    def prep(d):
        x = d[FEATURES].copy()
        return pd.get_dummies(x, columns=["equipment"], dtype=float)
    xa, xb = prep(train), prep(test)
    y = np.log(train["posted_rate"] / train["distance"])
    m = Ridge(alpha=1.0).fit(xa, y)
    return np.exp(m.predict(xb.reindex(columns=xa.columns, fill_value=0.0))) * test["distance"].values


def hgb_predict(train, test):
    return predict_rate(fit(train), test)


MODELS = {"baseline (equipment median)": baseline_predict,
          "ridge (log rate/mile)": ridge_predict,
          "boosted blend (final)": hgb_predict}


def featurize(raw_train, raw_test, daily):
    wf = weight_medians(raw_train)
    tr = build_features(raw_train, daily, wf)
    te = build_features(raw_test, daily, wf)
    for f in (tr, te):
        f["equipment"] = pd.Categorical(f["equipment"].astype(str), categories=sorted(raw_train["equipment"].unique()))
    return tr, te


def score_all(raw_train, raw_test, daily, label, rows, mask=None):
    print("running:", label, flush=True)
    tr, te = featurize(raw_train, raw_test, daily)
    mask = np.ones(len(te), bool) if mask is None else np.asarray(mask)
    for name, fn in MODELS.items():
        print("   fitting", name, flush=True)
        pred = np.asarray(fn(tr, te))
        rec = metrics(te["posted_rate"].values[mask], pred[mask])
        rec.update(split=label, model=name, n=int(mask.sum()))
        rows.append(rec)


def main():
    train = load_csv(DATA / "train_test.csv")
    val = load_csv(DATA / "validation.csv")
    daily = daily_market([train, val])
    rows = []

    # A. time split
    fit_a = train[train.date < "2025-09-01"]
    test_a = train[train.date >= "2025-09-01"].reset_index(drop=True)
    score_all(fit_a, test_a, daily, "A: fit Jan-Aug, test Sep-Oct", rows)

    # B. rolling origin
    for month in (7, 8, 9, 10):
        start = pd.Timestamp(2025, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        score_all(train[train.date < start], train[(train.date >= start) & (train.date < end)].reset_index(drop=True),
                  daily, f"B: rolling, test month {month}", rows)

    # C. unseen cities
    rng = np.random.default_rng(7)
    hidden = list(rng.choice(sorted(set(train.pickup)), 8, replace=False))
    touches = test_a.pickup.isin(hidden) | test_a.delivery.isin(hidden)
    fit_c = fit_a[~(fit_a.pickup.isin(hidden) | fit_a.delivery.isin(hidden))]
    score_all(fit_c, test_a, daily, "C: unseen cities (Sep-Oct)", rows, mask=touches)
    print("hidden cities in C:", hidden)

    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "validation_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(res[["split", "model", "n", "MAE", "RMSE", "MAPE%", "MedAPE%"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
