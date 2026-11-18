import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silences a harmless Windows warning

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, daily_market, load_csv, weight_medians
from model import fit, predict_rate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def prepare(raw_train, raw_other, daily):
    wf = weight_medians(raw_train)
    cats = sorted(raw_train["equipment"].unique())
    tr = build_features(raw_train, daily, wf)
    other = build_features(raw_other, daily, wf)
    for f in (tr, other):
        f["equipment"] = pd.Categorical(f["equipment"].astype(str), categories=cats)
    return tr, other


def december_inputs(train, val):
    """The December chart file has no market columns, so I give each day the
    average market_index / quote_signal of that day's loads in validation.csv.
    Coordinates come from the cities' rows in the other files."""
    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    coords = pd.concat([train, val])
    pick = coords.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    drop = coords.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    dec = dec.join(pick, on="pickup").join(drop, on="delivery")
    dec["market_index"] = np.nan          # filled with the day average later
    dec["quote_signal"] = np.nan
    return dec


def main():
    train = load_csv(DATA / "train_test.csv")
    val = load_csv(DATA / "validation.csv")
    daily = daily_market([train, val])

    tr, va = prepare(train, val, daily)
    models = fit(tr)

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    va["predicted_rate"] = predict_rate(models, va)
    out = template[["load_id"]].merge(va[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and len(out) == 12_000
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print("wrote validation_predictions.csv", out.shape)

    dec = december_inputs(train, val)
    _, dec_feats = prepare(train, dec, daily)
    dec_feats["quote_signal"] = dec_feats["day_quote_signal"]
    dec_feats["predicted_rate"] = predict_rate(models, dec_feats)
    cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    final = dec_feats[cols].copy()
    final["weight"] = final["weight"].round(0).astype(int)
    final["distance"] = final["distance"].round(0).astype(int)
    final["date"] = final["date"].dt.strftime("%Y-%m-%d")
    final["predicted_rate"] = final["predicted_rate"].round(2)
    final.to_csv(ROOT / "december_predictions.csv", index=False)
    print("wrote december_predictions.csv", final.shape)
    print(final["predicted_rate"].describe().round(2))


if __name__ == "__main__":
    main()
