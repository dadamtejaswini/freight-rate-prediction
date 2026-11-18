# Freight Rate Prediction

This project predicts the rate (in dollars) for freight loads. It trains on loads from January to October 2025 and predicts the 12,000 loads in the November–December validation file.

## Quick start

1. Install Python 3.10 or newer.
2. Install the libraries:

```bash
python -m pip install -r requirements.txt
```

3. Run the project from this folder, in this order:

```bash
python src/validate.py
python src/train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
python src/make_report.py
```

| Step | What it does |
|---|---|
| `validate.py` | Tests the model and prints the error table. It takes a few minutes. |
| `train_predict.py` | Trains the final model and writes the prediction files. |
| `score.py` | The provided checker. It confirms the files are valid and draws the December chart. |
| `make_report.py` | Builds the Word report, including the chart. |

## Output files

- `validation_predictions.csv`: predicted rate for each of the 12,000 validation loads (`load_id,predicted_rate`)
- `december_predictions.csv`: predictions for the fixed December chart
- `Freight_Rate_Report.docx`: validation approach, results and the December chart

## Project layout

```
data/                  input CSV files
src/features.py        data cleaning and feature building
src/model.py           the model
src/validate.py        validation experiments
src/train_predict.py   final training and prediction
src/make_report.py     report builder
score.py               provided scorer
```

## How it works

**Model.** It predicts the log of the rate per mile, then multiplies by distance. The model is an average of three scikit-learn gradient boosting models, trained with an absolute-error loss so that a few unusual rates do not distort it.

**Data cleaning.**
- Negative weights are turned positive.
- Missing weights are filled with the median for that equipment type.
- Missing `market_index` values are filled with that day's median.

**Validation.** The validation data is from the future of the training data, so I did not use a random split. I tested:
- a time split (train Jan–Aug, test Sep–Oct)
- rolling months (July to October)
- cities the model has never seen

About 12% of validation loads use cities missing from training, so the model uses coordinates and distance instead of city names.

**Result.** Typical error is about 3–4%, compared with about 10% for a simple baseline. The weakest month was August (about 9.7%), when the market index moved faster than actual rates. More detail is in the report.