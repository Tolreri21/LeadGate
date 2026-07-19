# LeadGate

Lead scoring for bank telemarketing: predicts who subscribes to a term deposit, trained on
45k real calls (UCI Bank Marketing) and served via S3 → Lambda → API Gateway.

Built on one constraint: **no feature the call centre couldn't know before dialling.** That
rules out the dataset's single strongest predictor — the call's own duration — which is only
known *after* the call it's supposed to help decide. The API contract has no field for it,
and that absence is the proof, not a corner cut.

## Problem statement

- **Task:** binary classification — will a client subscribe to a term deposit after a call.
- **Unit of observation:** a single contact (the last call of a campaign to a client), not a client.
- **Target:** `y` (`yes` = subscribed). Positive rate **11.7%** (5,289 / 45,211).
- **Metric:** PR-AUC (`average_precision`) + minority-class recall. *Not* accuracy — the
  "nobody subscribes" constant already scores **88.3%**.
- **Baseline:** `DummyClassifier(strategy="most_frequent")` and `LogisticRegression`.
- **Threshold:** not 0.5 — tuned on out-of-fold predictions, never on the test set.

**Business framing:** call prioritisation — the model tells the call centre who to dial first.

## Data

UCI Bank Marketing — **`bank-full.csv`**: 45,211 rows, 16 features + target. Real Portuguese
bank telemarketing campaigns, May 2008 – November 2010, ordered by date.
Source: <https://archive.ics.uci.edu/dataset/222/bank+marketing>

> ⚠️ Use `bank-full.csv`, **not** `bank-additional-full.csv`. In the latter, after dropping
> `duration`, the strongest remaining predictors are month-level macro indicators identical
> for every client in a month — a random split then peeks into the future. `bank-full.csv`
> doesn't carry those columns.

Raw data is git-ignored and lives in `data/raw/` (immutable; never edited).

### Data dictionary

From `bank-names.txt` in the archive.

| Feature | Type | Meaning |
|---|---|---|
| `age` | numeric | Client age |
| `job` | categorical | Type of job |
| `marital` | categorical | Marital status |
| `education` | categorical | Education level |
| `default` | categorical | Has credit in default? |
| `balance` | numeric | Average yearly balance (€) |
| `housing` | categorical | Has a housing loan? |
| `loan` | categorical | Has a personal loan? |
| `contact` | categorical | Contact communication type |
| `day` | numeric | Last contact day of the month |
| `month` | categorical | Last contact month |
| `duration` | numeric | Last contact duration (seconds) — **dropped: leakage, unknown before the call** |
| `campaign` | numeric | Contacts during this campaign (incl. last) |
| `pdays` | numeric | Days since last contact in a previous campaign (`-1` = not previously contacted) |
| `previous` | numeric | Contacts before this campaign |
| `poutcome` | categorical | Outcome of the previous campaign |
| `y` | target | Subscribed a term deposit? (`yes` / `no`) |

## Key findings (EDA)

- **Class imbalance:** 11.7% positives (5,289 / 45,211) — drives the PR-AUC choice; a constant
  "no" already scores 88.3% accuracy.
- **`duration` is a leak → dropped.** The single strongest signal (event rate climbs 0.2% → 45%
  across deciles, Spearman 0.34), but it's known only *after* the call and the causality is
  reversed (an interested client causes a long call). Found by hand, then confirmed by the docs.
- **`pdays` ≈ `previous` (ρ = 0.99).** They share the `-1` "never contacted" sentinel and are
  collinear → keep `previous`, drop `pdays`.
- **`unknown` means two different things.** In `poutcome` (81.7%) it's information — "no prior
  campaign", the same rows as `pdays = -1`; in `education` / `job` it's a genuine missing value.
- **`balance`** is right-skewed with negative values → `PowerTransformer("yeo-johnson")` (plain
  `log` fails on negatives).
- **`age`** is U-shaped (young and senior subscribe more) → binned, not fed raw to a linear model.
- **`month`** is kept (known before the call) but confounded — three years are collapsed into 12
  labels, so it isn't clean seasonality.

Full per-feature verdicts and the analysis: `notebooks/01-eda.ipynb`.

## Layout

```
data/
  raw/          # immutable source (bank-full.csv, git-ignored)
  processed/    # final datasets for models
notebooks/
  01-eda.ipynb
  02-preprocessing.ipynb
models/         # serialized models
reports/figures/
tests/
```

## Getting started

```bash
uv sync          # install dependencies from pyproject.toml / uv.lock
```

## Status

- **PR1** — project scaffold, CI, dependencies, problem statement. ✅
- **PR2** — EDA: sanity checks, distributions, event-rate analysis, feature verdicts. ✅
- **Next (PR3)** — preprocessing + split: the verdicts above become a `ColumnTransformer` /
  `Pipeline`, `duration` is dropped, and the test set is held out.
