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

## Preprocessing (PR3)

Turns the EDA verdicts into a single `ColumnTransformer`, fit on **train only**:

- **Dropped:** `duration` (leakage), `day` (noise), `pdays` (ρ = 0.99 with `previous`).
- **Cleaned:** 5 rows where `poutcome = unknown` contradicts `previous > 0` → 45,206 rows.
- **Encoded:** 9 categoricals → `OneHotEncoder(handle_unknown="ignore")`; `age` → 5 quantile
  bins → OHE (captures the U-shape); `campaign` / `previous` passed through raw.
- **Transformed:** `balance` → `PowerTransformer("yeo-johnson")` (right-skew with negatives).
- **Split:** stratified 80/20 (`random_state=42`) — the test set is never touched by any `fit`.
- **No outlier removal:** extreme values are real; deleting them would bias the model and drop
  minority-class rows. Skew is handled by the transforms, not by deletion.

Output: **52 features**. The fitted transformer is saved to `models/preprocessor.joblib` for the
serving pipeline; processed splits go to `data/processed/` (git-ignored, regenerable).

## Baseline models (PR4)

The preprocessor is **rebuilt unfitted** and dropped into a `Pipeline([("preprocessor", ct),
("model", …)])`, so cross-validation re-fits it inside every fold — no leakage. The saved
`preprocessor.joblib` is for serving, never for scoring. Metric is PR-AUC (`average_precision`);
the test set is touched exactly once.

- **Floor — `DummyClassifier("most_frequent")`:** PR-AUC = **0.117**, i.e. the positive rate.
  Anything at or below this has learned nothing.
- **`LogisticRegression(class_weight="balanced")`:** PR-AUC = **0.40 ± 0.03** (5-fold stratified
  CV on train) — **~3.4×** the floor.
- **Held-out test:** PR-AUC = **0.41**; at the default 0.5 threshold, minority-class recall
  **0.64** / precision **0.27** — flags most subscribers, but with many false positives.

`class_weight="balanced"` offsets the 11.7% imbalance so the model doesn't collapse to "always
no". The 0.5 threshold is left untouched — tuning it on out-of-fold predictions is a later PR.
PR curve: `reports/figures/PR-AUC.png`.

## Model comparison (PR5)

Do tree models beat the logistic-regression baseline? Same protocol — 5-fold stratified
CV on train, PR-AUC, test untouched.

- **RandomForest** (ordinal categoricals, raw numerics): **0.38** — below the LogReg
  baseline. Trees gain nothing from OHE/binning/scaling, and the signal is too simple for
  RF to exploit.
- **HistGradientBoosting** (native categoricals, raw numerics): **0.42** — the strongest,
  but only marginally above LogReg's **0.40**, and within one CV std.
- **Light `RandomizedSearchCV` on HGB:** stays ~0.42. Tuning moves nothing; the ~0.42
  ceiling is confirmed.

**Decision:** keep **LogisticRegression** for serving. At a statistical tie the cheaper,
faster, interpretable model wins — its held-out test PR-AUC is **0.414** (PR4). HGB's
marginal edge doesn't justify a heavier, opaque artifact on Lambda.

## Imbalance handling (PR6)

At 11.7% positives, does reweighting or resampling beat leaving the class ratio alone? Same
protocol — champion (LogReg), 5-fold stratified CV on train, PR-AUC, test untouched. Resamplers
sit **inside** an `imblearn.Pipeline`, so they touch only the training folds; the validation fold
keeps its real 11.7% ratio (resampling it would leak synthetic points into the score).

| Strategy | CV PR-AUC |
|---|---|
| None (no reweighting) | **0.402 ± 0.026** |
| `class_weight="balanced"` | 0.400 ± 0.025 |
| SMOTE (after one-hot) | 0.398 ± 0.025 |
| SMOTENC (native categoricals) | 0.357 ± 0.019 |

**None, `class_weight` and SMOTE are a statistical tie** — a 0.004 spread inside one CV std
(~0.025). SMOTENC is clearly *worse*: to reach 50/50 it synthesises ~4× the minority rows on data
that is 9 of 13 columns categorical, distorting the decision boundary more than any reweighting
fixes.

Why resampling buys nothing here: **PR-AUC scores ranking, and imbalance doesn't break ranking —
it breaks the threshold.** Reweighting and resampling move where the "0.5" line falls, not the
order of the ranked call list, so a ranking metric can't reward them and the synthetic noise can
only cost. This is the expected result at a moderate 11.7% (SMOTE earns its keep below ~1%).

**Decision: keep `class_weight="balanced"` — not because it wins PR-AUC (it ties), but because it
fixes the operating point for free.** Fit on the true ratio, plain LogReg outputs probabilities
averaging **0.117** — the base rate, well-calibrated, but so low that at threshold 0.5 it flags
almost nobody (recall ≈ 0). `class_weight="balanced"` inflates them to average **0.41**, making the
default threshold usable. SMOTE/SMOTENC are rejected: more machinery and synthetic data for a lower
or equal score. The operating point itself — the real lever for a call list — is tuned on
out-of-fold predictions in PR7, not here.

Analysis: `notebooks/05-imbalance.ipynb`.

## Operating point (PR7)

PR-AUC rates the *ranking*; it says nothing about **where to cut the ranked call list**. That
cut — the threshold — is the real lever for a call centre, and 0.5 is arbitrary. PR7 sets it from
business economics on out-of-fold predictions (same 5-fold CV, `cross_val_predict`), test untouched.

**Cost model (illustrative figures):** €100 profit per subscription, €10 per call attempt. For a
threshold `t`, profit = `TP·100 − calls·10`. Sweep `t`, keep the max.

**`class_weight` dropped — the threshold makes it redundant.** It only ever existed to rescue the
default 0.5 (PR6). Tuning the threshold explicitly does that job directly, so PR7 switches to
**plain, unweighted `LogisticRegression`**, which returns *calibrated* probabilities. The two are
equivalent for the decision: identical ranking (PR6), so the same clients are called for the same
profit — only the threshold *number* differs (0.11 on the plain scale vs ~0.50 on the balanced one).

| Scenario | Threshold | Calls | Recall | Precision | Profit |
|---|---|---|---|---|---|
| Unlimited (economic optimum) | **0.11** | 11,727 (⅓ of leads) | 0.67 | 0.24 | €167k |
| Capacity 2,000/day (top-N) | ~0.37 | 2,004 | 0.27 | **0.56** | €93k |

Two readings of one ranked list. **Unlimited:** call everyone still profitable — the optimum lands
at **0.11 ≈ the break-even `c/v` = 0.1**, catching two-thirds of subscribers at precision **0.24**
(2× the 11.7% base rate). **Under a 2,000-call cap** you skim the top: precision jumps to **0.56** —
one call in two converts — because when calls are scarce, ranking quality (the PR-AUC of PR5/6) pays
off directly.

**Policy: floor + capacity.** 0.11 is an *economic floor* — below it each marginal call loses money,
so never go lower. Capacity decides how far down toward it you reach: `t = max(0.11, top-N cut)`.
Only the floor is frozen (`models/threshold.json`); the daily call budget is a runtime input.

**Calibration check.** The decision leans on the probability *equalling* the break-even, so plain
LogReg's calibration is verified on OOF predictions: the reliability curve tracks the diagonal and
the Brier score is **0.086** — well-calibrated, and notably so around the 0.1 operating region. A
predicted 0.11 really is a ~11% chance, so the cost-based threshold isn't lying; no
`CalibratedClassifierCV` needed. Reliability diagram: `reports/figures/calibration.png`.

Threshold **0.11** is frozen and applied once to the held-out test in PR8.
Analysis: `notebooks/06-threshold.ipynb`.

## Layout

```
data/
  raw/          # immutable source (bank-full.csv, git-ignored)
  processed/    # final datasets for models
notebooks/
  01-eda.ipynb
  02-preprocessing.ipynb
  03-baseline.ipynb
  04-models.ipynb
  05-imbalance.ipynb
  06-threshold.ipynb
models/         # serialized models + threshold.json
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
- **PR3** — preprocessing + split: verdicts become a `ColumnTransformer`, `duration`/`day`/`pdays`
  dropped, stratified test set held out, fitted preprocessor saved for serving. ✅
- **PR4** — baseline models: `DummyClassifier` + `LogisticRegression` in a leakage-free
  `Pipeline`, scored by PR-AUC via stratified CV — LogReg **0.40** vs the **0.117** floor. ✅
- **PR5** — model comparison: RandomForest (**0.38**) and HistGradientBoosting (**0.42**, tuned)
  vs the LogReg baseline (**0.40**); trees don't clear the bar, LogReg kept for serving. ✅
- **PR6** — imbalance handling: none / `class_weight` / SMOTE / SMOTENC compared leakage-free in an
  `imblearn.Pipeline` — all tie on PR-AUC except SMOTENC (**0.357**, worse); resampling rejected,
  `class_weight="balanced"` kept for the operating point, not for the score. ✅
- **PR7** — operating point: threshold tuned by profit (€100/subscription, €10/call) on OOF
  predictions; `class_weight` dropped for a calibrated plain LogReg (Brier **0.086**); economic
  optimum **0.11** (≈ break-even) with a `max(floor, top-N)` capacity policy, floor frozen to
  `threshold.json`. ✅
- **Next (PR8)** — final evaluation on the held-out test (apply the frozen **0.11** once) plus a
  time-based split as a leakage sanity check.
