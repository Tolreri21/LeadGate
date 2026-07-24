import pandas as pd
import numpy as np


def calculate_profit(y_true, y_pred, revenue=100, cost=10):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    calls = (y_pred == 1).sum()
    profit = tp * revenue - calls * cost
    return int(profit)


def threshold_sweep(y_true, proba, thresholds=None, revenue=100, cost=10):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    results = []

    for t in thresholds:
        pred = proba >= t
        tp = (pred & (y_true == 1)).sum()
        calls = pred.sum()
        prec = tp / calls if calls > 0 else 0
        rec = tp / (y_true == 1).sum()
        money = calculate_profit(y_true, pred, revenue, cost)
        results.append(
            {
                "threshold": t,
                "money": money,
                "n_calls": calls,
                "precision": prec,
                "recall": rec,
            }
        )

    df = pd.DataFrame(results)
    return df


def pick_best(sweep_df, max_calls=None):
    if max_calls is None:
        best_threshold_nolimit = sweep_df.loc[sweep_df["money"].idxmax()]
        return best_threshold_nolimit

    df_limit = sweep_df[sweep_df["n_calls"] <= max_calls]
    if df_limit.empty:
        raise ValueError(f"even the strictest threshold exceeds {max_calls} calls")
    best_threshold_limit = df_limit.loc[df_limit["money"].idxmax()]
    return best_threshold_limit
