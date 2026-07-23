from leadgate.threshold import calculate_profit, threshold_sweep, pick_best
import pytest


def test_calculate_profit(y_true, y_pred):
    assert calculate_profit(y_true, y_pred) == 80


def test_threshold_sweep(y_true, proba):
    df = threshold_sweep(y_true, proba, thresholds=[0.999])
    assert df["n_calls"].iloc[0] == 0 and df["precision"].iloc[0] == 0


def test_pick_best(sweep_df):
    assert pick_best(sweep_df)["threshold"] == 0.30
    assert pick_best(sweep_df, max_calls=100)["threshold"] == 0.7
    with pytest.raises(ValueError):
        pick_best(sweep_df, max_calls=10)
