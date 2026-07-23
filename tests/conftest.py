import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "age": [99, 33, 40, 55],
            "job": ["admin.", "student", "blue-collar", "retired"],
            "marital": ["single", "single", "married", "divorced"],
            "balance": [1200, 50, -300, 8000],
            "pdays": [5, -1, 100, -1],
            "day": [15, 3, 21, 9],
            "duration": [180, 95, 600, 42],
            "poutcome": ["unknown", "unknown", "success", "unknown"],
            "y": ["no", "yes", "yes", "no"],
        }
    )


@pytest.fixture
def y_true():
    return pd.Series([1, 1, 0, 0])


@pytest.fixture
def y_pred():
    return pd.Series([1, 0, 1, 0])


@pytest.fixture
def proba():
    return pd.Series([0.9, 0.2, 0.8, 0.1])


@pytest.fixture
def X():
    # Мини-матрица для make_preprocessor: str-категории (OHE), balance (yeo-johnson,
    # есть отрицательный), age (KBins n_bins=5 -> нужны разные значения).
    return pd.DataFrame(
        {
            "age": [22, 35, 48, 29, 57, 41, 63, 33],
            "job": [
                "admin.",
                "blue-collar",
                "retired",
                "student",
                "admin.",
                "technician",
                "retired",
                "student",
            ],
            "marital": [
                "single",
                "married",
                "divorced",
                "single",
                "married",
                "single",
                "married",
                "divorced",
            ],
            "balance": [1200, -300, 50, 8000, -50, 420, 15000, 0],
            "campaign": [1, 3, 2, 1, 5, 2, 1, 4],
        }
    )


@pytest.fixture
def sweep_df():
    # Готовая таблица свипа для тестов pick_best.
    # money максимален на строке t=0.30 (money=250) -> её ждём без лимита.
    # n_calls растёт вниз; при max_calls=250 отсекаются две нижние строки,
    # среди оставшихся лучший money=250 -> та же строка t=0.30.
    return pd.DataFrame(
        {
            "threshold": [0.10, 0.30, 0.50, 0.70],
            "money": [120, 250, 200, 90],
            "n_calls": [400, 250, 120, 40],
            "precision": [0.30, 0.55, 0.70, 0.80],
            "recall": [0.90, 0.70, 0.40, 0.15],
        }
    )
