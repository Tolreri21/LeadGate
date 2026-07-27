import pandas as pd
import pytest

from leadgate.pipeline import make_champion_pipeline


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
    return pd.DataFrame(
        {
            "threshold": [0.10, 0.30, 0.50, 0.70],
            "money": [120, 250, 200, 90],
            "n_calls": [400, 250, 120, 40],
            "precision": [0.30, 0.55, 0.70, 0.80],
            "recall": [0.90, 0.70, 0.40, 0.15],
        }
    )


@pytest.fixture
def train_df():
    return pd.DataFrame(
        {
            "age": [22, 35, 48, 29, 57, 41, 63, 33],
            "balance": [1200, -300, 50, 8000, -50, 420, 15000, 0],
            "campaign": [1, 3, 2, 1, 5, 2, 1, 4],
            "previous": [0, 1, 0, 2, 0, 1, 0, 3],
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
            "education": [
                "secondary",
                "primary",
                "tertiary",
                "secondary",
                "tertiary",
                "secondary",
                "primary",
                "tertiary",
            ],
            "default": ["no", "no", "no", "yes", "no", "no", "no", "no"],
            "housing": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
            "loan": ["no", "no", "yes", "no", "no", "yes", "no", "no"],
            "contact": [
                "cellular",
                "telephone",
                "cellular",
                "unknown",
                "cellular",
                "cellular",
                "telephone",
                "cellular",
            ],
            "month": ["may", "jun", "jul", "aug", "may", "oct", "mar", "nov"],
            "poutcome": [
                "unknown",
                "failure",
                "success",
                "unknown",
                "unknown",
                "success",
                "failure",
                "unknown",
            ],
        }
    )


@pytest.fixture
def y_bin():
    return pd.Series([0, 1, 1, 0, 1, 0, 1, 0])


@pytest.fixture
def fitted_pipe(train_df, y_bin):
    pipe = make_champion_pipeline()
    pipe.fit(train_df, y_bin)
    return pipe


@pytest.fixture
def lead(train_df):
    return train_df.iloc[0].to_dict()
