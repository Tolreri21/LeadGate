from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PowerTransformer,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def make_preprocessor():
    ct = ColumnTransformer(
        [
            (
                "Categorical columns",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include="str"),
            ),
            ("Balance", PowerTransformer(method="yeo-johnson"), ["balance"]),
            (
                "age",
                KBinsDiscretizer(n_bins=5, strategy="quantile", encode="onehot"),
                ["age"],
            ),
        ],
        remainder="passthrough",
    )
    return ct


def make_cv():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cv


def make_champion_pipeline(class_weight=None):
    log_reg = LogisticRegression(max_iter=1000, class_weight=class_weight)

    pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", log_reg)])
    return pipeline
