import numpy as np

from leadgate.pipeline import make_preprocessor


def test_preprocessor_no_nan(X):
    # yeo-johnson должен пережить отрицательный balance без NaN (log1p бы упал).
    ct = make_preprocessor()
    out = ct.fit_transform(X)
    dense = out.toarray() if hasattr(out, "toarray") else out
    assert not np.isnan(dense).any()


def test_preprocessor_unknown_category(X):
    # handle_unknown="ignore": категория, не виденная на fit, не должна ронять transform.
    ct = make_preprocessor()
    ct.fit(X)

    X_new = X.iloc[[0]].copy()
    X_new["job"] = "spaceman"

    out = ct.transform(X_new)
    assert out.shape[0] == 1
