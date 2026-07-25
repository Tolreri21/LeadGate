import json

import pandas as pd
import pytest

from leadgate.serving import predict_lead


def test_predict_lead_matches_pipeline(fitted_pipe, lead):
    threshold = 0.11
    res = predict_lead(fitted_pipe, threshold, lead)

    direct = fitted_pipe.predict_proba(pd.DataFrame([lead]))[:, 1][0]
    assert res["probability"] == pytest.approx(direct)
    assert isinstance(res["subscribe"], bool)
    assert res["threshold"] == threshold
    json.dumps(res)


def test_unknown_category_survives(fitted_pipe, lead):
    lead = {**lead, "job": "spaceman"}
    res = predict_lead(fitted_pipe, 0.11, lead)
    assert 0.0 <= res["probability"] <= 1.0
