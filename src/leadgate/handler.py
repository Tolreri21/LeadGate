import json

from leadgate.serving import load_artifacts, predict_lead

_artifacts = None


def _get_artifacts():
    global _artifacts
    if _artifacts is None:
        _artifacts = load_artifacts()
    return _artifacts


def _response(status, body):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event, context):
    raw = event.get("body", event)
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        pipe, threshold = _get_artifacts()
        result = predict_lead(pipe, threshold, payload)
    except ValueError as e:
        return _response(400, {"error": str(e)})
    except Exception as e:
        return _response(500, {"error": str(e)})
    return _response(200, result)
