"""Extract the standalone JSON global_score used by both training stages."""

import json
import math


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def extract_global_score(response):
    try:
        answer = json.loads(response, parse_constant=_reject_constant)
        if not isinstance(answer, dict):
            raise ValueError("Final output must be a JSON object")
        value = answer["global_score"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("global_score must be a JSON number")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("global_score must be finite")
        if not 0 <= score <= 1:
            raise ValueError("global_score must be between 0 and 1")
        return {"score": score, "valid": True, "error": None}
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
        return {"score": None, "valid": False, "error": str(error)}
