import json
import math
from typing import Any, Optional, Tuple, Union


REWARD_NAME = "omniverifier_c1"
REWARD_TYPE = "batch"

FORMAT_WEIGHTS = {"global_score": 0.1, "fine_grained_verification": 0.1}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _parse_response(response: str) -> Tuple[Optional[dict[str, Any]], bool]:
    """Require a standalone JSON object, matching the reward-service prompts."""
    try:
        answer = json.loads(response, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError):
        return None, False

    if not isinstance(answer, dict):
        return None, False

    return answer, True


def _as_score(value: Any) -> Optional[float]:
    """Accept finite numeric scores, including numeric string ground truths."""
    if isinstance(value, bool):
        return None

    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return score if math.isfinite(score) else None


def _as_answer(value: Any) -> Optional[bool]:
    """Accept JSON booleans and True/False strings, ignoring case and spaces."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("true", "false"):
            return value == "true"

    return None


def _ground_truth_value(ground_truth: Any, key: str) -> Any:
    """Ground truths may be scalar labels, dictionaries, or JSON objects."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth, parse_constant=_reject_json_constant)
        except (ValueError, RecursionError):
            pass

    return ground_truth.get(key) if isinstance(ground_truth, dict) else ground_truth


def _prediction_value(answer: Optional[dict[str, Any]], task: str) -> Union[float, bool, None]:
    if answer is None:
        return None

    if task == "global_score":
        value = answer.get("global_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        score = _as_score(value)
        return score if score is not None and 0 <= score <= 1 else None

    return _as_answer(answer.get("answer"))


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Compute mixed-task batch rewards using each row's explicit ``task``.

    global_score: 0.1 * format + 0.9 * (1 - abs(prediction - ground_truth)).
    fine_grained_verification: 0.1 * format + 0.9 * (prediction == ground_truth).
    Format requires one standalone JSON object containing a valid task-specific
    prediction: global_score (a number in [0, 1]) or answer (a binary value).
    Invalid predictions/labels receive zero accuracy reward.
    """
    scores = []
    for reward_input in reward_inputs:
        task = reward_input.get("task")
        if task not in FORMAT_WEIGHTS:
            raise ValueError(
                f"Unsupported or missing task: {task!r}; expected 'global_score' or 'fine_grained_verification'."
            )

        answer, has_json_format = _parse_response(reward_input["response"])
        prediction = _prediction_value(answer, task)
        format_score = float(has_json_format and prediction is not None)

        if task == "global_score":
            target = _as_score(_ground_truth_value(reward_input["ground_truth"], "global_score"))
            accuracy_score = 1.0 - abs(prediction - target) if prediction is not None and target is not None else 0.0
        else:
            target = _as_answer(_ground_truth_value(reward_input["ground_truth"], "answer"))
            accuracy_score = float(prediction is not None and target is not None and prediction == target)

        format_weight = FORMAT_WEIGHTS[task]
        scores.append(
            {
                "overall": format_weight * format_score + (1.0 - format_weight) * accuracy_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
