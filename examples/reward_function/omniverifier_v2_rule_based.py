import re
from typing import Any
import json

def filter_thinking_part(response, eos_token=None):
    """
    Extract answer from the response.
    """
    response_start = 0
    success = False
    think_start = response.find('<think>', response_start)
    think_end = response.rfind('</think>', response_start)
    if think_start != -1 and think_end != -1 and think_start < think_end:
        response_start = think_end + len('</think>')
        success = True
    if eos_token is not None:
        response_end = response.find(eos_token, response_start)
    else:
        response_end = len(response)
    response = response[response_start:response_end]
    return response, success

def format_reward(response: str) -> float:
    r = (response or "").strip()
    if r.startswith("<think>") and "</think>" in r:
        think, ans = r[7:].split("</think>", 1)
        if think.strip() and ans.strip() and "<think>" not in ans:
            return 1.0
    return 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = json.loads(ground_truth)
    match = re.search(r'"answer"\s*:\s*(true|false)', response, re.IGNORECASE)
    if match and response != None:
        extracted_value = match.group(1).lower() == "true"
        is_same = extracted_value == answer['answer']
        if is_same:
            return 1.0
        else:
            return 0.0
    return 0.0

def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for univerifier.")
    
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  
        format_score = format_reward(response)
        response, _ = filter_thinking_part(response)  
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
    