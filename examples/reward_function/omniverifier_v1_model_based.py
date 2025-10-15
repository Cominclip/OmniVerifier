import os
import random
import time
import re
from openai import OpenAI
from typing import Any
import json

base_url = ""
ak = ""
model_name = ""

client = OpenAI(
    base_url=base_url,
    api_key=ak,
)

ANSWER_JUDGE_PROMPT = '''You are an expert evaluator. Given a reference answer to a question and a student's response, your task is to determine whether the student's response aligns with the reference answer. If it does, respond with 'Yes'; otherwise, respond with 'No'. Please note that your response must be either 'Yes' or 'No'—no additional text is allowed.
<Question>
{}
<Student's Response>
{}
<Reference Answer>
{}
'''

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

def extract_detailed_judgement(response):
    """
    Extract detailed judgement from the response.
    """
    match = re.search(r'"explanation"\s*:\s*"([^"]*?)"', response)
    if match:
        return match.group(1)
    return None

def answer_judge(query, answer, ref_answer):
    max_retry_num = 5
    prompt = ANSWER_JUDGE_PROMPT.format(query, answer, ref_answer)

    for _ in range(max_retry_num):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
            )
            matched_results_raw = completion.choices[0].message.content
            matched_results_raw = matched_results_raw.strip().lower()
            assert matched_results_raw in ["yes", "no"], f"matched_results_raw {matched_results_raw}"
            if matched_results_raw == "yes":
                score = 1.0
            else:
                score = 0.0
            return score
        except Exception:
            sleep_time = random.random() * 10 + 5
            time.sleep(sleep_time)
            continue
    return 0.0

def format_reward(response: str) -> float:
    r = (response or "").strip()
    if r.startswith("<think>") and "</think>" in r:
        think, ans = r[7:].split("</think>", 1)
        if think.strip() and ans.strip() and "<think>" not in ans:
            return 1.0
    return 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = json.loads(ground_truth)
    prompt = answer['problem']
    match = re.search(r'"answer"\s*:\s*(true|false)', response, re.IGNORECASE)
    if match and response != None:
        extracted_value = match.group(1).lower() == "true"
        is_same = extracted_value == answer['answer']
        if is_same:
            if extracted_value:
                return 1.0
            else: 
                detailed_judgement = extract_detailed_judgement(response)
                prompt = prompt.split('otherwise')[0] + ' otherwise, briefly summarize the main error.'
                return answer_judge(prompt, detailed_judgement, answer['explanation'])
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
    