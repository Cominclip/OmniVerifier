import re
from collections import Counter
from typing import Any, Dict, List


def filter_thinking_part(response: str, eos_token=None):
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", response or "", flags=re.DOTALL)
    return cleaned.strip(), True


def format_reward(response: str) -> float:
    r = (response or "").strip()
    start_tag = "<thinking>"
    end_tag = "</thinking>"

    if r.startswith(start_tag) and end_tag in r:
        think, ans = r[len(start_tag):].split(end_tag, 1)
        if think.strip() and ans.strip() and start_tag not in ans and end_tag not in ans:
            return 1.0

    return 0.0


def extract_pred_bool(response: str):
    try:
        match = re.search(r'"answer"\s*:\s*(true|false)', response, re.IGNORECASE)
        if not match:
            return None
        return match.group(1).lower() == "true"
    except Exception:
        return None


def extract_pred_bbox(response: str):
    try:
        match = re.search(r'"bbox"\s*:\s*\[([^\]]*)\]', response, re.IGNORECASE)
        if not match:
            return None

        nums = match.group(1).split(",")
        if len(nums) != 4:
            return None

        return [float(n.strip()) for n in nums]
    except Exception:
        return None


def judge_reward(pred_bool: bool, gt_bool: bool) -> float:
    if pred_bool is None:
        return 0.0
    return 1.0 if pred_bool == gt_bool else 0.0


def compute_iou(pred_bbox, gt_bbox) -> float:
    if pred_bbox is None or gt_bbox is None:
        return 0.0

    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = max(0.0, (px2 - px1) * (py2 - py1))
    gt_area = max(0.0, (gx2 - gx1) * (gy2 - gy1))

    union_area = pred_area + gt_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def iou_gate_reward(iou: float, threshold: float = 0.6) -> float:
    return 1.0 if iou >= threshold else 0.0


def get_task_weight(task: str, task_counts: Counter, batch_size: int) -> float:
    task_count = task_counts[task]
    if task_count <= 0:
        return 1.0
    return batch_size / task_count


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    iou_threshold: float = 0.6,
) -> List[Dict[str, float]]:
    valid_tasks = {"Judge", "Grounding"}

    for reward_input in reward_inputs:
        task = reward_input["task"]
        if task not in valid_tasks:
            raise ValueError(f"Unknown task type: {task}")

    batch_size = len(reward_inputs)
    task_counts = Counter(reward_input["task"] for reward_input in reward_inputs)
    scores = []

    for reward_input in reward_inputs:
        response = reward_input["response"]
        task = reward_input["task"]
        gt = reward_input["ground_truth"]

        format_score = format_reward(response)
        clean_response, _ = filter_thinking_part(response)

        pred_bool = extract_pred_bool(clean_response)
        pred_bbox = extract_pred_bbox(clean_response)

        judge_score = 0.0
        iou_gate = 0.0

        if task == "Judge":
            gt_bool = gt["bool"]
            judge_score = judge_reward(pred_bool, gt_bool)
            overall = format_weight * format_score + (1.0 - format_weight) * judge_score

        elif task == "Grounding":
            gt_bbox = gt.get("bbox", None)
            iou = compute_iou(pred_bbox, gt_bbox)
            iou_gate = iou_gate_reward(iou, threshold=iou_threshold)
            overall = format_weight * format_score + (1.0 - format_weight) * iou_gate

        task_weight = get_task_weight(task, task_counts, batch_size)

        if task == "Judge":
            judge_score = judge_score * task_weight

        elif task == "Grounding":
            iou_gate = iou_gate * task_weight

        scores.append(
            {
                "overall": float(overall),
                "format": float(format_score),
                "judge": float(judge_score),
                "iou_gate": float(iou_gate),
            }
        )

    return scores