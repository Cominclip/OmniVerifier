import re
from typing import Any, List, Dict


def filter_thinking_part(response: str, eos_token=None):
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", response or "", flags=re.DOTALL)
    return cleaned.strip(), True


def format_reward(response: str) -> float:
    r = (response or "").strip()
    start_tag = "<thinking>"
    end_tag = "</thinking>"

    if r.startswith(start_tag) and end_tag in r:
        think, ans = r[len(start_tag):].split(end_tag, 1)
        if think.strip() and ans.strip() and start_tag not in ans:
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
    if pred_bool is None or gt_bool is None:
        return 0.0
    return 1.0 if pred_bool == gt_bool else 0.0


def iou_reward(pred_bbox, gt_bbox) -> float:
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


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    iou_threshold: float = 0.6,
) -> List[Dict[str, float]]:
    batch_size = len(reward_inputs)
    false_count = sum(
        1 for reward_input in reward_inputs
        if reward_input["ground_truth"].get("bool", None) is False
    )
    false_weight = batch_size / false_count if false_count > 0 else 1.0

    scores = []

    for reward_input in reward_inputs:
        response = reward_input["response"]
        gt = reward_input["ground_truth"]

        format_score = format_reward(response)
        clean_response, _ = filter_thinking_part(response)

        pred_bool = extract_pred_bool(clean_response)
        pred_bbox = extract_pred_bbox(clean_response)

        gt_bool = gt.get("bool", None)
        gt_bbox = gt.get("bbox", None)

        judge_score = judge_reward(pred_bool, gt_bool)
        iou_gate = 0.0

        if gt_bool is False and gt_bbox:
            raw_iou = iou_reward(pred_bbox, gt_bbox)
            iou_gate = 1.0 if raw_iou >= iou_threshold else 0.0

        if gt_bool is True:
            overall = format_weight * format_score + (1.0 - format_weight) * judge_score
            logged_iou_gate = 0.0
        else:
            overall = format_weight * format_score + (1.0 - format_weight) * judge_score * iou_gate
            logged_iou_gate = iou_gate * false_weight

        scores.append(
            {
                "overall": float(overall),
                "format": float(format_score),
                "judge": float(judge_score),
                "iou_gate": float(logged_iou_gate),
            }
        )

    return scores