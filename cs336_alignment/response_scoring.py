
import re
from typing import Any, Callable
import pandas as pd
import json

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    match = re.search(
        r"the correct answer is[^A-Za-z0-9]*([ABCD])\b",
        model_output,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    options = mmlu_example.get("options", [])
    for i, option in enumerate(options):
        if re.search(r'\b' + re.escape(option.lower().strip()) + r'\b', model_output.lower()):
            return "ABCD"[i]

    match = re.search(r"\b([ABCD])\b",model_output)
    if match:
        return match.group(1).upper()

    return None

def parse_gsm8k_response(model_output: str) -> str | None:

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        model_output.replace(",",""),
        )
    if not numbers:
        return None
    return numbers[-1]

def gsm8k_reward_fn(data: dict, model_output: str) -> int:
    response = parse_gsm8k_response(model_output)
    gt = data["answer"].partition("####")[-1].strip()
    gt = gt.replace(",","")
    if response is not None:
        score = int(response == gt)
    else:
        score = 0
    return score

def mmlu_reward_fn(data: dict, model_output: str) -> int:
    response = parse_mmlu_response(data, model_output)
    gt = data.get("answer").upper()
    if response is not None:
        score = int(response == gt)
    else:
        score = 0
    return score

def score_responses(
    data: list[dict[str, Any]],
    model_outputs: list[str],
    reward_fn: Callable,
    extract_gt_fn: Callable | None = None,
    ):
    scores = []
    for d,model_output in zip(data, model_outputs):
        if extract_gt_fn: # 只针对question_only_reward_fn/r1_zero_reward_fn
            gt = extract_gt_fn(d)
            score = reward_fn(model_output, gt)
        else:
            score = reward_fn(d, model_output)
        scores.append(score)
    return scores

def dump_output(
        data: list[dict[str, Any]], 
        responses: list[str],
        file,
        scores: list[int] | list[dict[str, int]] | None = None, 
        generator: str | None = None,
        ):
    outputs = []
    for idx,(d,response) in enumerate(zip(data, responses)):
        o = d.copy()
        o["output"] = response
        if scores:
            o["score"] = scores[idx]
        if generator:
            o["generator"] = generator
        outputs.append(o)

    if file.endswith('.json'):
        with open(file,"w") as f:
            json.dump(outputs, f)
    elif file.endswith('.csv'):
        outputs_df = pd.DataFrame(outputs)
        outputs_df.to_csv(file, index=False)
