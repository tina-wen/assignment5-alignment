import re
from typing import Any

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