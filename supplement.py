import re
from typing import Any
import json
import torch
import random
from pathlib import Path

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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_path: str, seq_length: int, shuffle: bool):

        template = Path('cs336_alignment/prompts_safety/alpaca_sft.prompt').read_text(encoding='utf-8').strip()
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]

        # prompts: list[str]
        prompts = []
        for d in data:
            d['instruction'] = d.pop('prompt')
            prompts.append(template.format(**d))
        if shuffle:
            random.shuffle(prompts)

        tokens = []
        for prompt in prompts:
            tokens.extend(tokenizer.encode(prompt))
            tokens.append(tokenizer.eos_token_id)

        inputs = tokens[:-1]
        labels = tokens[1:]
        num_chunks = len(inputs) // seq_length
        self.n = num_chunks

        inputs = [inputs[i*seq_length:(i+1)*seq_length] for i in range(num_chunks)]
        labels = [labels[i*seq_length:(i+1)*seq_length] for i in range(num_chunks)]

        self.inputs = torch.tensor(inputs, dtype=torch.int64)
        self.labels = torch.tensor(labels, dtype=torch.int64)


    def __len__(self):
        return self.n

    def __getitem__(self,i):
        return {"input_ids": self.inputs[i,:], "labels": self.labels[i,:]}


def compute_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    from function import get_response_log_probs
    device = next(lm.parameters()).device
    data = [
        {"instruction": prompt, "response": response_chosen},
        {"instruction": prompt, "response": response_rejected},
        ]

    template = Path('./cs336_alignment/prompts_safety/alpaca_sft.prompt').read_text(encoding='utf-8').strip()
    # 计算prompt的长度，隐式获得response_mask
    prefix = template.format(
        instruction = prompt,
        response = '',
    )
    prefix_len = len(tokenizer.encode(prefix))

    # 计算成对log_prob
    inputs = [template.format(**d) for d in data]

    log_prob_diffs = [] # [chosen_diff, rejected_diff]
    for input in inputs:
        tokens = tokenizer.encode(input)
        tokens.append(tokenizer.eos_token_id)
        input_id,label = tokens[:-1], tokens[1:]
        log_prob = get_response_log_probs(
            lm, 
            torch.tensor(input_id, dtype = torch.int64).unsqueeze(0).to(device), 
            torch.tensor(label, dtype = torch.int64).unsqueeze(0).to(device),
            )['log_probs']
        ref_log_prob = get_response_log_probs(
            lm_ref, 
            torch.tensor(input_id, dtype = torch.int64).unsqueeze(0).to(device), 
            torch.tensor(label, dtype = torch.int64).unsqueeze(0).to(device),
            )['log_probs']
        log_prob_diff = log_prob[:, prefix_len-1:].sum() - ref_log_prob[:, prefix_len-1:].sum()
        log_prob_diffs.append(log_prob_diff)

    return -torch.log(torch.sigmoid((log_prob_diffs[0] - log_prob_diffs[1]) * beta))


