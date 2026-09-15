from pathlib import Path
import pandas as pd
import os
from cs336_alignment.vllm_utils import VLLMServer
from supplement import parse_mmlu_response, parse_gsm8k_response
import time
import json
from typing import Any

# mmlu数据集的读取和组织方式
def get_mmlu_prompts(data_dir: str, prompt_dir: str):
    mmlu_examples = []
    for name in os.listdir(data_dir):
        data = pd.read_csv(data_dir+name,header=None)
        for row in data.itertuples():
            mmlu_examples.append({
                "subject":name.split('_val')[0],
                "question":row[1],
                "options":list(row[2:-1]),
                "answer":row[-1],
                })
    template = Path(prompt_dir).read_text(encoding='utf-8')
    prompts = [template.format(**exp) for exp in mmlu_examples]
    return mmlu_examples, prompts

# gsm8K数据集的读取
def get_gsm8k_prompts(
    data_dir: str, # 数据集路径
    prompt_dir: str, # 使用question占位符的prompt模版
    outer_prompt_dir: str,
):
    with open(data_dir, 'r', encoding = 'utf-8') as f:
        data = [json.loads(line) for line in f]

    template = Path(prompt_dir).read_text(encoding='utf-8')
    prompts = [{"instruction":template.format(**d)} for d in data]
    outer_template = Path(outer_prompt_dir).read_text(encoding='utf-8')
    prompts = [outer_template.format(**p) for p in prompts]
    return data, prompts

def score_responses(
    data: list[dict[str, Any]],
    responses: list[str],
    template_name: str,
    ):

    def _get_response_and_gt(single_data: dict[str, Any], single_response: str, template_name: str) -> tuple[str | None, str]:
        if template_name == 'mmlu':
            parsed_response = parse_mmlu_response(single_data, single_response)
            gt = single_data.get("answer").upper()
            if parsed_response is not None:
                parsed_response = parsed_response.upper()
            return parsed_response, gt
        elif template_name == 'gsm8k':
            parsed_response = parse_gsm8k_response(single_response)
            gt = single_data["answer"].partition("####")[-1].strip()
            return parsed_response, gt.replace(",","")

    scores = []
    for d,response in zip(data, responses):
        parsed_response, gt = _get_response_and_gt(d, response, template_name)
        if parsed_response is not None:
            score = int(parsed_response == gt)
        else:
            score = 0
        d["generations"] = response
        d["parsed"] = parsed_response
        d["score"] = score
        scores.append(score) 

    results = pd.DataFrame(data)
    return results, scores


if __name__ == "__main__":

    sampling_params = {
        "temperature": 0.0, 
        "top_p": 1.0, 
        "max_tokens": 512, 
        "seed": 42, 
        "stop": None, 
        "n": 1
        }

    # template_name = 'mmlu'
    # prompt_dir='cs336_alignment/prompts_safety/mmlu_zero_shot.prompt'
    # data_dir='data/mmlu/val/'
    # data, prompts = get_mmlu_prompts(data_dir, prompt_dir)

    template_name = 'gsm8k'
    prompt_dir='cs336_alignment/prompts_safety/gsm8k_zero_shot.prompt'
    data_dir='data/gsm8k/test.jsonl'
    outer_prompt_dir = 'cs336_alignment/prompts_safety/zero_shot_system_prompt.prompt'
    data, prompts = get_gsm8k_prompts(data_dir, prompt_dir, outer_prompt_dir)

    # 基于prompts批量推理
    vllm_server = VLLMServer(
        model_id='../../weights/Llama-3.1-8B',
        seed=sampling_params.get("seed"),
        gpu=0,
        gpu_memory_utilization=0.5
        )
    vllm_server.start()
    start_time = time.perf_counter()
    responses = vllm_server.generate_completions(prompts,sampling_params=sampling_params)
    responses = [r.text for r in responses]
    generation_seconds = time.perf_counter() - start_time
    throughput = len(prompts) / generation_seconds

    # 根据数据集（ground_truths）评估生成response质量
    results, scores = score_responses(data, responses, template_name)

    results.to_csv(f"{template_name}_baseline_results.csv", index = False)
    print(f"{template_name}评分为：{sum(scores)/len(scores)}，吞吐率：{throughput} samples/s")