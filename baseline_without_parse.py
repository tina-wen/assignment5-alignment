from pathlib import Path
import json
from cs336_alignment.vllm_utils import VLLMServer
import time
import pandas as pd

def get_alpaca_eval_data(data_dir):
    with open(data_dir,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_simple_safety_tests_data(data_dir):
    data_df = pd.read_csv(data_dir)
    data = []
    for _, row in data_df.iterrows():
        data.append({
            "instruction": row["prompts_final"],
        })
    return data


task_name = "alpaca_eval"

if task_name == "alpaca_eval":
    data_dir = 'data/alpaca_eval/alpaca_eval_gpt4_turbo.json'
    data = get_alpaca_eval_data(data_dir)
elif task_name == "simple_safety_tests":
    data_dir = 'data/simple_safety_tests/simple_safety_tests.csv'
    data = get_simple_safety_tests_data(data_dir)

prompt_dir = 'cs336_alignment/prompts_safety/zero_shot_system_prompt.prompt'
template = Path(prompt_dir).read_text(encoding='utf-8')

prompts = [template.format(**d) for d in data]

sampling_params = {
    "temperature": 0.0, 
    "top_p": 1.0, 
    "max_tokens": 512, 
    "seed": 42, 
    "stop": None, 
    "n": 1
}

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

for single_data,response in zip(data, responses):
    single_data["output"] = response
    if task_name == "alpaca_eval":
        single_data["generator"] = "llama-3.1-8b-base"

with open(f"{task_name}_output.json","w") as f:
    json.dump(data, f)

print(f"{task_name}吞吐率：{throughput} samples/s")