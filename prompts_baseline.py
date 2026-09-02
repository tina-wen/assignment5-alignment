from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from pathlib import Path
import json
import os
import random

PROMPTS_DIR='cs336_alignment/prompts/'

vllm_server = VLLMServer(model_id = './OLMo-2-0425-1B/')
vllm_server.start()
sampling_params = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512, "seed": 42, "stop": ["</answer>"], "n": 1}


with open('./data/gsm8k/test.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]


for prompt in os.listdir(PROMPTS_DIR):
    prompt_template = Path(PROMPTS_DIR+prompt).read_text(encoding='utf-8')
    formatted_prompts = [prompt_template.format(question=q['question']) for q in data]
    responses = vllm_server.generate_completions(formatted_prompts, sampling_params=sampling_params)
    
    reward_fn = question_only_reward_fn if 'question_only' in prompt else r1_zero_reward_fn
    
    format_scores,total_scores = [],[] 

    for i in range(len(data)):
        ground_truth = data[i]["answer"].partition("####")[-1].strip()
        score = reward_fn(responses[i].text, ground_truth)
        format_scores.append(score['format_reward'])
        total_scores.append(score['reward'])

    # # if prompt == 'r1_zero.prompt':
    # if 'r1_zero_three_shot' in prompt:
    #     idxs = random.sample((range(len(data))),10)
    #     for idx in idxs:
    #         print(f"\n\n模型输出：{responses[idx].text}\n 正确答案：{data[idx]["answer"].partition("####")[-1].strip()}\n 格式\
    #         奖励：{format_scores[idx]}\n 正确性奖励：{total_scores[idx]}")

    print(f"\n\n使用{prompt}模板：")
    print(f"格式和正确性奖励均为1的样本数量：{sum([int(x+y==2) for x,y in zip(format_scores,total_scores)])}")
    print(f"格式奖励为1但正确性奖励为0的样本数量：{sum([int(x==1 and y==0)for x,y in zip(format_scores,total_scores)])}")
    print(f"格式奖励和正确性奖励均为0的样本数量：{sum([int(x+y==0) for x,y in zip(format_scores,total_scores)])}\n\n")