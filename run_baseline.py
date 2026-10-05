import os
from typing import Any, Callable

from cs336_alignment.data_loading import load_dataset, load_mmlu, get_prompts, load_simple_safety_tests
from cs336_alignment.response_scoring import gsm8k_reward_fn, mmlu_reward_fn, score_responses, dump_output
from cs336_alignment.vllm_utils import generate_with_vllm
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
# 我将要建立一个pipeline：

DATA_PATH = 'data/'
PROMPT_PATH = 'cs336_alignment/'
MODEL_PATH = '../../weights'
OUTPUT_PATH = './vLLM_outputs'

sampling_params = {
        "temperature": 0.0, 
        "max_tokens": 512, 
        "seed": 42, 
        "stop": None, 
        "n": 1
        }

prompt_sampling_params = {
    "temperature": 1.0, 
    "max_tokens": 512, 
    "seed": 42, 
    "stop": ["</answer>"], 
    "n": 1
    }

class TaskConfig:
    def __init__(
            self,
            # prompt/dataset/model加载路径
            prompt_type: str,
            prompt_name: str,
            dataset: str,
            datafile_name: str,
            model_name: str,
            # 数据加载函数
            loader_fn: Callable, 
            # vLLM推理参数
            sampling_params: dict[str, Any],
            gpu_memory_utilization: float = 0.5,
            # 评分函数
            score_fn: Callable | None = score_responses,
            reward_fn: Callable | None = None,
            extract_gt_fn: Callable | None = None,
            # 结果落盘设置
            dump_fn: Callable | None = None,
            file: str | None = None,
            generator: str | None = None,
            # 多重prompt
            outer_prompt_name: str | None = None,
            outer_prompt_key: str | None = None,
        ):
        self.data_dir = os.path.join(DATA_PATH, dataset, datafile_name)
        self.model_dir = os.path.join(MODEL_PATH, model_name)
        self.prompt_type = prompt_type
        self.prompt_dir = os.path.join(PROMPT_PATH, prompt_type, prompt_name+'.prompt')
        self.sampling_params = sampling_params
        self.loader_fn = loader_fn
        self.gpu_memory_utilization = gpu_memory_utilization
        self.score_fn = score_fn
        self.reward_fn = reward_fn
        self.extract_gt_fn = extract_gt_fn
        
        self.dump_fn = dump_fn
        if file:
            self.target_file = os.path.join(OUTPUT_PATH, file)
            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)
        self.generator = generator

        if outer_prompt_name:
            self.outer_prompt_dir = os.path.join(PROMPT_PATH, self.prompt_type, outer_prompt_name + '.prompt')
        self.outer_prompt_key = outer_prompt_key

    def run(self):
        data = self.loader_fn(self.data_dir)
        prompts = get_prompts(data, self.prompt_dir)
        if self.outer_prompt_key:
            prompts = [{self.outer_prompt_key: prompt} for prompt in prompts]
            prompts = get_prompts(prompts, self.outer_prompt_dir)

        responses, throughput = generate_with_vllm(
            self.model_dir,
            prompts,
            self.sampling_params,
            gpu_memory_utilization=self.gpu_memory_utilization,
            )
        
        if self.score_fn:
            acc = self.score_fn(data, responses, self.reward_fn, self.extract_gt_fn)
            if self.dump_fn:
                self.dump_fn(data, responses, scores = acc, generator = self.generator, file = self.target_file)

            if isinstance(acc[0], int):
                accuracy = sum(acc)/len(acc)
            elif isinstance(acc[0], dict):
                accuracy = {}
                for k,_ in acc[0].items():
                    accuracy[k] = []
                for score in acc:
                    for key, value in score.items():
                        accuracy[key].append(value)
                for k,v in accuracy.items():
                    accuracy[k] = sum(v)/len(v)
            print(f"推理评分{accuracy}")

        elif self.dump_fn:
            self.dump_fn(data, responses, generator = self.generator, file = self.target_file)

            
        print(f"吞吐率{throughput} samples/s")


promptingConfigs = [TaskConfig(
    prompt_type='prompts',
    prompt_name=prompt,
    dataset = 'gsm8k',
    datafile_name = 'test.jsonl',
    model_name='OLMo-2-0425-1B',
    sampling_params=prompt_sampling_params,
    loader_fn = load_dataset,
    reward_fn = question_only_reward_fn if 'question_only' in prompt else r1_zero_reward_fn,
    extract_gt_fn = lambda x: x["answer"].partition("####")[-1].strip(),
    dump_fn = dump_output,
    file = f'{prompt}_baseline.csv',
) for prompt in ['question_only', 'r1_zero', 'r1_zero_three_shot_gsm8k']]

mmluConfig = TaskConfig(
    prompt_type='prompts_safety',
    prompt_name='mmlu_zero_shot',
    dataset='mmlu',
    datafile_name='val',
    model_name='Llama-3.1-8B',
    sampling_params=sampling_params,
    loader_fn=load_mmlu,
    reward_fn=mmlu_reward_fn,
)

gsm8kConfig = TaskConfig(
    prompt_type='prompts_safety',
    prompt_name='gsm8k_zero_shot',
    dataset='gsm8k',
    datafile_name='test.jsonl',
    model_name='Llama-3.1-8B',
    sampling_params=sampling_params,
    loader_fn=load_dataset,
    reward_fn=gsm8k_reward_fn,
    outer_prompt_name = 'zero_shot_system_prompt',
    outer_prompt_key = 'instruction',
)

AlpacaEvalConfig = TaskConfig(
    prompt_type='prompts_safety',
    prompt_name='zero_shot_system_prompt',
    dataset='alpaca_eval',
    datafile_name='alpaca_eval_gpt4_turbo.json',
    model_name='Llama-3.1-8B',
    sampling_params=sampling_params,
    score_fn=None,
    loader_fn=load_dataset,
    dump_fn=dump_output,
    file='alpaca_eval_baseline.json',
    generator="llama-3.1-8b-base",
)

SimpleSafetyConfig = TaskConfig(
    prompt_type='prompts_safety',
    prompt_name='zero_shot_system_prompt',
    dataset='simple_safety_tests',
    datafile_name='simple_safety_tests.csv',
    model_name='Llama-3.1-8B',
    sampling_params=sampling_params,
    loader_fn=load_simple_safety_tests,
    score_fn=None,
    dump_fn=dump_output,
    file='simple_safety_tests_baseline.json',
)

task_configs = {
    'mmlu':mmluConfig, 
    'gsm8k': gsm8kConfig, 
    'alpaca_eval': AlpacaEvalConfig,
    'simple_safety_tests': SimpleSafetyConfig,
    'prompt': promptingConfigs,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="run_baseline")
    parser.add_argument("--task", type=str, choices = ['mmlu','gsm8k', 'alpaca_eval', 'simple_safety_tests', 'prompt'])
    args = parser.parse_args()

    task_config = task_configs[args.task]

    if isinstance(task_config, list):
        for tc in task_config:
            tc.run()
    else:
        task_config.run()
    