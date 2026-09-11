from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from pathlib import Path
import json
from function import grpo_train_step, get_response_log_probs, tokenize_prompt_and_output
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from typing import Literal

def load_prompt_gt_pairs(prompt_template, data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    prompts = [prompt_template.format(question=q['question']) for q in data]
    ground_truths = [q["answer"].partition("####")[-1].strip() for q in data]
    return prompts, ground_truths


def validate(vllm_server, prompts, ground_truths, sampling_params, reward_fn):
    val_sampling_params = sampling_params.copy()
    val_sampling_params["n"] = 1
    responses = vllm_server.generate_completions(prompts, sampling_params=val_sampling_params)
    format_scores,total_scores = [],[] 

    for r,gt in zip(responses, ground_truths):
        score = reward_fn(r.text, gt)
        format_scores.append(score['format_reward'])
        total_scores.append(score['reward'])

    return sum(format_scores)/len(format_scores), sum(total_scores)/len(total_scores)

def RLTrain(
        model_path: str,
        prompts_path: str,
        train_path: str,
        valid_path: str,
        reward_fn, 
        device: str,
        group_size: int,
        rollout_n_prompts: int,
        train_n_prompts: int,
        off_policy: bool, 
        gradient_accumulation_steps: int,
        valid_step: int,
        max_grad_norm: float = 1.0,
        seed: int = 42, 
        importance_reweighting_method: str = 'none',
        cliprange: float | None = None,
        baseline: Literal["mean", "none"] = "mean",
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
        loss_normalization: Literal["sequence","constant"] = "sequence",
        normalization_constant: int | None = None,
        learning_rate: float = 1e-05,
        weight_decay: float = 0.0,
        grpo_name: str | None = None,
    ):

    assert off_policy or (rollout_n_prompts == train_n_prompts and importance_reweighting_method == "none"), "on-policy下，一轮rollout处理的样本数必须等于一轮训练更新消耗的样本数"

    sampling_params = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512, "seed": seed, "stop": ["</answer>"], "n": group_size, "include_stop_str_in_output":True}

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    prompt_template = Path(prompts_path).read_text(encoding='utf-8')
    trained_prompts, trained_gt = load_prompt_gt_pairs(prompt_template, train_path)
    valid_prompts, valid_gt = load_prompt_gt_pairs(prompt_template, valid_path)


    vllm_server = VLLMServer(model_id=model_path, gpu=0, seed=seed, gpu_memory_utilization=0.5)
    vllm_server.start()

    vllm_server.init_weight_sync(device)    
    vllm_server.sync_policy_weights(model)

    policy_name = "offpolicy" if off_policy else "onpolicy"
    wandb_name = f"{policy_name}_{importance_reweighting_method}"
    if grpo_name is not None:
        wandb_name = f"{policy_name}_{grpo_name}"
    run = wandb.init(
        project=f"cs336-ass5-{policy_name}", 
        name=wandb_name,
        group=f"{policy_name}_{importance_reweighting_method}",
        )

    total_step = len(trained_prompts) // rollout_n_prompts
    train_step = rollout_n_prompts // train_n_prompts
    step = 0
    while step < total_step:
        # 推理生成responses
        batched_prompts = trained_prompts[step*rollout_n_prompts:(step+1)*rollout_n_prompts]
        batched_ground_truths = trained_gt[step*rollout_n_prompts:(step+1)*rollout_n_prompts]
        responses = vllm_server.generate_completions(batched_prompts, sampling_params=sampling_params)
        rollout_responses = [r.text for r in responses]
        # 按照group_size对齐prompt response和ground_truths
        repeated_prompts = [p for p in batched_prompts for _ in range(group_size)]
        repeated_ground_truths = [gt for gt in batched_ground_truths for _ in range(group_size)]

        # 计算本轮rollout生成的old_log_probs
        # 按照mini_train_step循环，避免出现old_log_probs和policy_log_probs的shape不一致问题
        if off_policy:
            old_log_probs = []
            for mini_train_step in range(train_step):
                start, end = mini_train_step*train_n_prompts*group_size, (mini_train_step+1)*train_n_prompts*group_size
                mini_prompts = repeated_prompts[start:end]
                mini_responses = rollout_responses[start:end]

                inputs = tokenize_prompt_and_output(mini_prompts, mini_responses, tokenizer)    
                input_ids, labels, _ = inputs["input_ids"].to(device), inputs["labels"].to(device), inputs["response_mask"].to(device)

                with torch.no_grad():
                    old_log_prob = get_response_log_probs(model, input_ids, labels)["log_probs"]
                old_log_probs.append(old_log_prob)

        mini_train_step = 0
        while mini_train_step < train_step:
            start, end = mini_train_step*train_n_prompts*group_size, (mini_train_step+1)*train_n_prompts*group_size
            mini_prompts = repeated_prompts[start:end]
            mini_ground_truths = repeated_ground_truths[start:end]
            mini_responses = rollout_responses[start:end]
            if off_policy:
                mini_old_log_probs = old_log_probs[mini_train_step]

            # grpo_train
            total_loss,grad_norm,info = grpo_train_step(
                model,
                tokenizer,
                optimizer,
                gradient_accumulation_steps,
                max_grad_norm,
                reward_fn,
                mini_prompts,
                mini_responses,
                mini_ground_truths,
                group_size,
                baseline=baseline,
                advantage_normalizer=advantage_normalizer,
                importance_reweighting_method = importance_reweighting_method,
                old_log_probs=mini_old_log_probs if off_policy else None,
                cliprange = cliprange,
                loss_normalization=loss_normalization,
                normalization_constant=normalization_constant,
            )
            wandb.log({
                "loss": total_loss,
                "grad_norm": grad_norm,
                "clip_frac": info.get("clip_frac",None),
                "train/response_len": info.get("response_len",None),
                "train/mean_rewards": info.get("mean_rewards",None),
            })
            mini_train_step += 1

        # 更新vllm权重
        vllm_server.sync_policy_weights(model)

        # 验证集测试
        if step % valid_step == 0:
            format_score, acc_score = validate(vllm_server, valid_prompts, valid_gt, sampling_params, reward_fn)
            wandb.log({
                "val/format_score": format_score,
                "val/acc_score": acc_score,
            })

        step += 1

    vllm_server.stop()
    run.finish()

if __name__ == "__main__":
    MODEL_PATH='./OLMo-2-0425-1B/'
    PROMPTS_DIR='cs336_alignment/prompts/r1_zero.prompt'
    TRAIN_DIR='./data/gsm8k/train.jsonl'
    VALID_DIR='./data/gsm8k/test.jsonl'

    seed_list = [42,123,2024,0]

    for seed in seed_list:
        RLTrain(
            MODEL_PATH,
            PROMPTS_DIR,
            TRAIN_DIR,
            VALID_DIR,
            r1_zero_reward_fn,
            device = 'cuda:1',
            group_size = 8,
            rollout_n_prompts = 4,
            train_n_prompts=1,
            off_policy=True,
            gradient_accumulation_steps = 4,
            valid_step = 5,
            max_grad_norm = 1,
            seed=seed,
            importance_reweighting_method="grpo",
            cliprange=0.2, # 3e-4 for gspo
            )