import torch
from typing import Literal, Callable, Dict

def tokenize_prompt_and_output(
        prompt_strs: list[str], 
        output_strs: list[str], 
        tokenizer,
        ) -> dict[str, torch.Tensor]:

    batch_size = len(prompt_strs)
    samples, prompt_lens = [],[]
    for x,y in zip(prompt_strs, output_strs):
        prompt = tokenizer.encode(x)
        output = tokenizer.encode(y)
        prompt_lens.append(len(prompt))
        sample = prompt + output
        samples.append(sample)

    sample_lens = [len(s) for s in samples]
    max_seq_len = max(sample_lens)
    batch = tokenizer.pad(
        [{"input_ids": sample} for sample in samples],
        padding=True,
        return_tensors="pt",
    ) 
    input_ids = batch["input_ids"][:,:-1]
    labels = batch["input_ids"][:,1:]

    positions = torch.arange(max_seq_len).unsqueeze(0) # (1,max_seq_len)
    prompt_lens = torch.tensor(prompt_lens).unsqueeze(1) # (batch_size,1)
    output_ends = torch.tensor(sample_lens).unsqueeze(1) # (batch_size,1)
    response_mask = (positions >= prompt_lens) & (positions < output_ends)
    response_mask = response_mask[:,1:]
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

def get_response_log_probs(
        model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids) # B,S,V
    logits = logits.logits
    norm_logits = logits - torch.max(logits,dim = -1, keepdim = True).values
    log_probs = norm_logits - torch.logsumexp(norm_logits, dim=-1, keepdim = True)
    if return_token_entropy:
        entropy = -torch.sum(torch.exp(log_probs) * log_probs,dim = -1)
    log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if not return_token_entropy:
        return {"log_probs": log_probs}
    return {"log_probs": log_probs, "token_entropy": entropy}

def compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths):
    raw_rewards, format_rewards = [],[]
    for res, gt in zip(rollout_responses, repeated_ground_truths):
        rewards = reward_fn(res, gt)
        raw_rewards.append(rewards["reward"])
        format_rewards.append(rewards["format_reward"])
    metadata = {"mean_rewards":sum(raw_rewards)/len(raw_rewards), "mean_format_rewards": sum(format_rewards)/len(format_rewards)}
    return torch.tensor(raw_rewards),metadata

def compute_group_normalized_rewards_grpo(
        raw_rewards: torch.Tensor,
        group_size: int,
        baseline: Literal["mean","none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std","mean","none"] = "std",
):
    advantages = raw_rewards.reshape(-1, group_size)
    mean_advantages = torch.mean(advantages, dim = 1, keepdim = True)
    std_advantages = torch.std(advantages, dim = 1, keepdim = True)
    if baseline == "mean":
        advantages = advantages - mean_advantages
    if advantage_normalizer == "std":
        advantages = advantages / (std_advantages + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (mean_advantages + advantage_eps)
    return advantages.flatten(), {}

def compute_policy_gradient_loss(
  raw_rewards_or_advantages: torch.Tensor,
  policy_log_probs: torch.Tensor,
  importance_reweighting_method: Literal["none","noclip","grpo","gspo"] = "none",
  old_log_probs: torch.Tensor | None = None,
  cliprange: float | None = None,
  response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = raw_rewards_or_advantages.reshape(-1,1)
    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = advantages * policy_log_probs
        return -per_token_policy_gradient_loss, {}

    if importance_reweighting_method == "noclip":
        resample_ratio = torch.exp(policy_log_probs - old_log_probs)
        unclipped_adv = advantages * resample_ratio
        return -unclipped_adv, {}

    if importance_reweighting_method == "grpo":
        resample_ratio = torch.exp(policy_log_probs - old_log_probs)
        unclipped_adv = advantages * resample_ratio

        clipped_tokens = (resample_ratio > 1+cliprange) | (resample_ratio < 1-cliprange)
        num_clip_token = torch.sum(clipped_tokens*response_mask)
        resample_ratio = torch.clamp(resample_ratio, min=1-cliprange, max=1+cliprange)
        clipped_adv = advantages * resample_ratio
        return -torch.minimum(unclipped_adv, clipped_adv), {"num_clip_token": num_clip_token}

    if importance_reweighting_method == "gspo":
        resample_ratio = torch.exp(
            torch.sum(
                (policy_log_probs - old_log_probs) * response_mask, 
                dim = -1,
                keepdim=True,
                ) 
                / torch.maximum(torch.sum(response_mask, dim = -1, keepdim=True),torch.ones((response_mask.shape[0],1),device=response_mask.device)))
        unclipped_adv = advantages * resample_ratio
        clipped_sample = (resample_ratio > 1+cliprange) | (resample_ratio < 1-cliprange)
        num_clip_sample = torch.sum(clipped_sample)

        resample_ratio = torch.clamp(resample_ratio, min=1-cliprange, max=1+cliprange)
        clipped_adv = advantages * resample_ratio
        return -torch.minimum(unclipped_adv, clipped_adv).expand_as(old_log_probs), {"num_clip_sample": num_clip_sample}

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence","constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor: 
    per_token_loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        loss = torch.sum(per_token_loss, dim = 1) / torch.maximum(torch.sum(mask, dim = 1), torch.ones((mask.shape[0],1),device=mask.device))
        loss = torch.mean(loss)
        return loss
    return torch.sum(per_token_loss) / normalization_constant

def grpo_train_step(
    model,
    tokenizer,
    optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    # 分词
    inputs = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)    
    # 看起来是计算policy_logprob
    device = next(model.parameters()).device
    input_ids, labels, response_mask = inputs["input_ids"].to(device), inputs["labels"].to(device), inputs["response_mask"].to(device)

    raw_rewards,mean_rewards_info = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages,_ = compute_group_normalized_rewards_grpo(raw_rewards.to(device), group_size, baseline, advantage_eps, advantage_normalizer)

    if baseline == "none":
        mask = advantages != 0
        advantages = advantages[mask]
        input_ids = input_ids[mask]
        labels = labels[mask]
        response_mask = response_mask[mask]
        if old_log_probs is not None:
            old_log_probs = old_log_probs[mask]
        if advantages.shape[0] < gradient_accumulation_steps:
            return 0,0,{}

    batch_size = input_ids.shape[0]
    micro_batch_size = batch_size // gradient_accumulation_steps

    optimizer.zero_grad(set_to_none=True)
    step, total_loss = 0, 0

    if importance_reweighting_method in ["grpo","gspo"]: 
        num_clips = 0

    while step < gradient_accumulation_steps:

        micro_input_ids = input_ids[step*micro_batch_size:(step+1)*micro_batch_size,:]
        micro_labels = labels[step*micro_batch_size:(step+1)*micro_batch_size,:]
        micro_response_mask = response_mask[step*micro_batch_size:(step+1)*micro_batch_size,:]

        entropy = get_response_log_probs(model,micro_input_ids,micro_labels)
        policy_log_probs = entropy["log_probs"]
        # 计算优势
        micro_advantages = advantages[step*micro_batch_size:(step+1)*micro_batch_size]
        if old_log_probs is not None:
            micro_old_logprobs = old_log_probs[step*micro_batch_size:(step+1)*micro_batch_size,:].to(device)
        else:
            micro_old_logprobs = None
        # 计算损失
        per_token_loss,clip_info = compute_policy_gradient_loss(micro_advantages, policy_log_probs, importance_reweighting_method, micro_old_logprobs, cliprange, micro_response_mask)

        if importance_reweighting_method == "grpo": 
            num_clips += clip_info["num_clip_token"]
        elif importance_reweighting_method == "gspo":
            num_clips += clip_info["num_clip_sample"]

        loss = aggregate_loss_across_microbatch(per_token_loss, micro_response_mask, loss_normalization, normalization_constant)
        if loss_normalization == "sequence":
            loss /= gradient_accumulation_steps
        total_loss += loss
        loss.backward()
        step += 1

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if importance_reweighting_method == "grpo": 
        mean_rewards_info.update({"clip_frac": num_clips / max(1, torch.sum(response_mask))})
    elif importance_reweighting_method == "gspo":
        mean_rewards_info.update({"clip_frac": num_clips / batch_size})

    mean_rewards_info.update({"response_len":torch.sum(response_mask) / response_mask.shape[0]})

    return total_loss,grad_norm,mean_rewards_info