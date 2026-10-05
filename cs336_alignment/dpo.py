import torch
from function import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.data_loading import get_alpaca_sft_prefix

def tokenize_preference_pair(
        prompt: str,
        response_chosen: str,
        response_rejected: str,
        tokenizer,
):
    prefix = get_alpaca_sft_prefix(prompt)
    # [chosen, rejected]
    inputs = tokenize_prompt_and_output(
        [prefix]*2,
        [response_chosen + tokenizer.eos_token, response_rejected + tokenizer.eos_token],
        tokenizer
        )
    input_ids, labels, mask = inputs["input_ids"], inputs["labels"], inputs["response_mask"] 
    return input_ids, labels, mask

def compute_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    
    device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    input_ids, labels, mask = tokenize_preference_pair(
        prompt,
        response_chosen,
        response_rejected,
        tokenizer,
    )

    log_prob_pair = get_response_log_probs(
        lm, 
        input_ids.to(device), 
        labels.to(device),
        )['log_probs']
    
    with torch.no_grad():
        ref_log_prob_pair = get_response_log_probs(
            lm_ref, 
            input_ids.to(ref_device), 
            labels.to(ref_device),
            )['log_probs']

    log_probs = (log_prob_pair * mask.to(device)).sum(dim = -1) # (2,)
    ref_log_probs = (ref_log_prob_pair * mask.to(ref_device)).sum(dim = -1).to(device) 
    
    diff = beta * torch.matmul(
        torch.tensor([1.0,-1.0], dtype=log_probs.dtype, device = device), 
        (log_probs - ref_log_probs))

    loss = -torch.log(torch.sigmoid(diff))
    return loss


