import json
import random
from supplement import compute_dpo_loss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from pathlib import Path
from function import tokenize_prompt_and_output, get_response_log_probs

# 初始化参数值
beta = 0.1
ref_device = 'cuda:0'
device = 'cuda:1'
batch_size = 64
lr = 1e-06

def _valid(
    lm,
    tokenizer,
    dataset: list[dict[str, str]],
    ) -> float:
    device = next(lm.parameters()).device

    template = Path('./cs336_alignment/prompts_safety/alpaca_sft.prompt').read_text(encoding = 'utf-8').strip()
    acc = 0
    # 成对responses(chosen/rejected)计算准确率
    for data in dataset:
        prompt, response_chosen, response_rejected = data['prompt'], data['response_chosen'], data['response_rejected']
         # 获得prompt
        prefix = template.format(
            instruction = prompt,
            response = '',
        )
        inputs = tokenize_prompt_and_output(
            [prefix]*2,
            [response_chosen + tokenizer.eos_token, response_rejected + tokenizer.eos_token],
            tokenizer
            )
        input_ids, labels, mask = inputs["input_ids"], inputs["labels"], inputs["response_mask"]
        with torch.no_grad():
            log_probs = get_response_log_probs(lm, input_ids.to(device), labels.to(device))['log_probs'] # (B,S)
        log_prob_pair = (log_probs * mask.to(device)).sum(dim = -1) # (B,)
        acc += int(log_prob_pair[0] > log_prob_pair[1])

    return acc/len(dataset)
    

# DPO训练循环
def DPOTrain(
        model_path: str,
        device: str,
        ref_device: str,
        train_set: list[dict[str, str]],
        valid_set: list[dict[str, str]],
        beta: float,
        lr: float,
        batch_size: int,
        ckpt_path: str,
    ):
    # 加载参考模型lm_ref和训练模型lm
    lm = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    lm_ref = AutoModelForCausalLM.from_pretrained(model_path).to(ref_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    optimizer = torch.optim.RMSprop(lm.parameters(), lr = lr)

    run = wandb.init(
        project=f"cs336-ass5-dpo", 
        # name=wandb_name,
        # group=f"{policy_name}_{importance_reweighting_method}",
        )

    lm.train()
    lm_ref.eval()

    num_samples, total_loss = 0, 0.0
    highest_accuracy = -1.0
    for data in train_set:
        prompt, response_chosen, response_rejected = data['prompt'], data['response_chosen'], data['response_rejected']
        loss = compute_dpo_loss(
            lm,
            lm_ref,
            tokenizer,
            beta,
            prompt,
            response_chosen,
            response_rejected,
        )
        loss /= batch_size
        total_loss += loss.detach().item()
        loss.backward()

        num_samples += 1
        if num_samples % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

            lm.eval()
            acc = _valid(lm, tokenizer, valid_set)
            if acc > highest_accuracy:
                lm.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                meta_data = {
                    'step': num_samples // batch_size, 
                    'accuracy': acc, 
                    'beta': beta, 
                    'learning_rate': lr
                    }
                with open(ckpt_path+'metadata.json','w',encoding='utf-8') as f:
                    json.dump(meta_data,f,ensure_ascii=False,indent=2)
                highest_accuracy = acc

            wandb.log({
                'loss': total_loss,
                'acc': acc,
            })
            total_loss = 0.0

            lm.train()

    # if num_samples % batch_size:
    #     optimizer.step()
    #     optimizer.zero_grad()
        
    #     acc = _valid(lm, tokenizer, valid_set)
    #     wandb.log({
    #         'loss': total_loss,
    #         'acc': acc,
    #     })
        
if __name__ == '__main__':
    # 读取数据
    with open('processed_hh_data.json','r',encoding='utf-8') as f:
        datasets = json.load(f)

    ckpt_path = './dpo_ckpts/'

    # 分出验证集；落盘训练/验证集
    valid_idx = random.sample(range(len(datasets)),200)
    valid_set = [datasets[idx] for idx in valid_idx]
    train_set = [datasets[idx] for idx in range(len(datasets)) if idx not in valid_idx]

    DPOTrain(model_path, device, ref_device, train_set, valid_set, beta, lr, batch_size, ckpt_path)




