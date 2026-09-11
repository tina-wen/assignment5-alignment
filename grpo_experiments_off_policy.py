from grpo_experiments import RLTrain
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

MODEL_PATH='./OLMo-2-0425-1B/'
PROMPTS_DIR='cs336_alignment/prompts/r1_zero.prompt'
TRAIN_DIR='./data/gsm8k/train.jsonl'
VALID_DIR='./data/gsm8k/test.jsonl'

for importance_reweighting_method in ["none", "noclip", "grpo", "gspo"]:
    if importance_reweighting_method == "grpo":
        cliprange = 0.2
    elif importance_reweighting_method == "gspo":
        cliprange = 3e-4
    else:
        cliprange = None

    RLTrain(
        MODEL_PATH,
        PROMPTS_DIR,
        TRAIN_DIR,
        VALID_DIR,
        r1_zero_reward_fn,
        device = 'cuda:1',
        group_size = 8,
        rollout_n_prompts = 32,
        train_n_prompts=1,
        off_policy=True,
        gradient_accumulation_steps = 1,
        valid_step = 100,
        importance_reweighting_method=importance_reweighting_method,
        cliprange=cliprange, # 3e-4 for gspo
    )