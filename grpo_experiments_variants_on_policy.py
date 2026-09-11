from grpo_experiments import RLTrain
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

MODEL_PATH='./OLMo-2-0425-1B/'
PROMPTS_DIR='cs336_alignment/prompts/r1_zero.prompt'
TRAIN_DIR='./data/gsm8k/train.jsonl'
VALID_DIR='./data/gsm8k/test.jsonl'


grpo_params = {
    "GRPO_constant":("mean","std"), 
    "Dr_GRPO":("mean","none"),
    "RFT":("none","none"),
    "MaxRL":("mean","mean"),
    }

for name, param in grpo_params.items():
    baseline, advantage_normalizer = param
    RLTrain(
        MODEL_PATH,
        PROMPTS_DIR,
        TRAIN_DIR,
        VALID_DIR,
        r1_zero_reward_fn,
        device='cuda:1',
        group_size=8,
        baseline=baseline,
        advantage_normalizer=advantage_normalizer,
        rollout_n_prompts=32,
        train_n_prompts=32,
        off_policy=False,
        gradient_accumulation_steps=32,
        valid_step=10,
        loss_normalization="constant",
        normalization_constant=500,
        grpo_name=name,
    )