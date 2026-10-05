import os
import pandas as pd
import json
from pathlib import Path
import gzip
from typing import Any

DATA_PATH = 'data/'
PROMPT_PATH = 'cs336_alignment/'

def load_dataset(
        data_dir: str | os.PathLike[str], 
        add_src: str | None = None, 
        ) -> list[dict[str, Any]]:

    if not isinstance(data_dir, str):
        data_dir = os.fspath(data_dir)
    
    if data_dir.endswith('.json'):
        with open(data_dir,'r',encoding='utf-8') as f:
            data = json.load(f)
    elif data_dir.endswith('.jsonl'):
        with open(data_dir, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    elif data_dir.endswith('.jsonl.gz'):
        with gzip.open(data_dir,"r") as f:
            data = [json.loads(line) for line in f]
    # 如果data_dir是一个路径
    else:
        data = []
        for file in os.listdir(data_dir):
            data += load_dataset(os.path.join(data_dir,file), add_src=file)
        return data
    if add_src is not None:
        for d in data:
            d['src'] = add_src
    return data

def load_simple_safety_tests(data_dir):
    data_df = pd.read_csv(data_dir)
    data = []
    for _, row in data_df.iterrows():
        data.append({
            "instruction": row["prompts_final"],
        })
    return data

def load_mmlu(data_dir: str):
    mmlu_examples = []
    for name in os.listdir(data_dir):
        data = pd.read_csv(os.path.join(data_dir,name),header=None,dtype=str,keep_default_na=False)
        for row in data.itertuples():
            mmlu_examples.append({
                "subject":name.split('_val')[0],
                "question":row[1],
                "options":list(row[2:-1]),
                "answer":row[-1],
                })
    return mmlu_examples

def get_prompts(data: list[dict[str,Any]], prompt_dir: str) -> list[str]:
    template = Path(prompt_dir).read_text(encoding='utf-8').strip()
    prompts = [template.format(**d) for d in data]
    return prompts


alpaca_sft_prompt_dir = os.path.join(PROMPT_PATH, 'prompts_safety', 'alpaca_sft' + '.prompt')
alpaca_sft_template = Path(alpaca_sft_prompt_dir).read_text(encoding='utf-8').strip()

def get_alpaca_sft_prefix(prompt: str) -> str:
    prefix = alpaca_sft_template.format(
        instruction = prompt,
        response = '',
    ) 
    return prefix