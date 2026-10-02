import os
import gzip
import json
PATH="./data/hh/"

datasets = []
for file in os.listdir(PATH):
    with gzip.open(PATH+file,"r") as f:
        for line in f:
            d = json.loads(line)
            raw_prompt_chosen = d['chosen'].split('\n\nHuman:')
            raw_prompt_rejected = d['rejected'].split('\n\nHuman:')
            # 确保只有一个Human: 
            if len(raw_prompt_chosen) != 2 or len(raw_prompt_rejected) != 2:
                continue
            data_chosen = raw_prompt_chosen[1].split('\n\nAssistant:')
            data_rejected = raw_prompt_rejected[1].split('\n\nAssistant:')
            # 确保只有一个Assistant: 
            if len(data_chosen) != 2 or len(data_rejected) != 2:
                continue
            prompt_chosen, response_chosen = data_chosen
            prompt_rejected, response_rejected = data_rejected
            # 判断chosen和rejected解出来的prompt一致
            if prompt_chosen.strip() != prompt_rejected.strip():
                continue
            datasets.append(
                {
                    'prompt': prompt_rejected.strip(),
                    'response_chosen': response_chosen.strip(),
                    'response_rejected': response_rejected.strip(),
                    'src': file,
                })

with open('processed_hh_data.json','w',encoding='utf-8') as f:
    json.dump(datasets, f, ensure_ascii=False, indent=2)
            