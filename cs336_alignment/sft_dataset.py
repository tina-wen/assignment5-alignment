
import random
import torch
from cs336_alignment.data_loading import alpaca_sft_template, load_dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_path: str, seq_length: int, shuffle: bool):

        data = load_dataset(dataset_path)
        text_list = [alpaca_sft_template.format(
            instruction = d['prompt'],
            response = d['response'],
        ) for d in data ] # list[str]

        if shuffle:
            random.shuffle(text_list)

        tokens = []
        for text in text_list:
            tokens.extend(tokenizer.encode(text))
            tokens.append(tokenizer.eos_token_id)

        inputs = tokens[:-1]
        labels = tokens[1:]
        num_chunks = len(inputs) // seq_length
        self.n = num_chunks

        inputs = [inputs[i*seq_length:(i+1)*seq_length] for i in range(num_chunks)]
        labels = [labels[i*seq_length:(i+1)*seq_length] for i in range(num_chunks)]

        self.inputs = torch.tensor(inputs, dtype=torch.int64)
        self.labels = torch.tensor(labels, dtype=torch.int64)


    def __len__(self):
        return self.n

    def __getitem__(self,i):
        return {"input_ids": self.inputs[i,:], "labels": self.labels[i,:]}