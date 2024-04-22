import torch
import sys
import os
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, BertTokenizer, BertForMaskedLM, AdamW
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import pandas as pd
import random
import numpy as np

from tqdm import tqdm
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sentence dataset struct
class SentenceDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len, lan_codes=None, split=None):
        self.tokenizer = tokenizer
        self.sentences = pd.read_csv(file_path)
        train_ratio = 0.999

        if lan_codes is not None:
            # filter languages
            self.sentences = self.sentences[self.sentences['lan_code'].isin(lan_codes)]
        if split is not None:
            length = len(self.sentences)
            if split == 'train':
                self.sentences = self.sentences.iloc[:int(length * train_ratio)]
            elif split == 'test':
                self.sentences = self.sentences.iloc[int(length * train_ratio):]
            else:
                raise ValueError(f"split {split} not recognized")
        self.max_len = max_len
        
        print(f"Done loading dataset for split {split}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences.iloc[item]['sentence']).lower()
        encoding = self.tokenizer(sentence, max_length=self.max_len, padding='max_length', truncation=True)
        return {key: torch.tensor(val) for key, val in encoding.items()}


def convert_weights_to_fp16(model: nn.Module, dtype=torch.bfloat16):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

    model.apply(_convert_weights_to_fp16)

def reinitialize_weights(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



def train_step(model, batch, optimizer, step_optim):
    inputs = {k: v.to(device) for k, v in batch.items()}
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    if step_optim:
        optimizer.step()
        optimizer.zero_grad()
    return loss

@torch.no_grad()
def eval_step(model, test_dataloader):
    # step has different meanings here lol but whatever
    total_loss = 0
    for batch in tqdm(test_dataloader, desc='Eval', total=len(test_dataloader)):
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(**inputs)
        total_loss += outputs.loss
    return total_loss / len(test_dataloader)

def train(model, optimizer, train_dataloader, test_dataloader, lan_code, args):
    # train
    log_freq = args['log_freq']
    eval_freq = args['eval_freq']
    ckpt_freq = args['ckpt_freq']
    epochs = args['epochs']
    grad_accums = args['grad_accums']
    
    step = 0
    for epoch in (range(epochs)):  # Number of training epochs
        model.train()
        train_loss = 0.0
        for i, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}", total=len(train_dataloader)):
            if (i + 1) % grad_accums != 0:
                train_loss += train_step(model, batch, optimizer, step_optim=False)
                continue
       
            train_loss += train_step(model, batch, optimizer, step_optim=True)
            train_loss /= grad_accums
            step += 1                

            if (step) % eval_freq == 0:
                test_loss = eval_step(model, test_dataloader)
                print("Eval Loss:" ,test_loss)
                wandb.log({"test.loss": test_loss})
    
            if (step) % log_freq == 0:
                print(f"Loss: {train_loss}")
                wandb.log({"train.loss": train_loss})

            if (step) % ckpt_freq == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(f"checkpoints/{lan_code}_{step}.pt", model)

            train_loss = 0.

# training arguments

def main():
    # get arguments
    lan_code = sys.argv[1]
    args = {
        'batch_size': 60,
        'lr': 2e-5,
        'log_freq': 1,
        'eval_freq': 100,
        'ckpt_freq': 100,
        'epochs': 3,
        'mlm_prob': 0.4,
        'grad_accums': 20
    }
    
    # construct dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = SentenceDataset(tokenizer,
                                    'data/big-language-detection/sentences.csv',
                                    max_len=512,
                                    lan_codes=[lan_code],
                                    split='train')
    test_dataset = SentenceDataset(tokenizer,
                                   'data/big-language-detection/sentences.csv',
                                   max_len=512,
                                   lan_codes=[lan_code],
                                   split='test')

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args['mlm_prob'])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args['batch_size'],
                                  shuffle=True,
                                  collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args['batch_size'] * 3,
                                 shuffle=True,
                                 collate_fn=data_collator)

    model = AutoModelForCausalLM.from_pretrained('bert-base-uncased')
    reinitialize_weights(model)    # randomize weights so we start from scratch
    convert_weights_to_fp16(model)
    optimizer = AdamW(model.parameters(), lr=args['lr'])
    model.to(device)
    print(f"training with {sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1e6:.2f}M parameters")

    wandb.init(project='ling')

    # start training
    train(model, optimizer, train_dataloader, test_dataloader, lan_code, args)

if __name__ == '__main__':
    main()
