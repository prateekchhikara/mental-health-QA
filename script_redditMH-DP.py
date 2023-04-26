import pandas as pd
import numpy as np
from transformers import T5Tokenizer
from transformers import T5Config, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import glob, json
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="4"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print(device)

from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_qa_with_options(input_str):
  qa_string = f"Based on following post by a social media user, are they at risk of suffering from any serious mental illness? \n"
  qa_string += input_str + " \\n (a) yes (b) no"

  return qa_string

def target(input_val):
  if input_val == 0:
    return "no"
  return "yes"

print(generate_qa_with_options("I am very sad"))

import pickle

# Load data from pickle file
with open("datasets-MH/RedditMH-prateek/redditMH_X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

# Load data from pickle file
with open("datasets-MH/RedditMH-prateek/redditMH_X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

# Load data from pickle file
with open("datasets-MH/RedditMH-prateek/redditMH_y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

# Load data from pickle file
with open("datasets-MH/RedditMH-prateek/redditMH_y_test.pkl", "rb") as f:
    y_test = pickle.load(f)


import random
random.seed(42)

num_elements = int(0.05 * len(X_train))
indices = random.sample(range(len(X_train)), num_elements)

X_train = [X_train[i] for i in indices]
y_train = [y_train[i] for i in indices]


print(len(X_train), len(X_test), len(y_train), len(y_test), type(X_train))


df_train = pd.DataFrame({"question": X_train, "answer": y_train})
df_train["question"] = df_train["question"].apply(generate_qa_with_options)
df_train["answer"] = df_train["answer"].apply(target)

import torch
from torch.utils.data import Dataset, DataLoader

class SummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame = df_train,
        tokenizer: T5Tokenizer = tokenizer,
        text_max_token_len: int = 200,
        summary_max_token_len: int = 15
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row['question']

        text_encoding = tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        summary_encoding = tokenizer(
            data_row['answer'],
            max_length=self.summary_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = summary_encoding['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100

        return dict(
            input_ids=text_encoding['input_ids'].flatten(),
            attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            decoder_attention_mask=summary_encoding['attention_mask'].flatten()
        )

batch_size=128

dataset = SummaryDataset(df_train, tokenizer)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True)


num_epochs = 20
num_training_steps = num_epochs * len(dataloader)
progress_bar = tqdm(range(num_training_steps))

from torch.optim.lr_scheduler import StepLR

optimizer = AdamW(model.parameters(), lr=0.001)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


state_dict = torch.load("redditMH-UnifiedQAsmall-bestModel.pt")
model.load_state_dict(state_dict)
model.to(device)


for param in model.encoder.parameters():
    param.requires_grad = False

for param in model.decoder.parameters():
    param.requires_grad = False
    
    
model = model.to(device)
model.train()


minLoss = 100000000

with open("redditMH-logs-small-DP.txt", "w+") as f:
    for epoch in range(num_epochs):
        for p in model.parameters():
            if p.requires_grad:
                p.accumulate_grads = []

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            for e in range(batch_size):    
                input_ids = batch['input_ids'][e].unsqueeze(0) 
                attention_mask = batch['attention_mask'][e].unsqueeze(0) 
                labels = batch['labels'][e].unsqueeze(0) 
                decoder_attention_mask = batch['decoder_attention_mask'][e].unsqueeze(0) 
                
                one_unit = {
                    'input_ids' : input_ids,
                    'attention_mask' : attention_mask,
                    'labels' : labels,
                    'decoder_attention_mask' : decoder_attention_mask
                }
                            

                outputs = model(**batch)
                logits = outputs.logits
            
                # loss = outputs.loss
                loss = outputs.loss.clone().detach().requires_grad_(True)
                loss.backward()
            
                for p in model.parameters():
                    if p.requires_grad:
                        per_sample_grad = p.grad.detach().clone()
                        torch.nn.utils.clip_grad_norm_(per_sample_grad, max_norm=1)
                        p.accumulate_grads.append(per_sample_grad)
                    
            for p in model.parameters():
                if p.requires_grad:
                    p.grad = torch.stack(p.accumulate_grads).mean(dim=0).to(device)
                    p.grad += torch.normal(0.0, 0.1*1, p.grad.shape).to(device)
                
            
            optimizer.step()
            lr_scheduler.step()
            
            optimizer.zero_grad()
            progress_bar.update()
        

        if loss < minLoss:
            print("model saved with loss = ", loss)
            torch.save(model.state_dict(), f"redditMH-UnifiedQA-small-DP-bestModel.pt")
            minLoss = loss
        f.write(f'epoch: {epoch + 1} -- loss: {loss}\n')
        print(f'epoch: {epoch + 1} -- loss: {loss}')