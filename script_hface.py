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
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

print(device)

batch_size=128

from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_qa_with_options(input_str):
  qa_string = f"Based on the following post by a social media user, which of the following mental illnesses could they be most likely suffering from? \n"
  qa_string += str(input_str) + " \\n (a) depression (b) ocd (c) aspergers (d) ptsd (e) adhd"

  return qa_string

def target(input_val):
  if input_val == "0" or input_val == 0:
    return "depression"
  if input_val == "1" or input_val == 1:
    return "ocd"
  if input_val == "2" or input_val == 2:
    return "aspergers"
  if input_val == "3" or input_val == 3:
    return "ptsd"
  if input_val == "4" or input_val == 4:
    return "adhd"
      
  return None


print(generate_qa_with_options("I am very sad"))

import pickle

# Load data from pickle file
with open("datasets-MH/Huggingface-John/huggingface_reddit_X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

# Load data from pickle file
with open("datasets-MH/Huggingface-John/huggingface_reddit_X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

# Load data from pickle file
with open("datasets-MH/Huggingface-John/huggingface_reddit_y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

# Load data from pickle file
with open("datasets-MH/Huggingface-John/huggingface_reddit_y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print(len(X_train), len(X_test), len(y_train), len(y_test), type(X_train))


df_train = pd.DataFrame({"question": X_train, "answer": y_train})
df_train["question"] = df_train["question"].apply(generate_qa_with_options)
df_train["answer"] = df_train["answer"].apply(target)

print(df_train.head())
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

dataset = SummaryDataset(df_train, tokenizer)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)


num_epochs = 50
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

model = model.to(device)
model.train()


minLoss = 100000000

with open("hface-logs-small-50epochs.txt", "w+") as f:
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            optimizer.zero_grad()
            progress_bar.update()
        

        if loss < minLoss:
            print("model saved with loss = ", loss)
            torch.save(model.state_dict(), f"hface-UnifiedQA-small-50epochs-bestModel.pt")
            minLoss = loss
        f.write(f'epoch: {epoch + 1} -- loss: {loss}\n')
        print(f'epoch: {epoch + 1} -- loss: {loss}')