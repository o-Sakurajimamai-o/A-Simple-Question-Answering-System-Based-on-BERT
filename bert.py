import json
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from d2l import torch as d2l
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.autonotebook import tqdm
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertForQuestionAnswering


save_path = "."
device = d2l.try_gpu()

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)


with open(r"D:\data\CMRC2018\cmrc2018_train.json") as f:
    train = json.load(f)

with open(r"D:\data\CMRC2018\cmrc2018_dev.json") as f:
    dev = json.load(f)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
model = BertForQuestionAnswering.from_pretrained(
    "bert-base-chinese"
).to(device)  # type: ignore


# 将模型包装到 DataParallel 里面
# model = nn.DataParallel(model)

paragraphs = []  # context
questions = []  
start_positions = []  # -> answer
end_positions = []  
for paragraph in train:
    for qa in paragraph['qas']:
        context = str(paragraph["context_text"])
        question = str(qa["query_text"])
        
        answer_text = str(qa["answers"][0])
        
        start_position = context.find(answer_text)
        if start_position == -1:
            continue
        answer_length = len(answer_text)
        end_position = start_position + answer_length
        
        paragraphs.append(context)
        questions.append(question)
        start_positions.append(start_position)
        end_positions.append(end_position)

# print(questions[:5])
# print(start_positions[:5])
# print(end_positions[:5])
# print(paragraphs[0][start_positions[2] : end_positions[2]])

train_encodings = tokenizer(
    paragraphs,
    questions,
    return_tensors = "pt", # 返回 tensor
    padding = True,
    truncation=True,
    max_length = 512,
)

# print(train_encodings.keys())
paragraph_and_question = tokenizer(
    paragraphs[0],
    questions[0],
    return_tensors = "pt",
    padding = True,
    truncation = True,
    max_length = 512,
)["input_ids"][0]

# print(paragraph_and_question)
# print(tokenizer.decode(paragraph_and_question[-16:]))
# print(tokenizer.decode(paragraph_and_question[-50:-16]))
# print(tokenizer.decode(paragraph_and_question[-50:]))
# print(train_encodings["token_type_ids"][0])
# print(train_encodings["token_type_ids"])
# print(train_encodings["attention_mask"][0])
# print(train_encodings["attention_mask"][10])
# print(tokenizer.decode(
#     tokenizer(
#         paragraphs[10],
#         questions[10],
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512,
#     )["input_ids"][0]
# ))
# print(train_encodings["input_ids"].shape)
# print(train_encodings["token_type_ids"].shape)
# print(train_encodings["attention_mask"].shape)

train_encodings["start_positions"] = torch.tensor(
    [
        train_encodings.char_to_token(idx, x)
        if train_encodings.char_to_token(idx, x) != None
        else -1
        for idx, x in enumerate(start_positions)
    ]
)

train_encodings["end_positions"] = torch.tensor(
    [
        train_encodings.char_to_token(idx, x - 1)
        if train_encodings.char_to_token(idx, x - 1) != None
        else -1
        for idx, x in enumerate(end_positions)
    ]
)

# print(train_encodings["start_positions"])
# print(train_encodings["end_positions"])
# print(questions[2])
# print("answer 在原始字符串中的起止索引为：", start_positions[2], end_positions[2], 
#       ", 长度为： " , end_positions[2] - start_positions[2] + 1)
# print(paragraphs[0][start_positions[2] : end_positions[2]])
# print(tokenizer.decode(train_encodings["input_ids"][0][86 : 110 + 1]))

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)
    
train_dataset = SquadDataset(train_encodings)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

### START CODE HERE ###
# Use AdamW as the optimizer, and learning rate 5e-5.
# https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
### END CODE HERE ###


# Change "fp16_training" to True to support automatic mixed precision training (fp16)
fp16_training = True

if fp16_training:
    accelerator = Accelerator()
    device = accelerator.device

# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
model, optim, train_loader = accelerator.prepare(model, optim, train_loader)

model.train()
num_epochs = 25
loss_sum = 0.0
acc_start_sum = 0.0
acc_end_sum = 0.0

for epoch in range(num_epochs):
    loss_sum = 0.0
    acc_start_sum = 0.0
    acc_end_sum = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        optim.zero_grad()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]

        outputs = model(
            input_ids,
            attention_mask = attention_mask,
            start_positions = start_positions,
            end_positions = end_positions,
        )
        loss = outputs.loss.mean()
        if fp16_training:
            accelerator.backward(loss)
        else:
            loss.backward()
        optim.step()

        loss_sum += loss.item()

        start_pred = torch.argmax(outputs.start_logits, dim=1)
        end_pred = torch.argmax(outputs.end_logits, dim=1)

        acc_start = (start_pred == start_positions).float().mean()
        acc_end = (end_pred == end_positions).float().mean()

        acc_start_sum += acc_start
        acc_end_sum += acc_end

        postfix = {
            "loss": f"{loss_sum/(batch_idx+1):.4f}",
            "acc_start": f"{acc_start_sum/(batch_idx+1):.4f}",
            "acc_end": f"{acc_end_sum/(batch_idx+1):.4f}",
        }

        batch_desc = f"Epoch {epoch}, train loss: {postfix['loss']}"
        pbar.set_postfix_str(
            f"{batch_desc}, acc start: {postfix['acc_start']}, acc end: {postfix['acc_end']}"
        )

    if epoch % 5 == 0:
        print("Saving model and tokenizer...")
        try:
            model_to_save = accelerator.unwrap_model(model)
        except Exception:
            model_to_save = getattr(model, "module", model)  # DataParallel -> .module
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Successfully saved to {save_path}")


print("Saving model and tokenizer...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Successfully saved to {save_path}")