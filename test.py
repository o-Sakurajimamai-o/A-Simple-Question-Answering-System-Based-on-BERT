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

device = d2l.try_gpu()
model = BertForQuestionAnswering.from_pretrained("./Bert_QA").to(device) # type: ignore
tokenizer = BertTokenizerFast.from_pretrained("./Bert_QA")

with open(r"D:\data\CMRC2018\cmrc2018_dev.json") as f:
    dev = json.load(f)


def predict(doc, query, first=False):
    if first:
        print("sequence: ", doc)
    print("question: ", query)
    
    item = tokenizer(
        [doc, query],
        max_length = 512,
        return_tensors = "pt",
        truncation = True,
        padding = True
    )

    with torch.no_grad():
        input_ids = item["input_ids"].to(device).reshape(1, -1)
        attention_mask = item["attention_mask"].to(device).reshape(1, -1)

        output = model(input_ids[:, :512], attention_mask[:, :512])

        start_pred = torch.argmax(output.start_logits, dim=1)
        end_pred = torch.argmax(output.end_logits, dim=1)

    
    try:
        start_pred = item.token_to_chars(0, start_pred)
        end_pred = item.token_to_chars(0, end_pred)
    except:
        return "No Answer ~"
    
    if start_pred.start > end_pred.end:
        return "No Answer ~"
    else:
        return "Answer：" + doc[start_pred.start : end_pred.end]

print(model)

model.eval()

# print(str(dev[100]["context_text"]))
# print(str(dev[100]["qas"][0]["query_text"]))

# print(predict(
#     str(dev[100]["context_text"]), 
#     str(dev[100]["qas"][0]["query_text"])
# ))

doc = "晗晗，学习的是工程造价专业，她一生要强，身高167cm，" \
"体重 110 斤，她现在喜欢玩王者荣耀，主打辅助和中单，辅助的话喜欢玩蔡文姬、墨子、刘禅，" \
"中单的话喜欢玩小乔、妲己。"
question1 = "晗晗学什么专业？"
question2 = "晗晗有多高？"
question3 = "晗晗有多重？"
question4 = "晗晗喜欢玩什么游戏？"
question5 = "晗晗主打王者的哪些分路？"
question6 = "中路和辅助分别喜欢玩什么？"
print(predict(doc, question1, True))
print(predict(doc, question2))
print(predict(doc, question3))
print(predict(doc, question4))
print(predict(doc, question5))
print(predict(doc, question6))
