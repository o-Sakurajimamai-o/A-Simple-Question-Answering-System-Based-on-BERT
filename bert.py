import json
import torch
import random
import collections
import numpy as np
import pandas as pd
from torch import nn
from d2l import torch as d2l
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.autonotebook import tqdm
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import (
    BertTokenizerFast, 
    BertForQuestionAnswering, 
    get_linear_schedule_with_warmup 
)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = False

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings["input_ids"])

def compute_f1(a_gold, a_pred):
    gold_toks = list(a_gold)
    pred_toks = list(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# 将模型包装到 DataParallel 里面
# model = nn.DataParallel(model)

def process_data(data_list, tokenizer, max_length=512, stride=128):
    paragraphs = []  # context
    questions = []  
    start_positions = []  # -> answer
    end_positions = []  
    answers_text = []
    answers_start = []    
    for paragraph in data_list:
        for qa in paragraph['qas']:
            context = str(paragraph["context_text"])
            question = str(qa["query_text"])
            
            answer_text = str(qa["answers"][0])
            start_char = context.find(answer_text)

            start_position = context.find(answer_text)
            if start_position == -1:
                continue
            answer_length = len(answer_text)
            end_position = start_position + answer_length
            
            paragraphs.append(context)
            questions.append(question)
            start_positions.append(start_position)
            end_positions.append(end_position)
            answers_text.append(answer_text)
            answers_start.append(start_char)


    tokenized_examples = tokenizer(
        questions,
        paragraphs,
        truncation="only_second", # 只截断文章(context)，绝不截断问题
        max_length=max_length,
        stride=stride,            # 滑动窗口大小，比如 128
        return_overflowing_tokens=True, # 允许超长的文章溢出到下一个切片
        return_offsets_mapping=True,    # 寻找字符在切片里的位置
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i) # 区分问题(0)和文章(1)

        sample_index = sample_mapping[i]
        start_char = answers_start[sample_index]
        end_char = start_char + len(answers_text[sample_index])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    # 转换为 PyTorch Tensor
    return {k: torch.tensor(v) for k, v in tokenized_examples.items()}

"""
    print(questions[:5])
    print(start_positions[:5])
    print(end_positions[:5])
    print(paragraphs[0][start_positions[2] : end_positions[2]])
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
    print(paragraph_and_question)
    print(tokenizer.decode(paragraph_and_question[-16:]))
    print(tokenizer.decode(paragraph_and_question[-50:-16]))
    print(tokenizer.decode(paragraph_and_question[-50:]))
    print(train_encodings["token_type_ids"][0])
    print(train_encodings["token_type_ids"])
    print(train_encodings["attention_mask"][0])
    print(train_encodings["attention_mask"][10])
    print(tokenizer.decode(
        tokenizer(
            paragraphs[10],
            questions[10],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )["input_ids"][0]
    ))
    print(train_encodings["input_ids"].shape)
    print(train_encodings["token_type_ids"].shape)
    print(train_encodings["attention_mask"].shape)
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
    print(train_encodings["start_positions"])
    print(train_encodings["end_positions"])
    print(questions[2])
    print("answer 在原始字符串中的起止索引为：", start_positions[2], end_positions[2], 
        ", 长度为： " , end_positions[2] - start_positions[2] + 1)
    print(paragraphs[0][start_positions[2] : end_positions[2]])
    print(tokenizer.decode(train_encodings["input_ids"][0][86 : 110 + 1]))
"""

if __name__ == '__main__':
    save_path = "./best_bert_model"
    device = d2l.try_gpu()
    
    
    with open(r"D:\data\CMRC2018\cmrc2018_train.json") as f:
        train = json.load(f)

    with open(r"D:\data\CMRC2018\cmrc2018_dev.json") as f:
        dev = json.load(f)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    model = BertForQuestionAnswering.from_pretrained(
        "bert-base-chinese"
    ).to(device)  # type: ignore

    print("Processing Train set...")
    train_encodings = process_data(train, tokenizer, max_length=512, stride=128)
    print("Processing Dev set...")
    dev_encodings = process_data(dev, tokenizer, max_length=512, stride=128)

    train_dataset = SquadDataset(train_encodings)
    dev_dataset = SquadDataset(dev_encodings)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 15
    total_training_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_training_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optim, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_training_steps
    )

    # Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
    accelerator = Accelerator(mixed_precision="fp16")
    model, optim, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optim, train_loader, dev_loader, scheduler
    )

    model.train()

    best_exact_match = 0.0
    best_f1_score = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            optim.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"],
            )
            loss = outputs.loss
            accelerator.backward(loss)
            
            optim.step()
            scheduler.step() 

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        model.eval()
        dev_loss = 0.0
        exact_match_sum = 0
        f1_score_sum = 0.0
        total_samples = 0

        print(f"Evaluating Epoch {epoch}...")
        with torch.no_grad(): 
            for batch in dev_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"],
                )
                dev_loss += outputs.loss.item()
                
                start_pred = torch.argmax(outputs.start_logits, dim=1)
                end_pred = torch.argmax(outputs.end_logits, dim=1)
                
                for i in range(len(start_pred)):
                    s_pred, e_pred = start_pred[i].item(), end_pred[i].item()
                    s_true, e_true = batch["start_positions"][i].item(), batch["end_positions"][i].item()
                    
                    if s_pred == s_true and e_pred == e_true:
                        exact_match_sum += 1
                        
                    if s_true == 0 and s_pred == 0:
                        f1_score_sum += 1.0
                    elif s_true != 0 and s_pred <= e_pred:
                        pred_tokens = batch["input_ids"][i][s_pred:e_pred+1].tolist()
                        true_tokens = batch["input_ids"][i][s_true:e_true+1].tolist()
                        
                        pred_str = tokenizer.decode(pred_tokens, skip_special_tokens=True).replace(" ", "")
                        true_str = tokenizer.decode(true_tokens, skip_special_tokens=True).replace(" ", "")
                        
                        f1_score_sum += compute_f1(true_str, pred_str)
                    
                    total_samples += 1

        epoch_em = exact_match_sum / total_samples
        epoch_f1 = f1_score_sum / total_samples
        print(f"Epoch {epoch} Summary: Train Loss: {train_loss/len(train_loader):.4f} | Dev Loss: {dev_loss/len(dev_loader):.4f}")
        print(f"Metrics: Dev EM: {epoch_em:.4f} | Dev F1: {epoch_f1:.4f}")

        if epoch_f1 > best_f1_score:
            best_f1_score = epoch_f1
            print(f"best Dev F1: {best_f1_score:.4f}，saving model...")
            
            model_to_save = accelerator.unwrap_model(model)
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    print(f"\nDone，best Dev F1 为: {best_f1_score:.4f}")