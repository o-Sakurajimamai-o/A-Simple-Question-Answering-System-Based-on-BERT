import os
import json
import torch
import random
import logging
import collections
import numpy as np
import pandas as pd
from torch import nn
from d2l import torch as d2l
from datetime import datetime
from torch.optim import AdamW
import matplotlib.pyplot as plt
from accelerate import Accelerator
from tqdm.autonotebook import tqdm
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import (
    T5Tokenizer, 
    BertTokenizerFast,
    BertForQuestionAnswering,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)


def setup_logger(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"bert_training_{timestamp}.log")

    logger = logging.getLogger("bert_qa")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_file


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

    # 将溢出的切片映射回原始样本的索引
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    # 为每一个切片计算答案的 start 和 end
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i) # 区分问题(0)和文章(1)

        # 找到该切片对应的原始样本
        sample_index = sample_mapping[i]
        start_char = answers_start[sample_index]
        end_char = start_char + len(answers_text[sample_index])

        # 获取该切片中，文章(context)部分的起始和结束 token 索引
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # 判断答案是否完整包含在这个滑动窗口(切片)内
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            # 答案不在这里面，把预测目标设为 [CLS]（代表此片段无答案）
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # 答案在这个片段里，精确计算它的起止 token 索引
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


if __name__ == '__main__':
    # ==================== 超参数配置 ====================
    SEED          = 42
    LEARNING_RATE = 3e-5    # 原 5e-5，3e-5 对 BERT 微调更稳定
    NUM_EPOCHS    = 5       # 原 15，BERT 通常 3-5 轮即可收敛
    BATCH_SIZE    = 16      # 原 32，更小 batch 梯度更稳定
    WEIGHT_DECAY  = 0.01    # 新增 L2 正则，防止过拟合
    MAX_GRAD_NORM = 1.0     # 新增梯度裁剪上限
    MAX_LENGTH    = 512
    STRIDE        = 128
    WARMUP_RATIO  = 0.1
    SAVE_PATH     = "./best_bert_model"
    # ====================================================

    same_seeds(SEED)
    logger, log_file = setup_logger("./logs")
    logger.info(f"日志文件: {log_file}")

    device = d2l.try_gpu()
    logger.info(f"使用设备: {device}")

    logger.info("=" * 60)
    logger.info("训练超参数:")
    logger.info(f"  Seed:           {SEED}")
    logger.info(f"  Learning Rate:  {LEARNING_RATE}")
    logger.info(f"  Epochs:         {NUM_EPOCHS}")
    logger.info(f"  Batch Size:     {BATCH_SIZE}")
    logger.info(f"  Weight Decay:   {WEIGHT_DECAY}")
    logger.info(f"  Max Grad Norm:  {MAX_GRAD_NORM}")
    logger.info(f"  Max Length:     {MAX_LENGTH}")
    logger.info(f"  Stride:         {STRIDE}")
    logger.info(f"  Warmup Ratio:   {WARMUP_RATIO}")
    logger.info("=" * 60)

    with open(r"D:\data\CMRC2018\cmrc2018_train.json") as f:
        train = json.load(f)

    with open(r"D:\data\CMRC2018\cmrc2018_dev.json") as f:
        dev = json.load(f)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    model = BertForQuestionAnswering.from_pretrained(
        "bert-base-chinese"
    ).to(device)  # type: ignore

    logger.info("Processing Train set...")
    train_encodings = process_data(train, tokenizer, max_length=MAX_LENGTH, stride=STRIDE)
    logger.info("Processing Dev set...")
    dev_encodings = process_data(dev, tokenizer, max_length=MAX_LENGTH, stride=STRIDE)

    train_dataset = SquadDataset(train_encodings)
    dev_dataset = SquadDataset(dev_encodings)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(dev_dataset)}")
    logger.info(f"训练 Batch 数: {len(train_loader)} | 验证 Batch 数: {len(dev_loader)}")

    # 对 bias 和 LayerNorm 不施加 weight decay（BERT 微调最佳实践）
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)

    total_training_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    logger.info(f"总训练步数: {total_training_steps} | Warmup 步数: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    accelerator = Accelerator(mixed_precision="fp16")
    model, optim, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optim, train_loader, dev_loader, scheduler
    )

    best_exact_match = 0.0
    best_f1_score = 0.0

    # ==================== 存储绘图数据 ====================
    history_train_loss = []
    history_dev_loss = []
    history_em = []
    history_f1 = []
    # ==========================================================

    for epoch in range(NUM_EPOCHS):
        # -------------------- 训练阶段 --------------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

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
            # 梯度裁剪，防止梯度爆炸
            accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optim.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_train_loss = train_loss / len(train_loader)

        # -------------------- 验证阶段 --------------------
        model.eval()
        dev_loss = 0.0
        exact_match_sum = 0
        f1_score_sum = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Eval] "):
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
        avg_dev_loss = dev_loss / len(dev_loader)

        # -------------------- Epoch 汇总日志 --------------------
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary")
        logger.info(f"  Train Loss : {avg_train_loss:.4f}")
        logger.info(f"  Dev   Loss : {avg_dev_loss:.4f}")
        logger.info(f"  Dev   EM   : {epoch_em:.4f}  ({exact_match_sum}/{total_samples})")
        logger.info(f"  Dev   F1   : {epoch_f1:.4f}")
        logger.info("=" * 60)

        # -------------------- 保存最优模型 --------------------
        if epoch_f1 > best_f1_score:
            best_f1_score = epoch_f1
            best_exact_match = epoch_em
            logger.info(f">>> 新的最佳记录! Dev F1: {best_f1_score:.4f} | EM: {best_exact_match:.4f}，正在保存模型...")
            model_to_save = accelerator.unwrap_model(model)
            model_to_save.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            logger.info(f">>> 模型已保存至 {SAVE_PATH}")
        
        # ==================== 记录当前 Epoch 数据 ====================
        history_train_loss.append(avg_train_loss)
        history_dev_loss.append(avg_dev_loss)
        history_em.append(epoch_em)
        history_f1.append(epoch_f1)
        # =================================================================

    # ==================== 新增：绘制并保存曲线图 ====================
    logger.info("正在生成训练曲线图...")
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1) 
    plt.plot(epochs_range, history_train_loss, 'b-o', label='Train Loss')
    plt.plot(epochs_range, history_dev_loss, 'r-s', label='Dev Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2) 
    plt.plot(epochs_range, history_em, 'g-^', label='Dev EM')
    plt.plot(epochs_range, history_f1, 'm-d', label='Dev F1')
    plt.title('Validation EM and F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join("./logs", f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"曲线图已保存至 : {plot_path}")
    # ==============================================================

    logger.info("=" * 60)
    logger.info("训练结束！")
    logger.info(f"最优模型保存路径 : {SAVE_PATH}")
    logger.info(f"历史最高 Dev F1  : {best_f1_score:.4f}")
    logger.info(f"历史最高 Dev EM  : {best_exact_match:.4f}")
    logger.info("=" * 60)

    