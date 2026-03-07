import os
import json
import torch
import random
import logging
import numpy as np
from d2l import torch as d2l
from datetime import datetime
import matplotlib.pyplot as plt
from accelerate import Accelerator
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)

def setup_logger(log_dir="./logs_t5"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"t5_csc_training_{timestamp}.log")

    logger = logging.getLogger("t5_csc")
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

class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

def ProcessData(data_path, tokenizer, max_length = 128):
    inputs = []
    outputs = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) == 2:
                wrong_text = parts[0]
                correct_text = parts[1]

                inputs.append(f"纠错：{wrong_text}")
                outputs.append(correct_text)

    embeddings = tokenizer(
        inputs,
        max_length = max_length,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    )            
    
    labels = tokenizer(
        outputs,
        max_length = max_length,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    )

    labels_with_ignore_index = torch.where(
        labels.input_ids == tokenizer.pad_token_id, 
        -100, 
        labels.input_ids
    )

    embeddings["target_texts"] = outputs
    embeddings["labels"] = labels_with_ignore_index

    return embeddings

def ProcessJsonData(data_path, tokenizer, max_length=128):
    """处理 T5data JSON 格式的纠错数据集。
    JSON 字段: id, original_text, wrong_ids, correct_text
    """
    inputs = []
    outputs = []

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        wrong_text = item["original_text"]
        correct_text = item["correct_text"]

        inputs.append(f"纠错：{wrong_text}")
        outputs.append(correct_text)

    embeddings = tokenizer(
        inputs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    labels = tokenizer(
        outputs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    labels_with_ignore_index = torch.where(
        labels.input_ids == tokenizer.pad_token_id,
        -100,
        labels.input_ids
    )

    embeddings["target_texts"] = outputs
    embeddings["labels"] = labels_with_ignore_index

    return embeddings

if __name__ == '__main__':

    # ==================== 超参数配置 ====================
    SEED          = 42
    LEARNING_RATE = 1e-4      # 大数据集适当降低学习率，训练更稳定
    NUM_EPOCHS    = 5         # 25万样本，5 轮足够收敛
    BATCH_SIZE    = 16
    WEIGHT_DECAY  = 0.01
    MAX_GRAD_NORM = 1.0
    MAX_LENGTH    = 128       # P99=102，128 覆盖 99%+ 样本
    WARMUP_RATIO  = 0.06      # 大数据集总步数多，缩短预热比例
    SAVE_PATH     = "./best_t5_model_2"
    DATA_PATH     = r"D:\data\T5data"
    # ====================================================

    same_seeds(SEED)
    logger, log_file = setup_logger("./logs_t5")
    logger.info(f"日志文件: {log_file}")

    device = d2l.try_gpu()
    logger.info(f"使用设备: {device}")

    tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")

    train_encodings = ProcessJsonData(DATA_PATH + r"\train.json", tokenizer, max_length=MAX_LENGTH)
    test_encodings = ProcessJsonData(DATA_PATH + r"\dev.json", tokenizer, max_length=MAX_LENGTH)

    train_dataset = T5Dataset(train_encodings)
    test_dataset = T5Dataset(test_encodings)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(test_dataset)}")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if not any (nd in n for nd in no_decay)],
                    "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any (nd in n for nd in no_decay)],
                    "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(optimizer_grouped_parameters, lr = LEARNING_RATE, eps=1e-8)

    total_training_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    accelerator = Accelerator(mixed_precision="fp16")
    model, optim, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optim, train_loader, test_loader, scheduler
    )

    best_exact_match = 0.0
    history_train_loss, history_dev_loss, history_em = [], [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        for batch in pbar:
            optim.zero_grad()

            outputs = model(
                input_ids = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                labels = batch["labels"]
            )

            loss = outputs.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optim.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        exact_match_sum = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Eval] "):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

                test_loss += outputs.loss.item()
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=MAX_LENGTH,
                    num_beams=4, # 束搜索，提升生成质量
                    early_stopping=True
                )

                pred_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                true_texts = batch["target_texts"]

                for pred, true in zip(pred_texts, true_texts):
                    if pred.replace(" ", "") == true.replace(" ", ""):
                        exact_match_sum += 1

                total_samples += len(pred_texts)

        avg_dev_loss = test_loss / len(test_loader)
        epoch_em = exact_match_sum / total_samples

        logger.info("=" * 60)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary")
        logger.info(f"  Train Loss : {avg_train_loss:.4f}")
        logger.info(f"  Dev   Loss : {avg_dev_loss:.4f}")
        logger.info(f"  Dev   EM   : {epoch_em:.4f}  ({exact_match_sum}/{total_samples})")
        logger.info("=" * 60)

        history_train_loss.append(avg_train_loss)
        history_dev_loss.append(avg_dev_loss)
        history_em.append(epoch_em)

        if epoch_em > best_exact_match:
            best_exact_match = epoch_em
            logger.info(f">>> 新的最佳记录! Dev EM: {best_exact_match:.4f}，正在保存模型...")
            model_to_save = accelerator.unwrap_model(model)
            model_to_save.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            logger.info(f">>> 模型已保存至 {SAVE_PATH}")

    # ==================== 绘制训练曲线 ====================
    logger.info("正在生成训练曲线图...")
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_train_loss, 'b-o', label='Train Loss')
    plt.plot(epochs_range, history_dev_loss, 'r-s', label='Dev Loss')
    plt.title('T5 Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_em, 'g-^', label='Dev Exact Match (EM)')
    plt.title('T5 Validation Sentence Correction EM')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join("./logs_t5", f"t5_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info("=" * 60)
    logger.info("T5 纠错模型训练结束！")
    logger.info(f"历史最高 Dev EM  : {best_exact_match:.4f}")
    logger.info("=" * 60)