# A-Simple-Question-Answering-System-Based-on-BERT

Based on cmrc2018 dataset, [here](https://github.com/ymcui/cmrc2018/tree/master/data) is the data

Pretrained model is "bert-base-chinese" from BertTokenizerFast

run the Bert.py and use the test.py to predict

This repository contains a PyTorch implementation for an Extractive Question Answering (QA) system. It fine-tunes the `bert-base-chinese` pre-trained model to extract exact answer spans from a given context based on a user's query.

## Overview

The project leverages the Hugging Face `transformers` library for model architecture and tokenization. It also utilizes the `accelerate` library to enable efficient Mixed Precision Training (FP16), significantly speeding up the training process on modern GPUs while reducing memory consumption.

##roject Structure

    ├── Bert_QA/                        # Directory for saved model weights and configs (git-ignored)
    ├── bert.py                         # Main training script (data loading, fine-tuning, saving)
    ├── dataset.py                      # Dataset processing and loading utilities
    ├── test.py                         # Inference script for model evaluation
    ├── .gitignore                      # Git ignore file (excludes heavy model weights)
    └── README.md                       # Project documentation

## Dataset

This model is trained on the **CMRC2018** (Chinese Machine Reading Comprehension) dataset.
* The dataset consists of human-annotated questions and answers based on Wikipedia articles.
* Ensure you have the dataset files (`cmrc2018_train.json` and `cmrc2018_dev.json`) downloaded locally.
* Update the file paths in `bert.py` to point to your local dataset directory before training.

## 🛠️ Requirements

Ensure you have the following dependencies installed:

    pip install torch numpy pandas tqdm
    pip install transformers accelerate

*Note: The script automatically detects and utilizes an available GPU via `d2l.try_gpu()` and `accelerate`.*

## Usage

### Training
To initiate the fine-tuning process, run the training script:

    python bert.py

* **Optimizer:** AdamW (`lr=5e-5`)
* **Precision:** Automatic Mixed Precision (FP16) via `accelerate`
* **Checkpointing:** The model and tokenizer are saved locally (e.g., to the root directory or `Bert_QA/`) every 5 epochs and at the end of training.

### Inference
*(Assuming `test.py` is configured for prediction)*
After training, you can use the saved weights to run predictions on new context-question pairs using the `test.py` script.
