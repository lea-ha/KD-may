#!/usr/bin/env python3
"""
Evaluate saved mismatch-KD student checkpoints (config_1p to config_4p).
Reports MCC and accuracy for CoLA.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset

TASK = "cola"
CONFIGS = ["config_1p", "config_2p", "config_3p", "config_4p"]
BASE_DIR = "./output/vanilla_kd_cola"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tokenize_dataset(dataset, tokenizer, max_length):
    def preprocess(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)

    tokenized = dataset.map(preprocess, batched=True)
    cols = [c for c in ["input_ids", "attention_mask", "token_type_ids", "label"] if c in tokenized.column_names]
    tokenized.set_format(type="torch", columns=cols)
    return tokenized


def evaluate_checkpoint(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(DEVICE)
    model.eval()

    raw = load_dataset("glue", "cola")
    val_dataset = tokenize_dataset(raw["validation"], tokenizer, MAX_SEQ_LENGTH)
    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    glue_metric = evaluate.load("glue", "cola")
    preds_list, labels_list = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(DEVICE)
            inputs = {k: batch[k].to(DEVICE) for k in ["input_ids", "attention_mask", "token_type_ids"] if k in batch}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(labels.cpu().numpy().tolist())

    metrics = glue_metric.compute(predictions=preds_list, references=labels_list)
    correct = sum(p == l for p, l in zip(preds_list, labels_list))
    metrics["accuracy"] = correct / len(labels_list)
    return metrics


def main():
    print(f"Task: CoLA  Device: {DEVICE}")
    print("=" * 60)

    results = []
    for config_name in CONFIGS:
        checkpoint_dir = os.path.join(BASE_DIR, f"best_student_cola_{config_name}")

        if not os.path.isdir(checkpoint_dir):
            print(f"[{config_name}] NOT FOUND at {checkpoint_dir} — skipping")
            continue

        print(f"[{config_name}] Evaluating {checkpoint_dir} ...")
        try:
            metrics = evaluate_checkpoint(checkpoint_dir)
            mcc = metrics["matthews_correlation"]
            acc = metrics["accuracy"]
            print(f"  MCC={mcc:.4f}  Acc={acc:.4f}")
            results.append({"config": config_name, "mcc": mcc, "accuracy": acc})
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"{'Config':<14} {'MCC':>8} {'Accuracy':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['config']:<14} {r['mcc']:>8.4f} {r['accuracy']:>10.4f}")

    if results:
        best = max(results, key=lambda x: x["mcc"])
        print(f"\nBest by MCC: {best['config']}  MCC={best['mcc']:.4f}  Acc={best['accuracy']:.4f}")

        out_path = os.path.join(BASE_DIR, "cola_eval_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()