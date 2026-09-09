#!/usr/bin/env python3
"""
Vanilla Knowledge Distillation (KD) for GLUE (teacher-student).

Main fixes vs the original version:
- Teacher and student are tokenized separately, so different architectures work
  (e.g. DeBERTa teacher + BERT student).
- Fast tokenizer fallback to use_fast=False.
- Robust handling of models/tokenizers with no token_type_ids.
- Teacher verification uses teacher-tokenized validation data.
- Per-config overrides for alpha / temperature (CLI flags remain as fallback).

Vanilla KD loss:
    total_loss = (1 - alpha) * hard_label_loss + alpha * kd_loss

Classification:
    kd_loss = KL( softmax(teacher/T) || softmax(student/T) ) * T^2

Regression (STS-B):
    kd_loss = MSE(student_logits, teacher_logits)

Examples
--------
python run_glue_vanilla_kd.py \
  --task mrpc \
  --teacher_model_path ./teacher_mrpc \
  --student_model bert-base-uncased \
  --output_dir ./runs/mrpc_kd

python run_glue_vanilla_kd.py \
  --task sst2 \
  --teacher_model_path ./teacher_sst2 \
  --student_model google-bert/bert-base-uncased \
  --no_search --learning_rate 2e-5 --batch_size 16 --epochs 6 \
  --alpha 0.7 --temperature 2.0
"""

import argparse
import json
import os
import warnings
from typing import Dict, Optional, Tuple, Any, List


warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm
from datasets import load_dataset, Dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)

# GLUE task -> (text1_key, text2_key or None)
TASK_TO_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "ax": ("premise", "hypothesis"),
}


DEFAULT_CONFIGS = [
    dict(name="config_1p", learning_rate=2e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42, lambda_align=0.5),
    # dict(name="config_2p", learning_rate=3e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42, lambda_align=0.5),
    # dict(name="config_3p", learning_rate=2e-5, batch_size=32, epochs=10, warmup_ratio=0.1,  seed=42, lambda_align=0.5),
    # dict(name="config_4p", learning_rate=3e-5, batch_size=32, epochs=10, warmup_ratio=0.10, seed=42, lambda_align=0.5),
    # dict(name="config_5p", learning_rate=2e-5, batch_size=8, epochs=10, warmup_ratio=0.10, seed=42, lambda_align=0.5),
    # dict(name="config_5p",  learning_rate=2e-5, batch_size=8,  epochs=10, warmup_ratio=0.10, seed=42,
    #      alpha=0.5, temperature=3),
    # dict(name="config_6p",  learning_rate=3e-5, batch_size=8,  epochs=10, warmup_ratio=0.10, seed=42,
    #      alpha=0.5, temperature=3),
    # dict(name="config_13p", learning_rate=2e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42,
    #      alpha=0.7, temperature=3),
    # dict(name="config_15p", learning_rate=2e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42,  
    #      alpha=0.5, temperature=5),
    # dict(name="config_16p", learning_rate=5e-5, batch_size=16, epochs=10, warmup_ratio=0.06, seed=42,
    #      alpha=0.5, temperature=3),
    ]


# -------------------------
# Utilities
# -------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_regression_task(task: str) -> bool:
    return task.lower() == "stsb"


def build_model(model_name_or_path: str, task: str):
    if is_regression_task(task):
        return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=3 if task == "mnli" else 2,
    )


def load_tokenizer_with_fallback(model_name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    except Exception as e:
        print(f"Fast tokenizer failed for {model_name_or_path}: {e}")
        print("Falling back to slow tokenizer.")
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)


def compute_selection_score(metrics: Dict[str, Any], task: str) -> float:
    if "combined_score" in metrics:
        return float(metrics["combined_score"])
    for k in ["f1", "accuracy", "matthews_correlation", "pearson", "spearmanr"]:
        if k in metrics:
            return float(metrics[k])
    if "accuracy_avg" in metrics:
        return float(metrics["accuracy_avg"])
    vals = [v for v in metrics.values() if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else float("-inf")


def build_dataloader(tokenized_dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)


def extract_inputs(batch: Dict[str, torch.Tensor], device: str, prefix: Optional[str] = None) -> Dict[str, torch.Tensor]:
    if prefix is None:
        keys = ["input_ids", "attention_mask", "token_type_ids"]
        return {k: batch[k].to(device) for k in keys if k in batch}

    out = {}
    for base_key in ["input_ids", "attention_mask", "token_type_ids"]:
        pref_key = f"{prefix}_{base_key}"
        if pref_key in batch:
            out[base_key] = batch[pref_key].to(device)
    return out


# -------------------------
# JSON override loading
# -------------------------
def load_json_as_hf_dataset(
    path: str,
    *,
    text1_key: str,
    text2_key: Optional[str],
    label_key: str = "label",
) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        list_candidates = [v for v in raw.values() if isinstance(v, list)]
        if not list_candidates:
            raise ValueError(f"JSON at {path} is a dict but contains no list of examples.")
        examples = list_candidates[0]
    elif isinstance(raw, list):
        examples = raw
    else:
        raise ValueError(f"JSON at {path} must be a list or a dict containing a list. Got: {type(raw)}")

    out = {text1_key: [], label_key: []}
    if text2_key is not None:
        out[text2_key] = []

    for i, ex in enumerate(examples):
        if text1_key not in ex:
            raise KeyError(f"Example {i} missing '{text1_key}'")
        if text2_key is not None and text2_key not in ex:
            raise KeyError(f"Example {i} missing '{text2_key}'")
        if label_key not in ex:
            raise KeyError(f"Example {i} missing '{label_key}'")

        out[text1_key].append(ex[text1_key])
        if text2_key is not None:
            out[text2_key].append(ex[text2_key])
        out[label_key].append(ex[label_key])

    return Dataset.from_dict(out)


def prepare_raw_dataset(
    task: str,
    *,
    train_json: Optional[str] = None,
    json_text1_key: Optional[str] = None,
    json_text2_key: Optional[str] = None,
    json_label_key: str = "label",
):
    dataset = load_dataset("glue", task)

    if "train" not in dataset:
        raise ValueError(f"Task '{task}' does not provide a train split in this GLUE mirror.")

    if train_json:
        k1, k2 = TASK_TO_KEYS[task]
        t1 = json_text1_key or k1
        t2 = json_text2_key if json_text2_key is not None else k2

        train_raw = load_json_as_hf_dataset(
            train_json,
            text1_key=t1,
            text2_key=t2,
            label_key=json_label_key,
        )

        rename_map = {}
        if t1 != k1:
            rename_map[t1] = k1
        if k2 is not None and t2 != k2:
            rename_map[t2] = k2
        if rename_map:
            train_raw = train_raw.rename_columns(rename_map)

        dataset["train"] = train_raw

    if task == "mnli":
        return {
            "train": dataset["train"],
            "validation_matched": dataset["validation_matched"],
            "validation_mismatched": dataset["validation_mismatched"],
        }

    if "validation" not in dataset:
        raise ValueError(f"Task '{task}' does not provide a validation split.")

    return {
        "train": dataset["train"],
        "validation": dataset["validation"],
    }


# -------------------------
# Tokenization
# -------------------------
def tokenize_dataset_for_eval(dataset, tokenizer, task: str, max_length: int):
    k1, k2 = TASK_TO_KEYS[task]

    def preprocess(examples):
        if k2 is None:
            return tokenizer(
                examples[k1],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        return tokenizer(
            examples[k1],
            examples[k2],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(preprocess, batched=True)
    cols = [c for c in ["input_ids", "attention_mask", "token_type_ids", "label"] if c in tokenized.column_names]
    tokenized.set_format(type="torch", columns=cols)
    return tokenized


def tokenize_dataset_for_kd(dataset, student_tokenizer, teacher_tokenizer, task: str, max_length: int):
    k1, k2 = TASK_TO_KEYS[task]

    def preprocess(examples):
        if k2 is None:
            s_enc = student_tokenizer(
                examples[k1],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            t_enc = teacher_tokenizer(
                examples[k1],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        else:
            s_enc = student_tokenizer(
                examples[k1],
                examples[k2],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            t_enc = teacher_tokenizer(
                examples[k1],
                examples[k2],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        out = {"label": examples["label"]}
        for key, val in s_enc.items():
            out[f"s_{key}"] = val
        for key, val in t_enc.items():
            out[f"t_{key}"] = val
        return out

    tokenized = dataset.map(preprocess, batched=True)
    cols = ["label"]

    for prefix in ["s", "t"]:
        for base_key in ["input_ids", "attention_mask", "token_type_ids"]:
            col = f"{prefix}_{base_key}"
            if col in tokenized.column_names:
                cols.append(col)

    tokenized.set_format(type="torch", columns=cols)
    return tokenized


def prepare_tokenized_datasets(
    task: str,
    student_tokenizer,
    teacher_tokenizer,
    max_length: int,
    *,
    train_json: Optional[str] = None,
    json_text1_key: Optional[str] = None,
    json_text2_key: Optional[str] = None,
    json_label_key: str = "label",
):
    raw_splits = prepare_raw_dataset(
        task,
        train_json=train_json,
        json_text1_key=json_text1_key,
        json_text2_key=json_text2_key,
        json_label_key=json_label_key,
    )

    train_kd = tokenize_dataset_for_kd(raw_splits["train"], student_tokenizer, teacher_tokenizer, task, max_length)

    if task == "mnli":
        return {
            "train_kd": train_kd,
            "student_validation_matched": tokenize_dataset_for_eval(raw_splits["validation_matched"], student_tokenizer, task, max_length),
            "student_validation_mismatched": tokenize_dataset_for_eval(raw_splits["validation_mismatched"], student_tokenizer, task, max_length),
            "teacher_validation_matched": tokenize_dataset_for_eval(raw_splits["validation_matched"], teacher_tokenizer, task, max_length),
            "teacher_validation_mismatched": tokenize_dataset_for_eval(raw_splits["validation_mismatched"], teacher_tokenizer, task, max_length),
        }

    return {
        "train_kd": train_kd,
        "student_validation": tokenize_dataset_for_eval(raw_splits["validation"], student_tokenizer, task, max_length),
        "teacher_validation": tokenize_dataset_for_eval(raw_splits["validation"], teacher_tokenizer, task, max_length),
    }


# -------------------------
# Evaluation
# -------------------------
def evaluate_glue(model, dataloader, device: str, task: str) -> Dict[str, Any]:
    model.eval()
    metric = evaluate.load("glue", task)

    preds_list: List[Any] = []
    labels_list: List[Any] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            inputs = extract_inputs(batch, device, prefix=None)
            outputs = model(**inputs)

            if is_regression_task(task):
                preds = outputs.logits.squeeze(-1)
                preds_list.extend(preds.detach().cpu().numpy().tolist())
                labels_list.extend(labels.float().detach().cpu().numpy().tolist())
            else:
                preds = torch.argmax(outputs.logits, dim=-1)
                preds_list.extend(preds.detach().cpu().numpy().tolist())
                labels_list.extend(labels.detach().cpu().numpy().tolist())

    return metric.compute(predictions=preds_list, references=labels_list)


def evaluate_mnli_both(model, matched_loader, mismatched_loader, device: str) -> Dict[str, float]:
    m = evaluate_glue(model, matched_loader, device, "mnli")
    mm = evaluate_glue(model, mismatched_loader, device, "mnli")

    acc_m = float(m.get("accuracy", 0.0))
    acc_mm = float(mm.get("accuracy", 0.0))
    return {
        "accuracy_matched": acc_m,
        "accuracy_mismatched": acc_mm,
        "accuracy_avg": (acc_m + acc_mm) / 2.0,
    }


# -------------------------
# KD Loss
# -------------------------
def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    T = float(temperature)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


# -------------------------
# Saving
# -------------------------
def save_best_student(
    student,
    tokenizer,
    output_dir: str,
    task: str,
    config_name: str,
    epoch: int,
    score: float,
    extra: Dict[str, Any],
) -> str:
    save_dir = os.path.join(output_dir, f"best_student_{task}_{config_name}")
    safe_mkdir(save_dir)

    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    meta = {
        "task": task,
        "config": config_name,
        "epoch": epoch,
        "selection_score": score,
        **extra,
    }
    with open(os.path.join(save_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    torch.save(meta, os.path.join(save_dir, "training_info.pt"))

    print(f"  💾 Saved best student to: {save_dir}")
    return save_dir


# -------------------------
# Training loops
# -------------------------
def train_one_config(
    config: Dict[str, Any],
    args,
    dataset_splits,
    teacher,
    student_tokenizer,
):
    set_seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Per-config overrides (fall back to CLI args if not present in config dict)
    alpha = config.get("alpha", args.alpha)
    temperature = config.get("temperature", args.temperature)

    student = build_model(args.student_model, args.task).to(args.device)

    train_loader = build_dataloader(dataset_splits["train_kd"], config["batch_size"], shuffle=True)
    val_loader = build_dataloader(dataset_splits["student_validation"], config["batch_size"], shuffle=False)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["learning_rate"])

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    print(f"\n{'='*96}")
    print(f"[VANILLA KD] Student={args.student_model}  Teacher={args.teacher_model_path}")
    print(f"Task={args.task}  Config={config['name']}  Device={args.device}  MaxLen={args.max_seq_length}")
    print(
        f"LR={config['learning_rate']}  BS={config['batch_size']}  Epochs={config['epochs']}  "
        f"Warmup={config['warmup_ratio']}  Seed={config['seed']}"
    )
    print(f"alpha={alpha}  temperature={temperature}")
    if args.train_json:
        print(f"Train override: {args.train_json}")
    print(f"Total steps={total_steps}  Warmup steps={warmup_steps}")
    print(f"{'='*96}")

    training_history: List[Dict[str, Any]] = []
    best = {
        "score": float("-inf"),
        "epoch": None,
        "metrics": None,
        "config": config["name"],
        "save_dir": None,
    }

    for epoch in range(config["epochs"]):
        student.train()
        total_loss = 0.0
        total_hard = 0.0
        total_kd = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            labels = batch["label"].to(args.device)

            s_inputs = extract_inputs(batch, args.device, prefix="s")
            t_inputs = extract_inputs(batch, args.device, prefix="t")

            with torch.no_grad():
                t_logits = teacher(**t_inputs).logits

            s_logits = student(**s_inputs).logits

            if is_regression_task(args.task):
                labels_f = labels.float()
                hard_loss = mse_loss_fn(s_logits.squeeze(-1), labels_f)
                kd_loss = mse_loss_fn(s_logits.squeeze(-1), t_logits.squeeze(-1).detach())
            else:
                hard_loss = ce_loss_fn(s_logits, labels.long())
                kd_loss = kd_kl_loss(s_logits, t_logits.detach(), temperature)

            loss = (1.0 - alpha) * hard_loss + alpha * kd_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            total_hard += float(hard_loss.item())
            total_kd += float(kd_loss.item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                hard=f"{hard_loss.item():.4f}",
                kd=f"{kd_loss.item():.4f}",
            )

        avg_loss = total_loss / max(1, len(train_loader))
        avg_hard = total_hard / max(1, len(train_loader))
        avg_kd = total_kd / max(1, len(train_loader))

        val_metrics = evaluate_glue(student, val_loader, args.device, args.task)
        score = compute_selection_score(val_metrics, args.task)

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_hard": avg_hard,
            "train_kd": avg_kd,
            "val_metrics": val_metrics,
            "selection_score": score,
        }
        training_history.append(row)

        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in val_metrics.items()
        )
        print(
            f"\nEpoch {epoch+1}: train_loss={avg_loss:.4f} "
            f"(hard={avg_hard:.4f}, kd={avg_kd:.4f}) | {metrics_str} | selection_score={score:.4f}"
        )

        if score > best["score"]:
            best.update({"score": score, "epoch": epoch + 1, "metrics": val_metrics})
            print(f"  >>> New best @ epoch {best['epoch']} (score={best['score']:.4f})")

            save_dir = save_best_student(
                student=student,
                tokenizer=student_tokenizer,
                output_dir=args.output_dir,
                task=args.task,
                config_name=config["name"],
                epoch=epoch + 1,
                score=score,
                extra={
                    "teacher_model_path": args.teacher_model_path,
                    "student_model": args.student_model,
                    "alpha": alpha,
                    "temperature": temperature,
                    "train_json": args.train_json,
                },
            )
            best["save_dir"] = save_dir

    best["training_history"] = training_history
    return best


def train_one_config_mnli(
    config: Dict[str, Any],
    args,
    dataset_splits,
    teacher,
    student_tokenizer,
):
    set_seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Per-config overrides (fall back to CLI args if not present in config dict)
    alpha = config.get("alpha", args.alpha)
    temperature = config.get("temperature", args.temperature)

    student = build_model(args.student_model, "mnli").to(args.device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    train_loader = build_dataloader(dataset_splits["train_kd"], config["batch_size"], shuffle=True)
    val_m_loader = build_dataloader(dataset_splits["student_validation_matched"], config["batch_size"], shuffle=False)
    val_mm_loader = build_dataloader(dataset_splits["student_validation_mismatched"], config["batch_size"], shuffle=False)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["learning_rate"])

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ce_loss_fn = nn.CrossEntropyLoss()

    print(f"\n{'='*96}")
    print(f"[VANILLA KD] MNLI Student={args.student_model}  Teacher={args.teacher_model_path}")
    print(
        f"Config={config['name']}  Device={args.device}  MaxLen={args.max_seq_length}  "
        f"LR={config['learning_rate']}  BS={config['batch_size']}  Epochs={config['epochs']}  "
        f"Warmup={config['warmup_ratio']}  Seed={config['seed']}"
    )
    print(f"alpha={alpha}  temperature={temperature}")
    print(f"Total steps={total_steps}  Warmup steps={warmup_steps}")
    print(f"{'='*96}")

    training_history: List[Dict[str, Any]] = []
    best = {
        "score": float("-inf"),
        "epoch": None,
        "metrics": None,
        "config": config["name"],
        "save_dir": None,
    }

    for epoch in range(config["epochs"]):
        student.train()
        total_loss = 0.0
        total_hard = 0.0
        total_kd = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            labels = batch["label"].to(args.device)

            s_inputs = extract_inputs(batch, args.device, prefix="s")
            t_inputs = extract_inputs(batch, args.device, prefix="t")

            with torch.no_grad():
                t_logits = teacher(**t_inputs).logits

            s_logits = student(**s_inputs).logits

            hard_loss = ce_loss_fn(s_logits, labels.long())
            kd_loss = kd_kl_loss(s_logits, t_logits.detach(), temperature)
            loss = (1.0 - alpha) * hard_loss + alpha * kd_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            total_hard += float(hard_loss.item())
            total_kd += float(kd_loss.item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                hard=f"{hard_loss.item():.4f}",
                kd=f"{kd_loss.item():.4f}",
            )

        avg_loss = total_loss / max(1, len(train_loader))
        avg_hard = total_hard / max(1, len(train_loader))
        avg_kd = total_kd / max(1, len(train_loader))

        val_metrics = evaluate_mnli_both(student, val_m_loader, val_mm_loader, args.device)
        score = float(val_metrics["accuracy_avg"])

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_hard": avg_hard,
            "train_kd": avg_kd,
            "val_metrics": val_metrics,
            "selection_score": score,
        })

        print(
            f"\nEpoch {epoch+1}: train_loss={avg_loss:.4f} "
            f"(hard={avg_hard:.4f}, kd={avg_kd:.4f}) | "
            f"acc_m={val_metrics['accuracy_matched']:.4f}, "
            f"acc_mm={val_metrics['accuracy_mismatched']:.4f}, "
            f"acc_avg={score:.4f}"
        )

        if score > best["score"]:
            best.update({"score": score, "epoch": epoch + 1, "metrics": val_metrics})
            print(f"  >>> New best @ epoch {best['epoch']} (score={best['score']:.4f})")

            save_dir = save_best_student(
                student=student,
                tokenizer=student_tokenizer,
                output_dir=args.output_dir,
                task="mnli",
                config_name=config["name"],
                epoch=epoch + 1,
                score=score,
                extra={
                    "teacher_model_path": args.teacher_model_path,
                    "student_model": args.student_model,
                    "alpha": alpha,
                    "temperature": temperature,
                    "train_json": args.train_json,
                },
            )
            best["save_dir"] = save_dir

    best["training_history"] = training_history
    return best


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=sorted(TASK_TO_KEYS.keys()))

    p.add_argument("--teacher_model_path", type=str, required=True,
                   help="Path to a fine-tuned teacher checkpoint (HF save_pretrained dir).")
    p.add_argument("--student_model", type=str, default="bert-base-uncased",
                   help="Student model name or path.")

    p.add_argument("--output_dir", type=str, default="./vanilla_kd")

    p.add_argument("--alpha", type=float, default=0.5,
                   help="Default weight on KD loss. (1-alpha) on hard labels. "
                        "Overridable per config via dict key 'alpha'.")
    p.add_argument("--temperature", type=float, default=3.0,
                   help="Default KD temperature. Overridable per config via dict key 'temperature'.")
    p.add_argument("--verify_teacher", action="store_true", help="Evaluate teacher on validation before training.")

    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--train_json", type=str, default=None, help="Optional JSON file to use as train set.")
    p.add_argument("--json_text1_key", type=str, default=None, help="Override JSON key for text1.")
    p.add_argument("--json_text2_key", type=str, default=None, help="Override JSON key for text2.")
    p.add_argument("--json_label_key", type=str, default="label", help="Override JSON key for label.")

    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--no_search", action="store_true", help="Run only one config from CLI params below.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    safe_mkdir(args.output_dir)

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1].")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")

    print("=" * 96)
    print("GLUE Vanilla KD (teacher-student)")
    print(f"Task: {args.task}")
    print(f"Teacher: {args.teacher_model_path}")
    print(f"Student: {args.student_model}")
    print(f"Device: {args.device}")
    print(f"CLI default alpha={args.alpha}, temperature={args.temperature} "
          f"(per-config dict values override these)")
    if args.train_json:
        print(f"Train JSON override: {args.train_json}")
    print("=" * 96)

    print("\nLoading tokenizers...")
    student_tokenizer = load_tokenizer_with_fallback(args.student_model)
    teacher_tokenizer = load_tokenizer_with_fallback(args.teacher_model_path)

    dataset_splits = prepare_tokenized_datasets(
        args.task,
        student_tokenizer,
        teacher_tokenizer,
        args.max_seq_length,
        train_json=args.train_json,
        json_text1_key=args.json_text1_key,
        json_text2_key=args.json_text2_key,
        json_label_key=args.json_label_key,
    )

    print("\nLoading teacher model...")
    teacher = build_model(args.teacher_model_path, args.task).to(args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if args.verify_teacher:
        print("\nVerifying teacher on validation...")
        if args.task == "mnli":
            teacher_m_loader = build_dataloader(dataset_splits["teacher_validation_matched"], args.batch_size, shuffle=False)
            teacher_mm_loader = build_dataloader(dataset_splits["teacher_validation_mismatched"], args.batch_size, shuffle=False)
            teacher_metrics = evaluate_mnli_both(teacher, teacher_m_loader, teacher_mm_loader, args.device)
            print(
                f"[Teacher val] acc_matched={teacher_metrics['accuracy_matched']:.4f}, "
                f"acc_mismatched={teacher_metrics['accuracy_mismatched']:.4f}, "
                f"acc_avg={teacher_metrics['accuracy_avg']:.4f}"
            )
        else:
            teacher_val_loader = build_dataloader(dataset_splits["teacher_validation"], args.batch_size, shuffle=False)
            teacher_metrics = evaluate_glue(teacher, teacher_val_loader, args.device, args.task)
            teacher_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in teacher_metrics.items()
            )
            print(f"[Teacher val] {teacher_str}")

    if args.no_search:
        configs = [dict(
            name="single_run",
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
        )]
    else:
        configs = DEFAULT_CONFIGS

    results = []
    for cfg in configs:
        try:
            if args.task == "mnli":
                best = train_one_config_mnli(cfg, args, dataset_splits, teacher, student_tokenizer)
            else:
                best = train_one_config(cfg, args, dataset_splits, teacher, student_tokenizer)
            results.append(best)
        except Exception as e:
            print(f"Error with {cfg['name']}: {e}")
            continue

    if not results:
        raise RuntimeError("No successful runs. Check your environment / task name / teacher path.")

    results_sorted = sorted(results, key=lambda r: float(r["score"]), reverse=True)
    best = results_sorted[0]

    print("\nRESULTS SUMMARY (sorted by selection score)")
    for r in results_sorted:
        print(
            f"{r['config']:12s} | best_epoch={r['epoch']} | score={r['score']:.4f} | "
            f"save_dir={r.get('save_dir')} | metrics={r['metrics']}"
        )

    out = {
        "task": args.task,
        "teacher_model_path": args.teacher_model_path,
        "student_model": args.student_model,
        "device": args.device,
        "max_seq_length": args.max_seq_length,
        "alpha_cli_default": args.alpha,
        "temperature_cli_default": args.temperature,
        "search": (not args.no_search),
        "train_json": args.train_json,
        "results": [
            {
                "config": r["config"],
                "score": r["score"],
                "epoch": r["epoch"],
                "metrics": r["metrics"],
                "save_dir": r.get("save_dir"),
            }
            for r in results_sorted
        ],
        "best": {
            "config": best["config"],
            "score": best["score"],
            "epoch": best["epoch"],
            "metrics": best["metrics"],
            "save_dir": best.get("save_dir"),
        },
    }
    out_path = os.path.join(
        args.output_dir,
        f"{args.task}{args.student_model.replace('/', '')}_vanilla_kd_results.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nBest config: {best['config']} | epoch={best['epoch']} | score={best['score']:.4f}")
    print(f"Saved run summary to: {out_path}")


if __name__ == "__main__":
    main()