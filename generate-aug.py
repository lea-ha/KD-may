#!/usr/bin/env python3
"""
Error-targeted Data Augmentation for GLUE tasks.

Targets samples where: Student is WRONG + Teacher is CORRECT.
For pair tasks: generates 3 augmentations per sample (s1', s2), (s1, s2'), (s1', s2').
For single-sentence tasks: generates 1 augmentation per sample (s1').

Paraphrases are generated via OpenAI gpt-4o-mini.
Augmented samples are filtered: teacher must still predict correctly on the
paraphrased input (prediction consistency check).

Output JSON matches the format expected by run_glue_align_kd.py.

Examples
--------
python generate-aug.py \\
  --task mrpc \\
  --teacher_model_path ./teacher_mrpc \\
  --student_model_path ./vanilla_kd/best_student_mrpc_config_1p \\
  --output_path ./data/mrpc/augmented.json

python generate-aug.py \\
  --task sst2 \\
  --teacher_model_path ./teacher_sst2 \\
  --student_model_path ./vanilla_kd/best_student_sst2_config_1p \\
  --output_path ./data/sst2/augmented.json
"""

import argparse
import json
import os
import random
import warnings
from typing import Dict, Optional, Tuple, Any, List

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

from tqdm import tqdm
from datasets import load_dataset
import evaluate as hf_evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
)
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# Constants
# =========================================================
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

TASK_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "stsb": 1,
    "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2, "ax": 3,
}

PAIR_PARAPHRASE_TASKS = {"mrpc", "qqp", "stsb"}

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# =========================================================
# Utilities
# =========================================================
def is_regression_task(task: str) -> bool:
    return task.lower() == "stsb"


def is_pair_task(task: str) -> bool:
    _, k2 = TASK_TO_KEYS[task]
    return k2 is not None


def load_tokenizer_with_fallback(model_name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)


# =========================================================
# Dataset wrapper for batch prediction
# =========================================================
class GLUEPredictionDataset(TorchDataset):
    """Tokenized dataset for batch prediction on any GLUE task."""

    def __init__(self, hf_dataset, tokenizer, task: str, max_length: int):
        self.items = []
        k1, k2 = TASK_TO_KEYS[task]

        for i in range(len(hf_dataset)):
            item = hf_dataset[i]
            if k2 is not None:
                enc = tokenizer(
                    item[k1], item[k2],
                    padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt",
                )
            else:
                enc = tokenizer(
                    item[k1],
                    padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt",
                )

            entry = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(item["label"], dtype=torch.float if is_regression_task(task) else torch.long),
            }
            if "token_type_ids" in enc:
                entry["token_type_ids"] = enc["token_type_ids"].squeeze(0)
            self.items.append(entry)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# =========================================================
# Prediction helpers
# =========================================================
@torch.no_grad()
def get_batch_predictions(model, dataloader, device: str):
    """Get predictions, labels, and probabilities for a full dataset."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        labels = batch["label"].to(device)
        inputs = {k: batch[k].to(device) for k in ["input_ids", "attention_mask", "token_type_ids"]
                  if k in batch}
        logits = model(**inputs).logits

        if logits.shape[-1] == 1:
            preds = logits.squeeze(-1)
            probs = preds
        else:
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy(),
    )


@torch.no_grad()
def get_single_prediction(model, tokenizer, texts: List[str], device: str,
                          max_length: int, task: str):
    """
    Get teacher prediction for a single example.
    texts: [text1] for single-sentence, [text1, text2] for pair tasks.
    Returns (pred, confidence).
    """
    if len(texts) == 2:
        enc = tokenizer(texts[0], texts[1], padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    else:
        enc = tokenizer(texts[0], padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in enc.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
    logits = model(**inputs).logits

    if is_regression_task(task):
        pred_val = logits.squeeze(-1).item()
        return pred_val, 1.0
    else:
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).item()
        conf = probs[0, pred].item()
        return pred, conf


# =========================================================
# Metrics
# =========================================================
def compute_task_metrics(preds, labels, task: str):
    """Compute GLUE metrics for a task."""
    metric = hf_evaluate.load("glue", task)
    if is_regression_task(task):
        metric.add_batch(predictions=preds.tolist(), references=labels.tolist())
    else:
        metric.add_batch(predictions=preds.astype(int), references=labels.astype(int))
    return metric.compute()


# =========================================================
# Error detection
# =========================================================
def find_error_indices(student_preds, teacher_preds, labels, task: str):
    """
    Find indices where student is wrong AND teacher is correct.
    For regression (STS-B): use threshold-based correctness (within 0.5 of label).
    """
    if is_regression_task(task):
        teacher_correct = np.abs(teacher_preds - labels) < 0.5
        student_wrong = np.abs(student_preds - labels) >= 0.5
    else:
        teacher_correct = teacher_preds == labels
        student_wrong = student_preds != labels

    error_mask = student_wrong & teacher_correct
    return np.where(error_mask)[0]


# =========================================================
# Paraphrase generation
# =========================================================
SINGLE_SENTENCE_FEW_SHOT = [
    {"sentence1": "The movie was absolutely wonderful and captivating.",
     "sentence2": "The film was truly marvelous and engaging."},
    {"sentence1": "Scientists have discovered a new species of butterfly in the Amazon rainforest.",
     "sentence2": "Researchers found a previously unknown type of butterfly in the Amazon jungle."},
    {"sentence1": "The company reported a significant increase in quarterly revenue.",
     "sentence2": "The firm announced a substantial rise in its earnings for the quarter."},
    {"sentence1": "Heavy rainfall caused widespread flooding in several coastal cities.",
     "sentence2": "Intense downpours led to extensive floods across multiple cities along the coast."},
    {"sentence1": "The new policy aims to reduce carbon emissions by 50 percent within a decade.",
     "sentence2": "The recently introduced regulation seeks to cut carbon output in half over the next ten years."},
]


def build_pair_paraphrase_pool(train_dataset, task: str) -> List[Dict[str, str]]:
    """
    Build a pool of paraphrase pairs from positive-label examples in the training set.
    For pair tasks only.
    """
    k1, k2 = TASK_TO_KEYS[task]
    if k2 is None:
        return []

    pairs = []
    for i in range(len(train_dataset)):
        ex = train_dataset[i]
        if is_regression_task(task):
            if ex["label"] > 3.5:
                pairs.append({"sentence1": ex[k1], "sentence2": ex[k2]})
        else:
            if ex["label"] == 1:
                pairs.append({"sentence1": ex[k1], "sentence2": ex[k2]})

    return pairs


def generate_paraphrase_pair_task(sentence: str, few_shot_pool: List[Dict],
                                  num_few_shot: int, client: OpenAI) -> Optional[str]:
    """Generate a paraphrase using few-shot examples from the pool."""
    examples = random.sample(few_shot_pool, min(num_few_shot, len(few_shot_pool)))

    prompt = ("Generate a paraphrase of the following text using different words "
              "and sentence structures while still conveying the same meaning.\n\n")
    for ex in examples:
        prompt += f"{ex['sentence1']}, in other words {ex['sentence2']}\n"
    prompt += f"\n{sentence}, in other words"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Paraphrase error: {e}")
        return None


def generate_paraphrase_single_sentence(sentence: str, client: OpenAI) -> Optional[str]:
    """Generate a paraphrase for single-sentence tasks using hardcoded few-shot examples."""
    prompt = ("Generate a paraphrase of the following text using different words "
              "and sentence structures while still conveying the same meaning.\n\n")
    for ex in SINGLE_SENTENCE_FEW_SHOT:
        prompt += f"{ex['sentence1']}, in other words {ex['sentence2']}\n"
    prompt += f"\n{sentence}, in other words"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Paraphrase error: {e}")
        return None


# =========================================================
# Augmentation
# =========================================================
def augment_pair_task(
    teacher, teacher_tokenizer, train_dataset,
    few_shot_pool: List[Dict], client: OpenAI,
    error_indices: np.ndarray, task: str, args,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Augment pair-task samples: generate (s1', s2), (s1, s2'), (s1', s2') per error sample."""
    k1, k2 = TASK_TO_KEYS[task]
    augmented = []
    counts = {"s1": 0, "s2": 0, "s1s2": 0, "discarded": 0}

    for idx in tqdm(error_indices, desc="Augmenting error samples (pair task, 3x per sample)"):
        item = train_dataset[int(idx)]
        s1, s2 = item[k1], item[k2]
        orig_label = item["label"]

        pred_orig, conf_orig = get_single_prediction(
            teacher, teacher_tokenizer, [s1, s2], args.device, args.max_seq_length, task
        )
        if is_regression_task(task):
            if abs(pred_orig - orig_label) >= 0.5:
                counts["discarded"] += 1
                continue
        else:
            if pred_orig != orig_label:
                counts["discarded"] += 1
                continue

        s1_prime = generate_paraphrase_pair_task(s1, few_shot_pool, args.num_few_shot, client)
        if s1_prime is None:
            counts["discarded"] += 1
            continue

        s2_prime = generate_paraphrase_pair_task(s2, few_shot_pool, args.num_few_shot, client)
        if s2_prime is None:
            counts["discarded"] += 1
            continue

        variants = [
            ("s1", s1_prime, s2),
            ("s2", s1, s2_prime),
            ("s1s2", s1_prime, s2_prime),
        ]

        for side, new_s1, new_s2 in variants:
            pred_new, conf_new = get_single_prediction(
                teacher, teacher_tokenizer, [new_s1, new_s2],
                args.device, args.max_seq_length, task
            )

            if is_regression_task(task):
                consistent = abs(pred_new - pred_orig) < 0.5
            else:
                consistent = pred_new == pred_orig

            if consistent:
                augmented.append({
                    "original_index": int(idx),
                    "task": task,
                    "text1_key": k1,
                    "text2_key": k2,
                    "original_text1": s1,
                    "original_text2": s2,
                    "paraphrased_side": side,
                    "text1": new_s1,
                    "text2": new_s2,
                    "label": int(orig_label) if not is_regression_task(task) else float(orig_label),
                    "teacher_pred_original": pred_orig if is_regression_task(task) else int(pred_orig),
                    "teacher_pred_new": pred_new if is_regression_task(task) else int(pred_new),
                    "teacher_conf_original": float(conf_orig),
                    "teacher_conf_new": float(conf_new),
                    "selection_reason": "student_wrong_teacher_correct",
                })
                counts[side] += 1
            else:
                counts["discarded"] += 1

    total = counts["s1"] + counts["s2"] + counts["s1s2"]
    stats = {
        "num_candidates": len(error_indices),
        "successful_s1": counts["s1"],
        "successful_s2": counts["s2"],
        "successful_s1s2": counts["s1s2"],
        "total_successful": total,
        "discarded": counts["discarded"],
        "success_rate": total / (len(error_indices) * 3) if len(error_indices) > 0 else 0,
    }
    return augmented, stats


def augment_single_sentence_task(
    teacher, teacher_tokenizer, train_dataset,
    client: OpenAI, error_indices: np.ndarray,
    task: str, args,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Augment single-sentence task samples: generate 1 paraphrase per error sample."""
    k1, _ = TASK_TO_KEYS[task]
    augmented = []
    counts = {"s1": 0, "discarded": 0}

    for idx in tqdm(error_indices, desc="Augmenting error samples (single-sentence, 1x per sample)"):
        item = train_dataset[int(idx)]
        s1 = item[k1]
        orig_label = item["label"]

        pred_orig, conf_orig = get_single_prediction(
            teacher, teacher_tokenizer, [s1], args.device, args.max_seq_length, task
        )
        if pred_orig != orig_label:
            counts["discarded"] += 1
            continue

        s1_prime = generate_paraphrase_single_sentence(s1, client)
        if s1_prime is None:
            counts["discarded"] += 1
            continue

        pred_new, conf_new = get_single_prediction(
            teacher, teacher_tokenizer, [s1_prime], args.device, args.max_seq_length, task
        )

        if pred_new == pred_orig:
            augmented.append({
                "original_index": int(idx),
                "task": task,
                "text1_key": k1,
                "text2_key": None,
                "original_text1": s1,
                "original_text2": None,
                "paraphrased_side": "s1",
                "text1": s1_prime,
                "text2": None,
                "label": int(orig_label),
                "teacher_pred_original": int(pred_orig),
                "teacher_pred_new": int(pred_new),
                "teacher_conf_original": float(conf_orig),
                "teacher_conf_new": float(conf_new),
                "selection_reason": "student_wrong_teacher_correct",
            })
            counts["s1"] += 1
        else:
            counts["discarded"] += 1

    stats = {
        "num_candidates": len(error_indices),
        "successful_s1": counts["s1"],
        "total_successful": counts["s1"],
        "discarded": counts["discarded"],
        "success_rate": counts["s1"] / len(error_indices) if len(error_indices) > 0 else 0,
    }
    return augmented, stats


# =========================================================
# Print helpers
# =========================================================
def print_metrics_table(task: str, teacher_train, student_train, teacher_val, student_val):
    """Print a compact metrics comparison table."""
    keys = sorted(set(list(teacher_train.keys()) + list(student_train.keys())))
    header_cols = "".join(f"{k:>12s}" for k in keys)

    print(f"\n{'='*60}")
    print(f"{'':20s}{header_cols}")
    print(f"{'-'*60}")
    for label, metrics in [
        ("Teacher (train)", teacher_train),
        ("Student (train)", student_train),
        ("Teacher (val)", teacher_val),
        ("Student (val)", student_val),
    ]:
        vals = "".join(f"{float(metrics.get(k, 0)):>12.4f}" for k in keys)
        print(f"{label:20s}{vals}")
    print(f"{'='*60}")


# =========================================================
# Args
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Error-targeted data augmentation for GLUE")
    p.add_argument("--task", type=str, required=True, choices=sorted(TASK_TO_KEYS.keys()))
    p.add_argument("--teacher_model_path", type=str, required=True,
                   help="Path to fine-tuned teacher checkpoint.")
    p.add_argument("--student_model_path", type=str, required=True,
                   help="Path to vanilla KD best student checkpoint.")
    p.add_argument("--output_path", type=str, required=True,
                   help="Output path for the augmented JSON file.")
    p.add_argument("--num_few_shot", type=int, default=15,
                   help="Number of few-shot examples for paraphrase prompt (pair tasks).")
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    task = args.task
    k1, k2 = TASK_TO_KEYS[task]
    pair_task = is_pair_task(task)
    num_labels = TASK_NUM_LABELS[task]

    print("=" * 80)
    print(f"ERROR-TARGETED DATA AUGMENTATION")
    print(f"Task: {task} ({'pair' if pair_task else 'single-sentence'})")
    print(f"Teacher: {args.teacher_model_path}")
    print(f"Student: {args.student_model_path}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # --- Load tokenizers ---
    print("\nLoading tokenizers...")
    teacher_tokenizer = load_tokenizer_with_fallback(args.teacher_model_path)
    student_tokenizer = load_tokenizer_with_fallback(args.student_model_path)

    # --- Load models ---
    print("Loading models...")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_model_path, num_labels=num_labels
    ).to(args.device).eval()
    student = AutoModelForSequenceClassification.from_pretrained(
        args.student_model_path, num_labels=num_labels
    ).to(args.device).eval()

    for p in teacher.parameters():
        p.requires_grad = False
    for p in student.parameters():
        p.requires_grad = False

    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")

    # --- Load GLUE data ---
    print("\nLoading GLUE data...")
    dataset = load_dataset("glue", task)
    train_dataset = dataset["train"]

    if task == "mnli":
        val_dataset = dataset["validation_matched"]
        print("  (Using validation_matched for MNLI)")
    else:
        val_dataset = dataset["validation"]

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    # --- Build prediction datasets ---
    print("\nTokenizing for prediction...")
    teacher_train_ds = GLUEPredictionDataset(train_dataset, teacher_tokenizer, task, args.max_seq_length)
    student_train_ds = GLUEPredictionDataset(train_dataset, student_tokenizer, task, args.max_seq_length)
    teacher_val_ds = GLUEPredictionDataset(val_dataset, teacher_tokenizer, task, args.max_seq_length)
    student_val_ds = GLUEPredictionDataset(val_dataset, student_tokenizer, task, args.max_seq_length)

    teacher_train_loader = DataLoader(teacher_train_ds, batch_size=args.batch_size, shuffle=False)
    student_train_loader = DataLoader(student_train_ds, batch_size=args.batch_size, shuffle=False)
    teacher_val_loader = DataLoader(teacher_val_ds, batch_size=args.batch_size, shuffle=False)
    student_val_loader = DataLoader(student_val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Get predictions ---
    print("\nGetting predictions...")
    print("  [Train - Teacher]")
    teacher_train_preds, train_labels, _ = get_batch_predictions(teacher, teacher_train_loader, args.device)
    print("  [Train - Student]")
    student_train_preds, _, _ = get_batch_predictions(student, student_train_loader, args.device)
    print("  [Val - Teacher]")
    teacher_val_preds, val_labels, _ = get_batch_predictions(teacher, teacher_val_loader, args.device)
    print("  [Val - Student]")
    student_val_preds, _, _ = get_batch_predictions(student, student_val_loader, args.device)

    # --- Metrics ---
    t_train_m = compute_task_metrics(teacher_train_preds, train_labels, task)
    s_train_m = compute_task_metrics(student_train_preds, train_labels, task)
    t_val_m = compute_task_metrics(teacher_val_preds, val_labels, task)
    s_val_m = compute_task_metrics(student_val_preds, val_labels, task)

    print_metrics_table(task, t_train_m, s_train_m, t_val_m, s_val_m)

    # --- Find error indices ---
    error_indices = find_error_indices(student_train_preds, teacher_train_preds, train_labels, task)

    if is_regression_task(task):
        teacher_correct = np.abs(teacher_train_preds - train_labels) < 0.5
        student_wrong = np.abs(student_train_preds - train_labels) >= 0.5
    else:
        teacher_correct = teacher_train_preds == train_labels
        student_wrong = student_train_preds != train_labels

    print(f"\n  Total train:     {len(train_labels)}")
    print(f"  Student errors:  {student_wrong.sum()} ({student_wrong.mean()*100:.1f}%)")
    print(f"  Teacher correct: {teacher_correct.sum()} ({teacher_correct.mean()*100:.1f}%)")
    if pair_task:
        print(f"  Target (both):   {len(error_indices)} -> up to {len(error_indices)*3} augmentations")
    else:
        print(f"  Target (both):   {len(error_indices)} -> up to {len(error_indices)} augmentations")

    if len(error_indices) == 0:
        print("\nNo error samples found. Nothing to augment.")
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        print(f"Saved empty augmentation file to: {args.output_path}")
        return

    # --- OpenAI client ---
    client = OpenAI(api_key=OPENAI_API_KEY)

    # --- Augment ---
    print(f"\n{'='*80}")
    print("AUGMENTING")
    print(f"{'='*80}")

    if pair_task:
        if task in PAIR_PARAPHRASE_TASKS:
            few_shot_pool = build_pair_paraphrase_pool(train_dataset, task)
            print(f"  Paraphrase few-shot pool: {len(few_shot_pool)} examples (task-specific pairs)")
        else:
            few_shot_pool = SINGLE_SENTENCE_FEW_SHOT
            print(f"  Paraphrase few-shot pool: {len(few_shot_pool)} examples (generic single-sentence)")
        augmented, stats = augment_pair_task(
            teacher, teacher_tokenizer, train_dataset,
            few_shot_pool, client, error_indices, task, args,
        )
    else:
        augmented, stats = augment_single_sentence_task(
            teacher, teacher_tokenizer, train_dataset,
            client, error_indices, task, args,
        )

    # --- Save ---
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"  Candidates:       {stats['num_candidates']}")
    if pair_task:
        print(f"  s1 paraphrased:   {stats.get('successful_s1', 0)}")
        print(f"  s2 paraphrased:   {stats.get('successful_s2', 0)}")
        print(f"  both paraphrased: {stats.get('successful_s1s2', 0)}")
    else:
        print(f"  Paraphrased:      {stats.get('successful_s1', 0)}")
    print(f"  Total saved:      {stats['total_successful']}")
    print(f"  Discarded:        {stats['discarded']}")
    print(f"  Success rate:     {stats['success_rate']*100:.1f}%")
    print(f"  Saved to:         {args.output_path}")


if __name__ == "__main__":
    main()
