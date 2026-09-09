#!/usr/bin/env python3
"""
Alignment-enhanced Knowledge Distillation for GLUE (teacher-student).

Extends vanilla KD with:
- Augmented training data (paraphrased variants of hard examples)
- Per-sentence cosine-embedding alignment loss between student and teacher
- Learned projection layer to bridge different representation spaces
  (e.g. DeBERTa teacher + TinyBERT student)
- Synthetic token_type_ids reconstruction for models that don't produce them
- Per-config overrides for alpha / temperature / lambda_align / cos_margin
  (CLI flags remain as fallback)

Loss:
    total_loss = kd_loss + lambda_align * align_loss

    kd_loss = alpha * KL(student/T || teacher/T) * T^2 + (1-alpha) * CE(student, labels)

    align_loss (for pair tasks, averaged across active terms):
        orig:  CosEmb(s_s1, s_s2, y_pair)              # intra-student, no projection
        s1:    CosEmb(proj(s_s1), t_s1, +1)             # cross-model, projected
        s2:    CosEmb(proj(s_s2), t_s2, +1)             # cross-model, projected
        s1s2:  avg(CosEmb(proj(s_s1), t_s1, +1),
                   CosEmb(proj(s_s2), t_s2, +1))        # both sides

    align_loss (for single-sentence tasks):
        orig:  no alignment term
        aug:   CosEmb(proj(s_emb), t_emb, +1)

    Regression (STS-B) orig alignment uses continuous y derived from label.

Examples
--------
python run_glue_align_kd.py \\
  --task mrpc \\
  --teacher_model_path ./teacher_mrpc \\
  --student_model huawei-noah/TinyBERT_General_6L_768D \\
  --aug_data_path ./data/mrpc/augmented.json \\
  --output_dir ./runs/mrpc_align_kd

python run_glue_align_kd.py \\
  --task mrpc \\
  --teacher_model_path ./teacher_mrpc \\
  --student_model huawei-noah/TinyBERT_General_6L_768D \\
  --aug_data_path ./data/mrpc/augmented.json \\
  --no_search --learning_rate 2e-5 --batch_size 16 --epochs 10 \\
  --alpha 0.5 --temperature 3.0 --lambda_align 0.5 --cos_margin 0.2
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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset

from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, Dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
    set_seed,
)

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


DEFAULT_CONFIGS = [
    # dict(name="config_1p", learning_rate=2e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42, lambda_align=0.5),
    # dict(name="config_2p", learning_rate=3e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42, lambda_align=0.5),
    # dict(name="config_3p", learning_rate=2e-5, batch_size=32, epochs=10, warmup_ratio=0.1,  seed=42, lambda_align=0.5),
    # dict(name="config_4p", learning_rate=3e-5, batch_size=32, epochs=10, warmup_ratio=0.10, seed=42, lambda_align=0.5),
    # dict(name="config_5p",  learning_rate=2e-5, batch_size=8,  epochs=10, warmup_ratio=0.10, seed=42, alpha=0.5, temperature=3, lambda_align=0.5, cos_margin=0.2),
    # dict(name="config_6p",  learning_rate=3e-5, batch_size=8,  epochs=10, warmup_ratio=0.10, seed=42, alpha=0.5, temperature=3, lambda_align=0.5, cos_margin=0.2),
    # dict(name="config_7p",  learning_rate=2e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42, alpha=0.5, temperature=3, lambda_align=1.0, cos_margin=0.2),
    dict(name="config_8p",  learning_rate=3e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42, alpha=0.5, temperature=3, lambda_align=1.0, cos_margin=0.2),
    dict(name="config_9p",  learning_rate=2e-5, batch_size=32, epochs=10, warmup_ratio=0.10, seed=42, alpha=0.5, temperature=3, lambda_align=1.0, cos_margin=0.2),
    dict(name="config_10p", learning_rate=3e-5, batch_size=32, epochs=10, warmup_ratio=0.10, seed=42, alpha=0.5, temperature=3, lambda_align=1.0, cos_margin=0.2),
    dict(name="config_11p", learning_rate=2e-5, batch_size=8,  epochs=10, warmup_ratio=0.10, seed=42, alpha=0.5, temperature=3, lambda_align=1.0, cos_margin=0.2),
    dict(name="config_12p", learning_rate=3e-5, batch_size=8,  epochs=10, warmup_ratio=0.10, seed=42, alpha=0.5, temperature=3, lambda_align=1.0, cos_margin=0.2),
    dict(name="config_13p", learning_rate=2e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42, alpha=0.7, temperature=3, lambda_align=0.5, cos_margin=0.2),
    dict(name="config_14p", learning_rate=2e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42, alpha=0.7, temperature=3, lambda_align=1.0, cos_margin=0.2),
    dict(name="config_15p", learning_rate=2e-5, batch_size=16, epochs=10, warmup_ratio=0.10, seed=42, alpha=0.5, temperature=5, lambda_align=0.5, cos_margin=0.2),
    dict(name="config_16p", learning_rate=5e-5, batch_size=16, epochs=10, warmup_ratio=0.06, seed=42, alpha=0.5, temperature=3, lambda_align=0.5, cos_margin=0.2),
]
 

# =========================================================
# Utilities
# =========================================================
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_regression_task(task: str) -> bool:
    return task.lower() == "stsb"


def is_pair_task(task: str) -> bool:
    _, k2 = TASK_TO_KEYS[task]
    return k2 is not None


def build_model(model_name_or_path: str, task: str):
    if is_regression_task(task):
        return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=3 if task == "mnli" else 2,
    )


def get_hidden_size(model_name_or_path: str) -> int:
    config = AutoConfig.from_pretrained(model_name_or_path)
    return config.hidden_size


def load_tokenizer_with_fallback(model_name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    except Exception as e:
        print(f"Fast tokenizer failed for {model_name_or_path}: {e}")
        print("Falling back to slow tokenizer.")
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)


def get_encoder(model):
    """Extract the transformer encoder backbone from an AutoModelForSequenceClassification."""
    for attr in ["bert", "deberta", "deberta_v2", "roberta", "distilbert",
                 "electra", "albert", "xlnet", "xlm_roberta", "camembert"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError(
        f"Cannot find encoder backbone in {type(model).__name__}. "
        f"Known attributes: bert, deberta, roberta, distilbert, electra, albert, xlnet"
    )


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


def build_dataloader(dataset, batch_size: int, shuffle: bool, collate_fn=None,
                     generator=None) -> DataLoader:
    kwargs = dict(batch_size=batch_size, shuffle=shuffle)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if generator is not None:
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


def extract_inputs(batch: Dict[str, torch.Tensor], device: str,
                   prefix: Optional[str] = None) -> Dict[str, torch.Tensor]:
    if prefix is None:
        keys = ["input_ids", "attention_mask", "token_type_ids"]
        return {k: batch[k].to(device) for k in keys if k in batch}

    out = {}
    for base_key in ["input_ids", "attention_mask", "token_type_ids"]:
        pref_key = f"{prefix}_{base_key}"
        if pref_key in batch:
            out[base_key] = batch[pref_key].to(device)
    return out


# =========================================================
# Synthetic token_type_ids for models that don't produce them
# =========================================================
def build_synthetic_token_type_ids(input_ids: torch.Tensor, sep_token_id: int) -> torch.Tensor:
    """
    Reconstruct token_type_ids from SEP token positions.
    Convention: [CLS] s1_tokens [SEP] s2_tokens [SEP] [PAD...]
    Everything up to and including first SEP -> 0, rest -> 1.
    For single-sentence inputs (one SEP), everything is 0.
    """
    batch_size, seq_len = input_ids.shape
    token_type_ids = torch.zeros_like(input_ids)

    for i in range(batch_size):
        sep_positions = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) >= 2:
            # Pair input: first SEP ends segment A
            first_sep_pos = sep_positions[0].item()
            token_type_ids[i, first_sep_pos + 1:] = 1
        # Single sentence or no SEP: all zeros (already initialized)

    return token_type_ids


# =========================================================
# Embedding extraction
# =========================================================
def masked_mean_pool(token_embeddings: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
    mask = mask_2d.unsqueeze(-1).type_as(token_embeddings)
    summed = (token_embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def get_sentence_embeddings_pair(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract L2-normalized mean-pooled embeddings for s1 and s2 from a paired input."""
    base_keep = attention_mask.bool() & ~special_tokens_mask.bool()
    s1_emb = F.normalize(masked_mean_pool(last_hidden_state, base_keep & (token_type_ids == 0)), p=2, dim=-1)
    s2_emb = F.normalize(masked_mean_pool(last_hidden_state, base_keep & (token_type_ids == 1)), p=2, dim=-1)
    return s1_emb, s2_emb


def get_sentence_embedding_single(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract L2-normalized mean-pooled embedding for a single-sentence input."""
    base_keep = attention_mask.bool() & ~special_tokens_mask.bool()
    return F.normalize(masked_mean_pool(last_hidden_state, base_keep), p=2, dim=-1)


# =========================================================
# Augmented data loading
# =========================================================
def load_augmented_data(path: str) -> List[Dict[str, Any]]:
    """Load augmented data JSON. Returns list of augmentation records."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        list_candidates = [v for v in data.values() if isinstance(v, list)]
        if not list_candidates:
            raise ValueError(f"JSON at {path} is a dict but contains no list of examples.")
        data = list_candidates[0]
    if not isinstance(data, list):
        raise ValueError(f"Expected list of records in {path}, got {type(data)}")
    print(f"  Loaded {len(data)} augmented records from {path}")
    return data


def subsample_augmented(aug_data: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """
    Select exactly 1 augmented row per original_index using round-robin
    across paraphrased_side values for balanced representation.
    """
    by_origin = defaultdict(lambda: defaultdict(list))
    for r in aug_data:
        oi = int(r["original_index"])
        side = str(r.get("paraphrased_side", "unk")).lower().strip()
        by_origin[oi][side].append(r)

    rng = random.Random(seed)
    sides_cycle = ["s1", "s2", "s1s2"]
    selected = []

    for cycle_idx, oi in enumerate(sorted(by_origin.keys())):
        preferred_side = sides_cycle[cycle_idx % len(sides_cycle)]
        side_pool = by_origin[oi]
        chosen_side = preferred_side if preferred_side in side_pool else rng.choice(list(side_pool.keys()))
        selected.append(rng.choice(side_pool[chosen_side]))

    return selected


def build_train_rows(
    base_train_dataset,
    aug_data: List[Dict[str, Any]],
    task: str,
) -> List[Dict[str, Any]]:
    """
    Combine original training data with augmented rows into a unified format.

    Each row has:
        student_s1, student_s2 (or just student_s1 for single-sentence)
        teacher_s1, teacher_s2 (or just teacher_s1 for single-sentence)
        label, variant
    """
    k1, k2 = TASK_TO_KEYS[task]
    rows = []

    # Original examples
    for i in range(len(base_train_dataset)):
        ex = base_train_dataset[i]
        row = {
            "student_s1": ex[k1],
            "teacher_s1": ex[k1],
            "label": ex["label"],
            "variant": "orig",
        }
        if k2 is not None:
            row["student_s2"] = ex[k2]
            row["teacher_s2"] = ex[k2]
        rows.append(row)

    # Augmented examples
    for r in aug_data:
        side = str(r.get("paraphrased_side", "s1")).lower().strip()

        if k2 is not None:
            # Pair task
            row = {
                "student_s1": r["text1"],
                "student_s2": r["text2"],
                "teacher_s1": r["original_text1"],
                "teacher_s2": r["original_text2"],
                "label": r["label"],
                "variant": side,
            }
        else:
            # Single-sentence task
            row = {
                "student_s1": r["text1"],
                "teacher_s1": r.get("original_text1", r["text1"]),
                "label": r["label"],
                "variant": "aug",  # Normalize all single-sentence augmentations
            }
        rows.append(row)

    return rows


# =========================================================
# Dataset for alignment KD
# =========================================================
class AlignKDDataset(TorchDataset):
    """
    Pre-tokenized dataset where each item has dual-tokenized inputs
    (student and teacher), along with label and variant info.
    """
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        student_tokenizer,
        teacher_tokenizer,
        task: str,
        max_length: int,
    ):
        self.items = []
        teacher_sep_id = teacher_tokenizer.sep_token_id
        teacher_produces_ttids = self._tokenizer_produces_token_type_ids(teacher_tokenizer, max_length)
        pair_task = is_pair_task(task)

        for r in rows:
            # Student encoding
            if pair_task:
                s_enc = student_tokenizer(
                    r["student_s1"], r["student_s2"],
                    padding="max_length", truncation=True,
                    max_length=max_length, return_special_tokens_mask=True,
                )
            else:
                s_enc = student_tokenizer(
                    r["student_s1"],
                    padding="max_length", truncation=True,
                    max_length=max_length, return_special_tokens_mask=True,
                )

            # Teacher encoding
            if pair_task:
                t_enc = teacher_tokenizer(
                    r["teacher_s1"], r["teacher_s2"],
                    padding="max_length", truncation=True,
                    max_length=max_length, return_special_tokens_mask=True,
                )
            else:
                t_enc = teacher_tokenizer(
                    r["teacher_s1"],
                    padding="max_length", truncation=True,
                    max_length=max_length, return_special_tokens_mask=True,
                )

            item = {
                "s_input_ids": torch.tensor(s_enc["input_ids"]),
                "s_attention_mask": torch.tensor(s_enc["attention_mask"]),
                "s_special_tokens_mask": torch.tensor(s_enc.get("special_tokens_mask",
                                                                [0] * len(s_enc["input_ids"]))),
                "t_input_ids": torch.tensor(t_enc["input_ids"]),
                "t_attention_mask": torch.tensor(t_enc["attention_mask"]),
                "t_special_tokens_mask": torch.tensor(t_enc.get("special_tokens_mask",
                                                                [0] * len(t_enc["input_ids"]))),
                "label": torch.tensor(r["label"], dtype=torch.float if is_regression_task(task) else torch.long),
                "variant": r["variant"],
            }

            # Student token_type_ids (BERT/TinyBERT produce these)
            if "token_type_ids" in s_enc:
                item["s_token_type_ids"] = torch.tensor(s_enc["token_type_ids"])

            # Teacher token_type_ids: use native if available, else reconstruct
            if "token_type_ids" in t_enc and teacher_produces_ttids:
                item["t_token_type_ids"] = torch.tensor(t_enc["token_type_ids"])
            elif pair_task and teacher_sep_id is not None:
                # Reconstruct synthetic token_type_ids
                t_ids = torch.tensor(t_enc["input_ids"])
                item["t_token_type_ids"] = build_synthetic_token_type_ids(
                    t_ids.unsqueeze(0), teacher_sep_id
                ).squeeze(0)
            # For single-sentence, token_type_ids not needed for embedding extraction

            self.items.append(item)

    @staticmethod
    def _tokenizer_produces_token_type_ids(tokenizer, max_length: int) -> bool:
        """Check if tokenizer actually produces meaningful (non-all-zero) token_type_ids for pair inputs."""
        try:
            test_enc = tokenizer("test sentence one", "test sentence two",
                                 padding="max_length", truncation=True, max_length=max_length)
            ttids = test_enc.get("token_type_ids", None)
            if ttids is None:
                return False
            return any(t != 0 for t in ttids)
        except Exception:
            return False

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def align_kd_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate that handles the variant string field."""
    batch = [{**b} for b in batch]
    variants = [b.pop("variant") for b in batch]

    # Collect all tensor keys present in every item
    tensor_keys = [k for k in batch[0].keys() if isinstance(batch[0][k], torch.Tensor)]
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out["variant"] = variants
    return out


# =========================================================
# Standard eval tokenization (same as vanilla KD)
# =========================================================
def tokenize_dataset_for_eval(dataset, tokenizer, task: str, max_length: int):
    k1, k2 = TASK_TO_KEYS[task]

    def preprocess(examples):
        if k2 is None:
            return tokenizer(examples[k1], padding="max_length", truncation=True, max_length=max_length)
        return tokenizer(examples[k1], examples[k2], padding="max_length", truncation=True, max_length=max_length)

    tokenized = dataset.map(preprocess, batched=True)
    cols = [c for c in ["input_ids", "attention_mask", "token_type_ids", "label"] if c in tokenized.column_names]
    tokenized.set_format(type="torch", columns=cols)
    return tokenized


# =========================================================
# Raw dataset loading
# =========================================================
def prepare_raw_dataset(task: str):
    dataset = load_dataset("glue", task)

    if "train" not in dataset:
        raise ValueError(f"Task '{task}' does not provide a train split.")

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


# =========================================================
# Evaluation
# =========================================================
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


# =========================================================
# KD Loss
# =========================================================
def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    task: str,
) -> torch.Tensor:
    if is_regression_task(task):
        mse = nn.MSELoss()
        hard_loss = mse(student_logits.squeeze(-1), labels.float())
        kd_loss = mse(student_logits.squeeze(-1), teacher_logits.squeeze(-1).detach())
        return (1.0 - alpha) * hard_loss + alpha * kd_loss

    T = float(temperature)
    kl = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)
    ce = F.cross_entropy(student_logits, labels.long())
    return alpha * kl + (1.0 - alpha) * ce


# =========================================================
# Alignment Loss computation
# =========================================================
def compute_alignment_loss(
    s_s1_emb: torch.Tensor,
    s_s2_emb: Optional[torch.Tensor],
    t_s1_emb: torch.Tensor,
    t_s2_emb: Optional[torch.Tensor],
    projection: nn.Module,
    align_criterion: nn.CosineEmbeddingLoss,
    variants: List[str],
    labels: torch.Tensor,
    task: str,
    device: str,
) -> torch.Tensor:
    """
    Compute alignment loss based on sample variants.

    For pair tasks:
        orig:  CosEmb(s_s1, s_s2, y_pair)          - intra-student, no projection
        s1:    CosEmb(proj(s_s1), t_s1, +1)         - cross-model s1
        s2:    CosEmb(proj(s_s2), t_s2, +1)         - cross-model s2
        s1s2:  avg of s1 and s2 cross-model terms

    For single-sentence tasks:
        aug:   CosEmb(proj(s_emb), t_emb, +1)       - cross-model
        orig:  no alignment
    """
    align_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    if is_pair_task(task):
        # Build masks
        orig_mask = torch.tensor([v == "orig" for v in variants], device=device)
        s1_mask = torch.tensor([v == "s1" for v in variants], device=device)
        s2_mask = torch.tensor([v == "s2" for v in variants], device=device)
        s1s2_mask = torch.tensor([v == "s1s2" for v in variants], device=device)

        # y_pair: +1 for paraphrase, -1 for non-paraphrase (classification)
        # For STS-B regression: scale label from [0,5] to [-1,+1]
        if is_regression_task(task):
            y_pair = (labels.float() / 2.5 - 1.0).clamp(-1.0, 1.0)
        else:
            y_pair = torch.where(labels == 1,
                                 torch.ones_like(labels, dtype=torch.float),
                                 -torch.ones_like(labels, dtype=torch.float))

        # y_aug: always +1 (same-sentence alignment, hardcoded)
        y_aug = torch.ones(labels.size(0), device=device)

        # orig: intra-student alignment (no projection needed)
        if orig_mask.any() and s_s2_emb is not None:
            align_loss = align_loss + align_criterion(
                s_s1_emb[orig_mask], s_s2_emb[orig_mask], y_pair[orig_mask]
            )
            n_terms += 1

        # s1: cross-model alignment on sentence1
        if s1_mask.any():
            align_loss = align_loss + align_criterion(
                projection(s_s1_emb[s1_mask]), t_s1_emb[s1_mask], y_aug[s1_mask]
            )
            n_terms += 1

        # s2: cross-model alignment on sentence2
        if s2_mask.any() and s_s2_emb is not None and t_s2_emb is not None:
            align_loss = align_loss + align_criterion(
                projection(s_s2_emb[s2_mask]), t_s2_emb[s2_mask], y_aug[s2_mask]
            )
            n_terms += 1

        # s1s2: cross-model alignment on both sides
        if s1s2_mask.any():
            cosemb_s1 = align_criterion(
                projection(s_s1_emb[s1s2_mask]), t_s1_emb[s1s2_mask], y_aug[s1s2_mask]
            )
            cosemb_s2_val = torch.tensor(0.0, device=device)
            if s_s2_emb is not None and t_s2_emb is not None:
                cosemb_s2_val = align_criterion(
                    projection(s_s2_emb[s1s2_mask]), t_s2_emb[s1s2_mask], y_aug[s1s2_mask]
                )
            align_loss = align_loss + (cosemb_s1 + cosemb_s2_val) / 2.0
            n_terms += 1

    else:
        # Single-sentence task
        aug_mask = torch.tensor([v != "orig" for v in variants], device=device)
        y_aug = torch.ones(labels.size(0), device=device)

        if aug_mask.any():
            align_loss = align_loss + align_criterion(
                projection(s_s1_emb[aug_mask]), t_s1_emb[aug_mask], y_aug[aug_mask]
            )
            n_terms += 1

    if n_terms > 1:
        align_loss = align_loss / n_terms

    return align_loss


# =========================================================
# Saving
# =========================================================
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


# =========================================================
# Training loop (non-MNLI)
# =========================================================
def train_one_config(
    config: Dict[str, Any],
    args,
    train_dataset: AlignKDDataset,
    val_dataset,
    teacher,
    teacher_encoder,
    student_tokenizer,
    teacher_tokenizer,
):
    set_seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # Per-config overrides (fall back to CLI args if not present in config dict)
    alpha = config.get("alpha", args.alpha)
    temperature = config.get("temperature", args.temperature)
    lambda_align = config.get("lambda_align", args.lambda_align)
    cos_margin = config.get("cos_margin", args.cos_margin)

    device = args.device
    pair_task = is_pair_task(args.task)

    student = build_model(args.student_model, args.task).to(device)
    student_encoder = get_encoder(student)

    # Projection layer: student_hidden_dim -> teacher_hidden_dim
    student_hidden = get_hidden_size(args.student_model)
    teacher_hidden = get_hidden_size(args.teacher_model_path)
    projection = nn.Linear(student_hidden, teacher_hidden).to(device)

    align_criterion = nn.CosineEmbeddingLoss(margin=cos_margin)

    # Teacher sep token id for synthetic token_type_ids
    teacher_sep_id = teacher_tokenizer.sep_token_id

    # DataLoaders
    g = torch.Generator()
    g.manual_seed(config["seed"])
    train_loader = build_dataloader(train_dataset, config["batch_size"], shuffle=True,
                                    collate_fn=align_kd_collate_fn, generator=g)
    val_loader = build_dataloader(val_dataset, config["batch_size"], shuffle=False)

    # Optimizer: student params + projection params
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
        {
            "params": projection.parameters(),
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["learning_rate"])

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n{'='*96}")
    print(f"[ALIGN KD] Student={args.student_model}  Teacher={args.teacher_model_path}")
    print(f"Task={args.task}  Config={config['name']}  Device={device}  MaxLen={args.max_seq_length}")
    print(
        f"LR={config['learning_rate']}  BS={config['batch_size']}  Epochs={config['epochs']}  "
        f"Warmup={config['warmup_ratio']}  Seed={config['seed']}"
    )
    print(f"alpha={alpha}  temperature={temperature}  "
          f"lambda_align={lambda_align}  cos_margin={cos_margin}")
    print(f"Projection: {student_hidden} -> {teacher_hidden}")
    print(f"Train samples={len(train_dataset)}  Total steps={total_steps}  Warmup steps={warmup_steps}")
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
        projection.train()
        total_loss = 0.0
        total_kd = 0.0
        total_align = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            labels = batch["label"].to(device)
            variants = batch["variant"]

            # Student inputs
            s_input_ids = batch["s_input_ids"].to(device)
            s_attention_mask = batch["s_attention_mask"].to(device)
            s_special_tokens_mask = batch["s_special_tokens_mask"].to(device)
            s_enc_inputs = {"input_ids": s_input_ids, "attention_mask": s_attention_mask}
            if "s_token_type_ids" in batch:
                s_token_type_ids = batch["s_token_type_ids"].to(device)
                s_enc_inputs["token_type_ids"] = s_token_type_ids
            else:
                s_token_type_ids = None

            # Teacher inputs
            t_input_ids = batch["t_input_ids"].to(device)
            t_attention_mask = batch["t_attention_mask"].to(device)
            t_special_tokens_mask = batch["t_special_tokens_mask"].to(device)
            t_enc_inputs = {"input_ids": t_input_ids, "attention_mask": t_attention_mask}
            if "t_token_type_ids" in batch:
                t_token_type_ids = batch["t_token_type_ids"].to(device)
                t_enc_inputs["token_type_ids"] = t_token_type_ids
            else:
                t_token_type_ids = None

            # Teacher forward (frozen)
            with torch.no_grad():
                teacher_logits = teacher(**t_enc_inputs).logits
                t_enc_out = teacher_encoder(**t_enc_inputs, return_dict=True)

                if pair_task:
                    # Need token_type_ids for splitting; use synthetic if needed
                    t_ttids_for_emb = t_token_type_ids
                    if t_ttids_for_emb is None and teacher_sep_id is not None:
                        t_ttids_for_emb = build_synthetic_token_type_ids(t_input_ids, teacher_sep_id)
                    t_s1_emb, t_s2_emb = get_sentence_embeddings_pair(
                        t_enc_out.last_hidden_state, t_attention_mask,
                        t_ttids_for_emb, t_special_tokens_mask,
                    )
                else:
                    t_s1_emb = get_sentence_embedding_single(
                        t_enc_out.last_hidden_state, t_attention_mask, t_special_tokens_mask,
                    )
                    t_s2_emb = None

            # Student forward
            student_logits = student(**s_enc_inputs).logits
            s_enc_out = student_encoder(**s_enc_inputs, return_dict=True)

            if pair_task:
                s_ttids_for_emb = s_token_type_ids
                if s_ttids_for_emb is None:
                    # Shouldn't happen for BERT/TinyBERT, but handle gracefully
                    s_ttids_for_emb = build_synthetic_token_type_ids(
                        s_input_ids, student_tokenizer.sep_token_id
                    )
                s_s1_emb, s_s2_emb = get_sentence_embeddings_pair(
                    s_enc_out.last_hidden_state, s_attention_mask,
                    s_ttids_for_emb, s_special_tokens_mask,
                )
            else:
                s_s1_emb = get_sentence_embedding_single(
                    s_enc_out.last_hidden_state, s_attention_mask, s_special_tokens_mask,
                )
                s_s2_emb = None

            # KD loss
            kd_loss = kd_loss_fn(student_logits, teacher_logits, labels,
                                 temperature, alpha, args.task)

            # Alignment loss
            align_loss = compute_alignment_loss(
                s_s1_emb=s_s1_emb,
                s_s2_emb=s_s2_emb,
                t_s1_emb=t_s1_emb,
                t_s2_emb=t_s2_emb,
                projection=projection,
                align_criterion=align_criterion,
                variants=variants,
                labels=labels,
                task=args.task,
                device=device,
            )

            loss = kd_loss + lambda_align * align_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(projection.parameters()),
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            total_kd += float(kd_loss.item())
            total_align += float(align_loss.item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                kd=f"{kd_loss.item():.4f}",
                align=f"{align_loss.item():.4f}",
            )

        n = max(1, len(train_loader))
        avg_loss = total_loss / n
        avg_kd = total_kd / n
        avg_align = total_align / n

        val_metrics = evaluate_glue(student, val_loader, device, args.task)
        score = compute_selection_score(val_metrics, args.task)

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_kd": avg_kd,
            "train_align": avg_align,
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
            f"(kd={avg_kd:.4f}, align={avg_align:.4f}) | {metrics_str} | score={score:.4f}"
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
                    "lambda_align": lambda_align,
                    "cos_margin": cos_margin,
                    "aug_data_path": args.aug_data_path,
                    "use_all_variants": args.use_all_variants,
                },
            )
            best["save_dir"] = save_dir

    best["training_history"] = training_history
    return best


# =========================================================
# Training loop (MNLI)
# =========================================================
def train_one_config_mnli(
    config: Dict[str, Any],
    args,
    train_dataset: AlignKDDataset,
    val_m_dataset,
    val_mm_dataset,
    teacher,
    teacher_encoder,
    student_tokenizer,
    teacher_tokenizer,
):
    set_seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # Per-config overrides (fall back to CLI args if not present in config dict)
    alpha = config.get("alpha", args.alpha)
    temperature = config.get("temperature", args.temperature)
    lambda_align = config.get("lambda_align", args.lambda_align)
    cos_margin = config.get("cos_margin", args.cos_margin)

    device = args.device

    student = build_model(args.student_model, "mnli").to(device)
    student_encoder = get_encoder(student)

    student_hidden = get_hidden_size(args.student_model)
    teacher_hidden = get_hidden_size(args.teacher_model_path)
    projection = nn.Linear(student_hidden, teacher_hidden).to(device)

    align_criterion = nn.CosineEmbeddingLoss(margin=cos_margin)
    teacher_sep_id = teacher_tokenizer.sep_token_id

    g = torch.Generator()
    g.manual_seed(config["seed"])
    train_loader = build_dataloader(train_dataset, config["batch_size"], shuffle=True,
                                    collate_fn=align_kd_collate_fn, generator=g)
    val_m_loader = build_dataloader(val_m_dataset, config["batch_size"], shuffle=False)
    val_mm_loader = build_dataloader(val_mm_dataset, config["batch_size"], shuffle=False)

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
        {
            "params": projection.parameters(),
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["learning_rate"])

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n{'='*96}")
    print(f"[ALIGN KD] MNLI Student={args.student_model}  Teacher={args.teacher_model_path}")
    print(
        f"Config={config['name']}  Device={device}  MaxLen={args.max_seq_length}  "
        f"LR={config['learning_rate']}  BS={config['batch_size']}  Epochs={config['epochs']}  "
        f"Warmup={config['warmup_ratio']}  Seed={config['seed']}"
    )
    print(f"alpha={alpha}  temperature={temperature}  "
          f"lambda_align={lambda_align}  cos_margin={cos_margin}")
    print(f"Projection: {student_hidden} -> {teacher_hidden}")
    print(f"Train samples={len(train_dataset)}  Total steps={total_steps}  Warmup steps={warmup_steps}")
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
        projection.train()
        total_loss = 0.0
        total_kd = 0.0
        total_align = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            labels = batch["label"].to(device)
            variants = batch["variant"]

            s_input_ids = batch["s_input_ids"].to(device)
            s_attention_mask = batch["s_attention_mask"].to(device)
            s_special_tokens_mask = batch["s_special_tokens_mask"].to(device)
            s_enc_inputs = {"input_ids": s_input_ids, "attention_mask": s_attention_mask}
            if "s_token_type_ids" in batch:
                s_token_type_ids = batch["s_token_type_ids"].to(device)
                s_enc_inputs["token_type_ids"] = s_token_type_ids
            else:
                s_token_type_ids = None

            t_input_ids = batch["t_input_ids"].to(device)
            t_attention_mask = batch["t_attention_mask"].to(device)
            t_special_tokens_mask = batch["t_special_tokens_mask"].to(device)
            t_enc_inputs = {"input_ids": t_input_ids, "attention_mask": t_attention_mask}
            if "t_token_type_ids" in batch:
                t_token_type_ids = batch["t_token_type_ids"].to(device)
                t_enc_inputs["token_type_ids"] = t_token_type_ids
            else:
                t_token_type_ids = None

            with torch.no_grad():
                teacher_logits = teacher(**t_enc_inputs).logits
                t_enc_out = teacher_encoder(**t_enc_inputs, return_dict=True)

                t_ttids_for_emb = t_token_type_ids
                if t_ttids_for_emb is None and teacher_sep_id is not None:
                    t_ttids_for_emb = build_synthetic_token_type_ids(t_input_ids, teacher_sep_id)
                t_s1_emb, t_s2_emb = get_sentence_embeddings_pair(
                    t_enc_out.last_hidden_state, t_attention_mask,
                    t_ttids_for_emb, t_special_tokens_mask,
                )

            student_logits = student(**s_enc_inputs).logits
            s_enc_out = student_encoder(**s_enc_inputs, return_dict=True)

            s_ttids_for_emb = s_token_type_ids
            if s_ttids_for_emb is None:
                s_ttids_for_emb = build_synthetic_token_type_ids(
                    s_input_ids, student_tokenizer.sep_token_id
                )
            s_s1_emb, s_s2_emb = get_sentence_embeddings_pair(
                s_enc_out.last_hidden_state, s_attention_mask,
                s_ttids_for_emb, s_special_tokens_mask,
            )

            kd_loss = kd_loss_fn(student_logits, teacher_logits, labels,
                                 temperature, alpha, "mnli")

            align_loss = compute_alignment_loss(
                s_s1_emb=s_s1_emb, s_s2_emb=s_s2_emb,
                t_s1_emb=t_s1_emb, t_s2_emb=t_s2_emb,
                projection=projection, align_criterion=align_criterion,
                variants=variants, labels=labels, task="mnli", device=device,
            )

            loss = kd_loss + lambda_align * align_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(projection.parameters()),
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            total_kd += float(kd_loss.item())
            total_align += float(align_loss.item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                kd=f"{kd_loss.item():.4f}",
                align=f"{align_loss.item():.4f}",
            )

        n = max(1, len(train_loader))
        avg_loss = total_loss / n
        avg_kd = total_kd / n
        avg_align = total_align / n

        val_metrics = evaluate_mnli_both(student, val_m_loader, val_mm_loader, device)
        score = float(val_metrics["accuracy_avg"])

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_kd": avg_kd,
            "train_align": avg_align,
            "val_metrics": val_metrics,
            "selection_score": score,
        })

        print(
            f"\nEpoch {epoch+1}: train_loss={avg_loss:.4f} "
            f"(kd={avg_kd:.4f}, align={avg_align:.4f}) | "
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
                    "lambda_align": lambda_align,
                    "cos_margin": cos_margin,
                    "aug_data_path": args.aug_data_path,
                    "use_all_variants": args.use_all_variants,
                },
            )
            best["save_dir"] = save_dir

    best["training_history"] = training_history
    return best


# =========================================================
# Args
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Alignment-enhanced KD for GLUE")
    p.add_argument("--task", type=str, required=True, choices=sorted(TASK_TO_KEYS.keys()))

    p.add_argument("--teacher_model_path", type=str, required=True,
                   help="Path to a fine-tuned teacher checkpoint.")
    p.add_argument("--student_model", type=str, default="huawei-noah/TinyBERT_General_6L_768D",
                   help="Student model name or path (initialized from scratch, not from a KD checkpoint).")

    p.add_argument("--output_dir", type=str, default="./align_kd")

    # KD params (per-config overridable via dict keys 'alpha', 'temperature')
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Default weight on KD loss vs hard labels. "
                        "Overridable per config via dict key 'alpha'.")
    p.add_argument("--temperature", type=float, default=3.0,
                   help="Default KD temperature. Overridable per config via dict key 'temperature'.")

    # Alignment params (per-config overridable via dict keys 'lambda_align', 'cos_margin')
    p.add_argument("--aug_data_path", type=str, default=None,
                   help="Path to augmented data JSON file. If not provided, runs vanilla KD only.")
    p.add_argument("--lambda_align", type=float, default=0.5,
                   help="Default weight on alignment loss. "
                        "Overridable per config via dict key 'lambda_align'.")
    p.add_argument("--cos_margin", type=float, default=0.2,
                   help="Default margin for CosineEmbeddingLoss. "
                        "Overridable per config via dict key 'cos_margin'.")
    p.add_argument("--use_all_variants", action="store_true",
                   help="Use all augmented variants. Default: subsample 1 per original_index.")

    # Verification
    p.add_argument("--verify_teacher", action="store_true",
                   help="Evaluate teacher on validation before training.")

    # General
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Single run / sweep
    p.add_argument("--no_search", action="store_true", help="Run only one config from CLI params.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    safe_mkdir(args.output_dir)

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1].")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")
    if args.lambda_align < 0:
        raise ValueError("--lambda_align must be >= 0.")

    print("=" * 96)
    print("GLUE Alignment-enhanced KD (teacher-student)")
    print(f"Task: {args.task}")
    print(f"Teacher: {args.teacher_model_path}")
    print(f"Student: {args.student_model}")
    print(f"Device: {args.device}")
    if args.aug_data_path:
        print(f"Augmented data: {args.aug_data_path}")
        print(f"Use all variants: {args.use_all_variants}")
    else:
        print("No augmented data provided — alignment will only use orig intra-student term.")
    print(f"CLI defaults: alpha={args.alpha} temperature={args.temperature} "
          f"lambda_align={args.lambda_align} cos_margin={args.cos_margin} "
          f"(per-config dict values override these)")
    print("=" * 96)

    # --- Tokenizers ---
    print("\nLoading tokenizers...")
    student_tokenizer = load_tokenizer_with_fallback(args.student_model)
    teacher_tokenizer = load_tokenizer_with_fallback(args.teacher_model_path)

    # --- Raw data ---
    print("\nLoading GLUE data...")
    raw_splits = prepare_raw_dataset(args.task)

    # --- Augmented data ---
    aug_data = []
    if args.aug_data_path:
        aug_data = load_augmented_data(args.aug_data_path)
        if not args.use_all_variants:
            aug_data = subsample_augmented(aug_data, args.seed)
            print(f"  Subsampled to {len(aug_data)} augmented rows (1 per original_index)")

    # --- Build training rows ---
    train_rows = build_train_rows(raw_splits["train"], aug_data, args.task)
    n_orig = len(raw_splits["train"])
    n_aug = len(aug_data)
    print(f"\nTotal training rows: {len(train_rows)} (orig={n_orig}, aug={n_aug})")

    # --- Build AlignKD training dataset ---
    print("Tokenizing training data (dual tokenization)...")
    train_dataset = AlignKDDataset(
        train_rows, student_tokenizer, teacher_tokenizer,
        args.task, args.max_seq_length,
    )

    # --- Validation datasets (student-tokenized for eval) ---
    print("Tokenizing validation data...")
    if args.task == "mnli":
        val_m_dataset = tokenize_dataset_for_eval(raw_splits["validation_matched"],
                                                  student_tokenizer, args.task, args.max_seq_length)
        val_mm_dataset = tokenize_dataset_for_eval(raw_splits["validation_mismatched"],
                                                   student_tokenizer, args.task, args.max_seq_length)
    else:
        val_dataset = tokenize_dataset_for_eval(raw_splits["validation"],
                                                student_tokenizer, args.task, args.max_seq_length)

    # --- Teacher model ---
    print("\nLoading teacher model...")
    teacher = build_model(args.teacher_model_path, args.task).to(args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_encoder = get_encoder(teacher)

    # --- Verify teacher ---
    if args.verify_teacher:
        print("\nVerifying teacher on validation...")
        if args.task == "mnli":
            # Tokenize val with teacher tokenizer for verification
            tv_m = tokenize_dataset_for_eval(raw_splits["validation_matched"],
                                             teacher_tokenizer, args.task, args.max_seq_length)
            tv_mm = tokenize_dataset_for_eval(raw_splits["validation_mismatched"],
                                              teacher_tokenizer, args.task, args.max_seq_length)
            t_m_loader = build_dataloader(tv_m, args.batch_size, shuffle=False)
            t_mm_loader = build_dataloader(tv_mm, args.batch_size, shuffle=False)
            teacher_metrics = evaluate_mnli_both(teacher, t_m_loader, t_mm_loader, args.device)
            print(
                f"[Teacher val] acc_matched={teacher_metrics['accuracy_matched']:.4f}, "
                f"acc_mismatched={teacher_metrics['accuracy_mismatched']:.4f}, "
                f"acc_avg={teacher_metrics['accuracy_avg']:.4f}"
            )
        else:
            tv = tokenize_dataset_for_eval(raw_splits["validation"],
                                           teacher_tokenizer, args.task, args.max_seq_length)
            t_loader = build_dataloader(tv, args.batch_size, shuffle=False)
            teacher_metrics = evaluate_glue(teacher, t_loader, args.device, args.task)
            teacher_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in teacher_metrics.items()
            )
            print(f"[Teacher val] {teacher_str}")

    # --- Config sweep or single run ---
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

    # --- Run ---
    results = []
    for cfg in configs:
        try:
            if args.task == "mnli":
                best = train_one_config_mnli(
                    cfg, args, train_dataset,
                    val_m_dataset, val_mm_dataset,
                    teacher, teacher_encoder,
                    student_tokenizer, teacher_tokenizer,
                )
            else:
                best = train_one_config(
                    cfg, args, train_dataset, val_dataset,
                    teacher, teacher_encoder,
                    student_tokenizer, teacher_tokenizer,
                )
            results.append(best)
        except Exception as e:
            import traceback
            print(f"Error with {cfg['name']}: {e}")
            traceback.print_exc()
            continue

    if not results:
        raise RuntimeError("No successful runs. Check your environment / task / teacher path.")

    # --- Results summary ---
    results_sorted = sorted(results, key=lambda r: float(r["score"]), reverse=True)
    best = results_sorted[0]

    print(f"\n{'='*96}")
    print("RESULTS SUMMARY (sorted by selection score)")
    print(f"{'='*96}")
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
        "lambda_align_cli_default": args.lambda_align,
        "cos_margin_cli_default": args.cos_margin,
        "aug_data_path": args.aug_data_path,
        "use_all_variants": args.use_all_variants,
        "search": (not args.no_search),
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
        f"{args.task}_{args.student_model.replace('/', '_')}_align_kd_results.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nBest config: {best['config']} | epoch={best['epoch']} | score={best['score']:.4f}")
    print(f"Saved run summary to: {out_path}")


if __name__ == "__main__":
    main()