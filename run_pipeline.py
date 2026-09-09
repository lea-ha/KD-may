#!/usr/bin/env python3
"""
Unified GLUE Knowledge Distillation Pipeline.

Orchestrates three stages per task:
  1. Vanilla KD                  -> vanilla-kd.py
  2. Error-targeted augmentation -> generate-aug.py
  3. Mismatch KD                 -> mismatch-kd.py  (1-3 variants)

Mismatch KD variants:
  - "1s"     : with augmented data, 1 sample per original_index
  - "3s"     : with augmented data, all variants per original_index
               (only for pair tasks; skipped for single-sentence tasks)
  - "no_aug" : no augmented data (orig intra-student alignment only)

For pair tasks (mrpc, rte, stsb, qqp): runs 1s, 3s, no_aug.
For single-sentence (cola, sst2):       runs 1s, no_aug.

Directory layout (defaults):
  output/
    vanilla_kd_<task>/best_student_<task>_<config>/     + *_vanilla_kd_results.json
    mismatch_kd_1s_<task>/best_student_<task>_<config>/ + *_align_kd_results.json
    mismatch_kd_3s_<task>/...                           (pair tasks only)
    mismatch_kd_<task>/...                              (no_aug variant; no suffix)
    logs/
      pipeline_<YYYYMMDD_HHMMSS>.log    # orchestrator output (banners, status, summary)
      vanilla_kd_<task>.log             # per-subprocess full transcript
      aug_<task>.log
      mismatch_kd_1s_<task>.log
      mismatch_kd_3s_<task>.log
      mismatch_kd_<task>.log            # no_aug variant
    pipeline_summary.json               # structured record incl. log_file paths
  data/
    augmented_<task>.json

Teacher paths are auto-constructed as: <teacher_root>/<task>
(e.g. ./models/mrpc, ./models/cola, ...)

Examples
--------
# Run all four tasks end-to-end
python run_pipeline.py --tasks mrpc rte stsb cola

# Run a single task
python run_pipeline.py --tasks mrpc

# Resume from already-trained vanilla students
python run_pipeline.py --tasks mrpc rte stsb cola --skip_vanilla

# Resume from already-generated augmentation
python run_pipeline.py --tasks mrpc --skip_vanilla --skip_aug

# Only run specific mismatch KD variants
python run_pipeline.py --tasks mrpc --only_mismatch_variants 1s no_aug
"""

import os

# Windows + Conda + PyTorch: avoid the "multiple OpenMP runtimes" crash.
# Must be set BEFORE any import that pulls in MKL / libiomp (e.g. torch, numpy).
# setdefault so the user can still override from the shell if they want.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import datetime
import glob
import json
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


# =========================================================
# Task configuration
# =========================================================
PAIR_TASKS = {"mrpc", "rte", "stsb", "qqp"}
SINGLE_TASKS = {"cola", "sst2"}
SUPPORTED_TASKS = sorted(PAIR_TASKS | SINGLE_TASKS)

VARIANTS_PAIR = ["1s", "3s", "no_aug"]
VARIANTS_SINGLE = ["1s", "no_aug"]


def is_pair_task(task: str) -> bool:
    return task in PAIR_TASKS


def task_variants(task: str) -> List[str]:
    return VARIANTS_PAIR if is_pair_task(task) else VARIANTS_SINGLE


def mismatch_kd_dir_name(task: str, variant: str) -> str:
    """Directory name for a mismatch KD variant. no_aug gets no suffix."""
    if variant == "no_aug":
        return f"mismatch_kd_{task}"
    return f"mismatch_kd_{variant}_{task}"


def mismatch_kd_log_name(task: str, variant: str) -> str:
    """Log file basename (no .log) for a mismatch KD variant."""
    return mismatch_kd_dir_name(task, variant)


# =========================================================
# Logging (tee subprocess output to terminal + per-stage log files)
# =========================================================
class Tee:
    """Duplicate writes to multiple file-like objects (e.g. terminal + master log)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


_MASTER_LOG_HANDLE = None


def setup_master_logging(output_root: str) -> str:
    """
    Open the master pipeline log and redirect sys.stdout/sys.stderr through a Tee
    so the orchestrator's own prints are captured. Subprocess output is NOT routed
    through Tee -- it goes straight to the real terminal + its own per-stage log
    (no duplication into the master log).
    """
    global _MASTER_LOG_HANDLE
    log_dir = os.path.join(output_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log_path = os.path.join(log_dir, f"pipeline_{ts}.log")
    _MASTER_LOG_HANDLE = open(master_log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, _MASTER_LOG_HANDLE)
    sys.stderr = Tee(sys.__stderr__, _MASTER_LOG_HANDLE)
    return master_log_path


def stage_log_path(output_root: str, stage_name: str) -> str:
    """Build a per-stage log file path under <output_root>/logs/."""
    log_dir = os.path.join(output_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{stage_name}.log")


# =========================================================
# Helpers
# =========================================================
def banner(text: str, char: str = "=", width: int = 96) -> None:
    print()
    print(char * width)
    print(text)
    print(char * width)


def run_subprocess(cmd: List[str], stage_desc: str, log_path: str) -> Tuple[bool, str]:
    """
    Run a subprocess, teeing stdout+stderr to both the terminal and a per-stage
    log file. Returns (success, error_msg).
    """
    print(f"\n>>> {stage_desc}")
    print(f"    cmd: {' '.join(cmd)}")
    print(f"    log: {log_path}")

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # Windows: stdout defaults to cp1252 when piped, which crashes on emoji
        # (the underlying scripts print things like 💾). Force UTF-8.
        "PYTHONIOENCODING": "utf-8",
    }

    try:
        with open(log_path, "w", buffering=1, encoding="utf-8") as logf:
            logf.write(f"# stage: {stage_desc}\n")
            logf.write(f"# cmd:   {' '.join(cmd)}\n")
            logf.write(f"# start: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
            logf.write("# " + "-" * 80 + "\n\n")
            logf.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
            )

            # Stream raw bytes so tqdm \r updates flow through cleanly.
            # Write to real terminal (sys.__stdout__) to avoid duplicating the
            # subprocess output into the master pipeline log; the per-stage log
            # already captures it.
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(1024)
                if not chunk:
                    break
                try:
                    decoded = chunk.decode("utf-8", errors="replace")
                except Exception:
                    decoded = chunk.decode("latin-1", errors="replace")
                sys.__stdout__.write(decoded)
                sys.__stdout__.flush()
                logf.write(decoded)
                logf.flush()

            proc.wait()

            logf.write(
                f"\n\n# end:   {datetime.datetime.now().isoformat(timespec='seconds')} "
                f"(exit={proc.returncode})\n"
            )

        if proc.returncode != 0:
            return False, f"exit code {proc.returncode} (see {log_path})"
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def find_vanilla_best(vanilla_output_dir: str) -> Tuple[Optional[float], Optional[str]]:
    """Find best score and save_dir from a completed vanilla KD run."""
    results_files = glob.glob(os.path.join(vanilla_output_dir, "*_vanilla_kd_results.json"))
    if not results_files:
        return None, None
    results_files.sort(key=os.path.getmtime, reverse=True)
    try:
        with open(results_files[0], "r") as f:
            data = json.load(f)
        best = data.get("best", {})
        return best.get("score"), best.get("save_dir")
    except Exception:
        return None, None


def find_mismatch_best(mismatch_output_dir: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Find best score and save_dir from a completed mismatch KD run.
    The underlying script writes *_align_kd_results.json -- keep that glob.
    """
    results_files = glob.glob(os.path.join(mismatch_output_dir, "*_align_kd_results.json"))
    if not results_files:
        return None, None
    results_files.sort(key=os.path.getmtime, reverse=True)
    try:
        with open(results_files[0], "r") as f:
            data = json.load(f)
        best = data.get("best", {})
        return best.get("score"), best.get("save_dir")
    except Exception:
        return None, None


def count_aug_samples(path: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            return len(json.load(f))
    except Exception:
        return None


# =========================================================
# Stage runners
# =========================================================
def stage_vanilla_kd(task: str, args, teacher_path: str) -> Dict[str, Any]:
    """Run vanilla KD. Returns dict with status / best info / log_file."""
    out_dir = os.path.join(args.output_root, f"vanilla_kd_{task}")
    log_path = stage_log_path(args.output_root, f"vanilla_kd_{task}")

    if args.skip_vanilla:
        score, save_dir = find_vanilla_best(out_dir)
        if save_dir and os.path.isdir(save_dir):
            print(f"  [skip_vanilla] existing best: {save_dir} (score={score})")
            return {"status": "skipped",
                    "best_score": score,
                    "best_save_dir": save_dir,
                    "output_dir": out_dir,
                    "log_file": None}
        print(f"  [skip_vanilla] no existing results in {out_dir} -- running anyway")

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, args.vanilla_script,
        "--task", task,
        "--teacher_model_path", teacher_path,
        "--student_model", args.student_model_vanilla,
        "--output_dir", out_dir,
        "--max_seq_length", str(args.max_seq_length),
        "--alpha", str(args.vanilla_alpha),
        "--temperature", str(args.vanilla_temperature),
        "--device", args.device,
    ]
    if args.verify_teacher:
        cmd.append("--verify_teacher")

    ok, err = run_subprocess(cmd, f"[Stage 1/3] Vanilla KD :: task={task}", log_path)
    if not ok:
        return {"status": "failed", "error": err, "output_dir": out_dir, "log_file": log_path}

    score, save_dir = find_vanilla_best(out_dir)
    if not save_dir or not os.path.isdir(save_dir):
        return {"status": "failed",
                "error": "could not find best save_dir after vanilla KD",
                "output_dir": out_dir,
                "log_file": log_path}

    return {"status": "ok",
            "best_score": score,
            "best_save_dir": save_dir,
            "output_dir": out_dir,
            "log_file": log_path}


def stage_generate_aug(task: str, args, teacher_path: str, student_path: str) -> Dict[str, Any]:
    """Generate augmented data. Returns dict with status / output_path / log_file."""
    out_path = os.path.join(args.data_root, f"augmented_{task}.json")
    log_path = stage_log_path(args.output_root, f"aug_{task}")

    if args.skip_aug and os.path.isfile(out_path):
        n = count_aug_samples(out_path)
        print(f"  [skip_aug] existing file: {out_path} ({n} samples)")
        return {"status": "skipped",
                "output_path": out_path,
                "n_samples": n,
                "log_file": None}

    os.makedirs(args.data_root, exist_ok=True)

    cmd = [
        sys.executable, args.aug_script,
        "--task", task,
        "--teacher_model_path", teacher_path,
        "--student_model_path", student_path,
        "--output_path", out_path,
        "--max_seq_length", str(args.max_seq_length),
        "--batch_size", str(args.aug_batch_size),
        "--num_few_shot", str(args.aug_num_few_shot),
        "--device", args.device,
    ]

    ok, err = run_subprocess(cmd, f"[Stage 2/3] Generate augmentation :: task={task}", log_path)
    if not ok:
        return {"status": "failed", "error": err, "output_path": out_path, "log_file": log_path}

    if not os.path.isfile(out_path):
        return {"status": "failed",
                "error": "aug script ran but output file is missing",
                "output_path": out_path,
                "log_file": log_path}

    n = count_aug_samples(out_path)
    if n is None:
        return {"status": "failed",
                "error": "could not parse augmented JSON",
                "output_path": out_path,
                "log_file": log_path}

    return {"status": "ok",
            "output_path": out_path,
            "n_samples": n,
            "log_file": log_path}


def stage_mismatch_kd(
    task: str,
    args,
    teacher_path: str,
    aug_path: Optional[str],
    variant: str,
) -> Dict[str, Any]:
    """Run mismatch KD for one variant."""
    out_dir = os.path.join(args.output_root, mismatch_kd_dir_name(task, variant))
    log_path = stage_log_path(args.output_root, mismatch_kd_log_name(task, variant))

    if args.skip_mismatch:
        score, save_dir = find_mismatch_best(out_dir)
        if save_dir and os.path.isdir(save_dir):
            print(f"  [skip_mismatch] {variant}: existing best={save_dir} score={score}")
            return {"status": "skipped",
                    "best_score": score,
                    "best_save_dir": save_dir,
                    "output_dir": out_dir,
                    "log_file": None}
        return {"status": "skipped_empty", "output_dir": out_dir, "log_file": None}

    if variant in ("1s", "3s"):
        if not aug_path or not os.path.isfile(aug_path):
            return {"status": "failed",
                    "error": f"variant {variant} needs aug data, but {aug_path} is missing",
                    "output_dir": out_dir,
                    "log_file": None}

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, args.mismatch_script,
        "--task", task,
        "--teacher_model_path", teacher_path,
        "--student_model", args.student_model_mismatch,
        "--output_dir", out_dir,
        "--max_seq_length", str(args.max_seq_length),
        "--alpha", str(args.mismatch_alpha),
        "--temperature", str(args.mismatch_temperature),
        "--lambda_align", str(args.mismatch_lambda),
        "--cos_margin", str(args.mismatch_cos_margin),
        "--device", args.device,
    ]
    if args.verify_teacher:
        cmd.append("--verify_teacher")

    if variant == "1s":
        cmd.extend(["--aug_data_path", aug_path])
    elif variant == "3s":
        cmd.extend(["--aug_data_path", aug_path, "--use_all_variants"])
    # variant == "no_aug": no --aug_data_path flag

    ok, err = run_subprocess(
        cmd, f"[Stage 3/3] Mismatch KD :: task={task} variant={variant}", log_path
    )
    if not ok:
        return {"status": "failed", "error": err, "output_dir": out_dir, "log_file": log_path}

    score, save_dir = find_mismatch_best(out_dir)
    return {"status": "ok",
            "best_score": score,
            "best_save_dir": save_dir,
            "output_dir": out_dir,
            "log_file": log_path}


# =========================================================
# Per-task pipeline
# =========================================================
def run_task_pipeline(task: str, args, summary: Dict[str, Any]) -> None:
    banner(f"  TASK: {task.upper()}  ", char="#")

    teacher_path = os.path.join(args.teacher_root, task)
    if not os.path.isdir(teacher_path):
        msg = f"teacher path not found: {teacher_path}"
        print(f"  [X] {msg}")
        summary["results"][task] = {"setup_error": msg}
        summary["failures"].append({"task": task, "stage": "setup", "error": msg, "log_file": None})
        return

    print(f"  Teacher:            {teacher_path}")
    print(f"  Student (vanilla):  {args.student_model_vanilla}")
    print(f"  Student (mismatch): {args.student_model_mismatch}")
    variants_for_task = task_variants(task)
    print(f"  Mismatch variants:  {variants_for_task}")

    task_result: Dict[str, Any] = {
        "teacher_path": teacher_path,
        "is_pair_task": is_pair_task(task),
    }
    summary["results"][task] = task_result

    # ----- Stage 1: Vanilla KD -----
    banner(f"  [{task}] Stage 1: Vanilla KD", char="-")
    vanilla = stage_vanilla_kd(task, args, teacher_path)
    task_result["vanilla_kd"] = vanilla

    if vanilla["status"] == "failed":
        err = vanilla.get("error", "unknown")
        print(f"  [X] Vanilla KD failed: {err}")
        summary["failures"].append({
            "task": task, "stage": "vanilla_kd",
            "error": err, "log_file": vanilla.get("log_file"),
        })
        return

    vanilla_best_student = vanilla.get("best_save_dir")
    print(f"  [ok] vanilla best student: {vanilla_best_student}")

    # ----- Decide which variants to run -----
    variants = list(variants_for_task)
    if args.only_mismatch_variants:
        variants = [v for v in variants if v in args.only_mismatch_variants]
        if not variants:
            print(f"  [info] no mismatch variants requested for {task}; done.")
            return

    # If --skip_mismatch, just record what's on disk and stop
    if args.skip_mismatch:
        task_result["mismatch_kd"] = {}
        for v in variants:
            task_result["mismatch_kd"][v] = stage_mismatch_kd(task, args, teacher_path, None, v)
        return

    needs_aug = any(v in ("1s", "3s") for v in variants)

    # ----- Stage 2: Generate augmentation -----
    aug_path: Optional[str] = None
    if needs_aug:
        banner(f"  [{task}] Stage 2: Generate augmentation", char="-")
        aug = stage_generate_aug(task, args, teacher_path, vanilla_best_student)
        task_result["augmented"] = aug
        if aug["status"] == "failed":
            err = aug.get("error", "unknown")
            print(f"  [X] Aug failed: {err}")
            summary["failures"].append({
                "task": task, "stage": "generate_aug",
                "error": err, "log_file": aug.get("log_file"),
            })
            # Drop variants that need aug; keep only no_aug
            variants = [v for v in variants if v == "no_aug"]
            if not variants:
                return
        else:
            aug_path = aug["output_path"]
            print(f"  [ok] aug data: {aug_path} ({aug.get('n_samples', '?')} samples)")
    else:
        task_result["augmented"] = {"status": "not_needed"}

    # ----- Stage 3: Mismatch KD variants -----
    task_result["mismatch_kd"] = {}
    for variant in variants:
        banner(f"  [{task}] Stage 3: Mismatch KD ({variant})", char="-")
        result = stage_mismatch_kd(task, args, teacher_path, aug_path, variant)
        task_result["mismatch_kd"][variant] = result
        if result["status"] == "failed":
            err = result.get("error", "unknown")
            print(f"  [X] Mismatch KD ({variant}) failed: {err}")
            summary["failures"].append({
                "task": task, "stage": f"mismatch_kd_{variant}",
                "error": err, "log_file": result.get("log_file"),
            })
        else:
            score = result.get("best_score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"  [ok] Mismatch KD ({variant}) best score: {score_str}")


# =========================================================
# Args
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Unified GLUE KD pipeline orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Tasks + paths
    p.add_argument("--tasks", nargs="+", required=True, choices=SUPPORTED_TASKS,
                   help="One or more GLUE tasks to run.")
    p.add_argument("--teacher_root", default="./models",
                   help="Teacher path is auto-constructed as <teacher_root>/<task>.")
    p.add_argument("--output_root", default="./output")
    p.add_argument("--data_root", default="./data")

    # Students
    p.add_argument("--student_model_vanilla", default="huawei-noah/TinyBERT_General_6L_768D")
    p.add_argument("--student_model_mismatch", default="huawei-noah/TinyBERT_General_6L_768D")

    # Skip / resume flags
    p.add_argument("--skip_vanilla", action="store_true",
                   help="Skip vanilla KD if results already exist for the task.")
    p.add_argument("--skip_aug", action="store_true",
                   help="Skip aug generation if augmented_<task>.json already exists.")
    p.add_argument("--skip_mismatch", action="store_true",
                   help="Skip mismatch KD entirely.")
    p.add_argument("--only_mismatch_variants", nargs="+",
                   choices=["1s", "3s", "no_aug"], default=None,
                   help="Restrict which mismatch variants to run.")

    # Shared passthroughs
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--device", default=None,
                   help="cuda/cpu. Auto-detected if not set.")
    p.add_argument("--verify_teacher", action="store_true")

    # Vanilla KD passthroughs
    p.add_argument("--vanilla_alpha", type=float, default=0.5)
    p.add_argument("--vanilla_temperature", type=float, default=3.0)

    # Mismatch KD passthroughs
    p.add_argument("--mismatch_alpha", type=float, default=0.5)
    p.add_argument("--mismatch_temperature", type=float, default=3.0)
    p.add_argument("--mismatch_lambda", type=float, default=0.5,
                   help="Weight on alignment loss (forwarded as --lambda_align).")
    p.add_argument("--mismatch_cos_margin", type=float, default=0.2)

    # Aug passthroughs
    p.add_argument("--aug_batch_size", type=int, default=16)
    p.add_argument("--aug_num_few_shot", type=int, default=15)

    # Script paths (in case they're not in cwd)
    p.add_argument("--vanilla_script", default="vanilla-kd.py")
    p.add_argument("--aug_script", default="generate-aug.py")
    p.add_argument("--mismatch_script", default="mismatch-kd.py")

    return p.parse_args()


# =========================================================
# Main
# =========================================================
def fmt_score(s: Any) -> str:
    if isinstance(s, (int, float)):
        return f"{s:.4f}"
    return "-"


def print_final_table(args, summary: Dict[str, Any]) -> None:
    print("\n" + "-" * 116)
    header = (
        f"{'Task':<8} {'Vanilla':<18} {'Aug':<16} "
        f"{'Mismatch 1s':<14} {'Mismatch 3s':<14} {'Mismatch no_aug':<16}"
    )
    print(header)
    print("-" * 116)

    for task in args.tasks:
        r = summary["results"].get(task, {})
        if "setup_error" in r:
            print(f"{task:<8} SETUP-ERROR: {r['setup_error']}")
            continue

        v = r.get("vanilla_kd", {})
        a = r.get("augmented", {})
        mk = r.get("mismatch_kd", {})

        v_status = v.get("status", "-")
        if v_status in ("ok", "skipped"):
            v_str = f"{v_status} ({fmt_score(v.get('best_score'))})"
        else:
            v_str = v_status

        a_status = a.get("status", "-")
        if a_status in ("ok", "skipped"):
            a_str = f"{a_status} ({a.get('n_samples', '?')})"
        else:
            a_str = a_status

        def fmt_variant(variant: str) -> str:
            res = mk.get(variant)
            if res is None:
                return "-"
            status = res.get("status", "-")
            if status in ("ok", "skipped"):
                return fmt_score(res.get("best_score"))
            return status.upper()[:12]

        print(
            f"{task:<8} {v_str:<18} {a_str:<16} "
            f"{fmt_variant('1s'):<14} {fmt_variant('3s'):<14} {fmt_variant('no_aug'):<16}"
        )
    print("-" * 116)


def main():
    args = parse_args()

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    # Set up master log BEFORE any prints we want captured
    master_log_path = setup_master_logging(args.output_root)
    log_dir = os.path.join(args.output_root, "logs")

    # Sanity check script paths
    missing = []
    for label, path in [
        ("vanilla", args.vanilla_script),
        ("aug", args.aug_script),
        ("mismatch", args.mismatch_script),
    ]:
        if not os.path.isfile(path):
            missing.append((label, path))
    if missing:
        print("WARNING: the following scripts were not found at the given paths:")
        for label, path in missing:
            print(f"  {label}: {path}")
        print("  -> the corresponding stages will fail unless paths are corrected.")

    banner("UNIFIED GLUE KD PIPELINE")
    print(f"Tasks:                  {args.tasks}")
    print(f"Teacher root:           {args.teacher_root}")
    print(f"Output root:            {args.output_root}")
    print(f"Data root:              {args.data_root}")
    print(f"Log dir:                {log_dir}")
    print(f"Master log:             {master_log_path}")
    print(f"Vanilla student:        {args.student_model_vanilla}")
    print(f"Mismatch student:       {args.student_model_mismatch}")
    print(f"Device:                 {args.device}")
    print(f"Skip vanilla:           {args.skip_vanilla}")
    print(f"Skip aug:               {args.skip_aug}")
    print(f"Skip mismatch:          {args.skip_mismatch}")
    if args.only_mismatch_variants:
        print(f"Only mismatch variants: {args.only_mismatch_variants}")

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    summary: Dict[str, Any] = {
        "started_at": started_at,
        "tasks": args.tasks,
        "args": vars(args),
        "master_log": master_log_path,
        "log_dir": log_dir,
        "results": {},
        "failures": [],
    }

    for task in args.tasks:
        try:
            run_task_pipeline(task, args, summary)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[X] Unhandled exception in task {task}:\n{tb}")
            summary["failures"].append({
                "task": task,
                "stage": "unhandled",
                "error": f"{type(e).__name__}: {e}",
                "log_file": None,
            })

    summary["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    # Save summary JSON
    summary_path = os.path.join(args.output_root, "pipeline_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"WARNING: could not write summary file: {e}")

    # Final report
    banner("PIPELINE COMPLETE")
    print(f"Started:  {summary['started_at']}")
    print(f"Finished: {summary['finished_at']}")
    print(f"Summary:  {summary_path}")
    print(f"Logs:     {log_dir}")

    print_final_table(args, summary)

    if summary["failures"]:
        banner("FAILURES", char="!")
        for f in summary["failures"]:
            log_hint = f"  (log: {f['log_file']})" if f.get("log_file") else ""
            print(f"  [{f['task']}] {f['stage']}: {f['error']}{log_hint}")
        print(f"\nTotal failures: {len(summary['failures'])}")
        sys.exit(1)
    else:
        print("\n[ok] all stages completed without failure.")


if __name__ == "__main__":
    main()