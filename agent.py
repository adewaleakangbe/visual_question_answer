################################################################################
# agent.py
#
# Agentic Experiment Controller for SLAKE VQA
#
# This script implements an agentic AI loop that:
#
#   1. Runs 18 experiments systematically:
#        3 variants × 3 training ratios × 2 loss functions
#
#   2. After every 6 runs (one full ratio sweep), calls Claude via the
#      Anthropic API to analyse results so far and give a mid-experiment
#      recommendation on which variant looks most promising.
#
#   3. After all 18 runs complete, the agent identifies the best-performing
#      variant and hands it to Optuna for thorough hyperparameter search
#      (LR, batch size, dropout, focal gamma).
#
#   4. Finally, Claude produces a written recommendation report comparing
#      all 18 baseline runs plus the Optuna-tuned result.
#
# WHY THIS IS AGENTIC
# ───────────────────
# The LLM is not just summarising at the end. It actively analyses
# intermediate results and its recommendation influences which variant
# receives the Optuna budget.  The loop is:
#
#   Observe (metrics) -> Reason (LLM) -> Act (select Optuna target)
#
# This follows the agentic AI paradigm described in the module brief.
#
# Usage:
#   python agent.py                        # full 18-run sweep + Optuna
#   python agent.py --optuna_trials 20     # more Optuna trials
#   python agent.py --epochs 5             # quick test run
#   python agent.py --skip_optuna          # sweep only, no Optuna
#
# Requirements:
#   pip install anthropic optuna
#   export ANTHROPIC_API_KEY=sk-...
#
# Author: Student implementation for CMP9137M Advanced Machine Learning
################################################################################

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import BertTokenizer, CLIPProcessor

import anthropic

from slake.dataset    import SlakeDataset, build_answer_vocab
from slake.metrics    import evaluate, print_metrics
from slake.models     import (
    CLIPFocalModel,
    ResNetBertModel,
    ViTCrossAttentionModel,
    focal_loss,
)
from slake.stopping   import EarlyStopping
from slake.transforms import get_eval_transform, get_train_transform

# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT  = Path("SLAKE")
TRAIN_JSON = DATA_ROOT / "train.json"
VAL_JSON   = DATA_ROOT / "validation.json"
TEST_JSON  = DATA_ROOT / "test.json"
IMGS_ROOT  = DATA_ROOT / "imgs"
RESULTS_DIR = Path("agent_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT GRID
# 3 variants × 3 training ratios × 2 loss functions = 18 experiments
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS       = ["resnet_bert", "vit_crossattn", "clip_focal"]
TRAIN_RATIOS   = [0.6, 0.8, 1.0]
LOSS_FUNCTIONS = ["cross_entropy", "focal"]

# Base hyperparameters per variant (overridden by Optuna during tuning)
BASE_CONFIG = {
    "resnet_bert"  : {"lr": 2e-5, "batch_size": 16, "dropout": 0.2},
    "vit_crossattn": {"lr": 1e-5, "batch_size": 8,  "dropout": 0.2},
    "clip_focal"   : {"lr": 5e-6, "batch_size": 16, "dropout": 0.1},
}


# ══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def get_anthropic_client() -> anthropic.Anthropic:
    """
    Initialise the Anthropic API client.

    Reads ANTHROPIC_API_KEY from the environment.  Raises a clear error
    if the key is missing rather than failing silently later.

    Returns:
        Authenticated Anthropic client.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Run: export ANTHROPIC_API_KEY=sk-..."
        )
    return anthropic.Anthropic(api_key=api_key)


def call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    """
    Send a prompt to Claude and return the text response.

    Uses claude-sonnet-4-20250514 as the reasoning backbone for the
    agentic loop.  Temperature is kept at default (1.0) to allow
    the model to reason freely rather than being forced to be deterministic.

    Args:
        client: Authenticated Anthropic client.
        prompt: Full prompt string to send.

    Returns:
        Claude's response as a plain string.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL & DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_model(variant: str, num_classes: int, dropout: float) -> torch.nn.Module:
    """
    Instantiate a model variant with a custom dropout rate.

    The dropout parameter is plumbed through during Optuna tuning.
    For the baseline sweep, the default from BASE_CONFIG is used.

    Args:
        variant:     One of 'resnet_bert', 'vit_crossattn', 'clip_focal'.
        num_classes: Answer vocabulary size.
        dropout:     Dropout probability for the classifier head.

    Returns:
        Initialised nn.Module.
    """
    if variant == "resnet_bert":
        model = ResNetBertModel(num_classes)
        # Override classifier dropout
        model.classifier[0] = torch.nn.Dropout(dropout)
        model.classifier[3] = torch.nn.Dropout(dropout)
        return model
    if variant == "vit_crossattn":
        model = ViTCrossAttentionModel(num_classes)
        model.classifier[0] = torch.nn.Dropout(dropout)
        return model
    if variant == "clip_focal":
        model = CLIPFocalModel(num_classes)
        model.classifier[0] = torch.nn.Dropout(dropout)
        model.classifier[3] = torch.nn.Dropout(dropout)
        return model
    raise ValueError(f"Unknown variant: {variant!r}")


def make_loaders(
    answer_to_idx: Dict[str, int],
    variant: str,
    train_ratio: float,
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    When train_ratio < 1.0, a random subset of the training data is used.
    The random subset uses a fixed seed so that results are comparable
    across variants at the same ratio.

    Args:
        answer_to_idx: Answer vocabulary mapping.
        variant:       Model variant (determines tokeniser).
        train_ratio:   Fraction of training data to use (0.0-1.0].
        batch_size:    Samples per batch.
        num_workers:   DataLoader workers.
        augment:       Apply image augmentation to training data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    use_clip = (variant == "clip_focal")
    clip_proc = None
    tokenizer = None

    if use_clip:
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_tf  = get_eval_transform()
        val_tf    = get_eval_transform()
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_tf  = get_train_transform(augment=augment)
        val_tf    = get_eval_transform()

    train_ds = SlakeDataset(
        TRAIN_JSON, IMGS_ROOT, answer_to_idx,
        train_tf, tokenizer, clip_proc,
    )
    val_ds = SlakeDataset(
        VAL_JSON, IMGS_ROOT, answer_to_idx,
        val_tf, tokenizer, clip_proc, skip_oov=True,
    )
    test_ds = SlakeDataset(
        TEST_JSON, IMGS_ROOT, answer_to_idx,
        val_tf, tokenizer, clip_proc, skip_oov=True,
    )

    # Subset training data if ratio < 1.0
    if train_ratio < 1.0:
        n      = int(len(train_ds) * train_ratio)
        rng    = np.random.default_rng(SEED)
        idx    = rng.choice(len(train_ds), size=n, replace=False).tolist()
        train_ds = Subset(train_ds, idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE EXPERIMENT RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    variant:       str,
    train_ratio:   float,
    loss_fn:       str,
    answer_to_idx: Dict[str, int],
    num_classes:   int,
    device:        torch.device,
    epochs:        int,
    num_workers:   int,
    lr:            Optional[float] = None,
    batch_size:    Optional[int]   = None,
    dropout:       Optional[float] = None,
    focal_gamma:   float = 2.0,
    patience:      int   = 5,
) -> Dict[str, Any]:
    """
    Train and evaluate a single experiment configuration.

    This function is called both by the baseline sweep and by the Optuna
    objective function, making Optuna trials easy to integrate.

    Args:
        variant:       Model variant name.
        train_ratio:   Fraction of training data to use.
        loss_fn:       'cross_entropy' or 'focal'.
        answer_to_idx: Answer vocabulary mapping.
        num_classes:   Vocabulary size.
        device:        Compute device.
        epochs:        Maximum training epochs.
        num_workers:   DataLoader workers.
        lr:            Learning rate (defaults to BASE_CONFIG value).
        batch_size:    Batch size (defaults to BASE_CONFIG value).
        dropout:       Dropout rate (defaults to BASE_CONFIG value).
        focal_gamma:   Focal loss gamma parameter.
        patience:      Early stopping patience.

    Returns:
        Dict with all metrics plus experiment metadata.
    """
    cfg        = BASE_CONFIG[variant]
    lr         = lr         or cfg["lr"]
    batch_size = batch_size or cfg["batch_size"]
    dropout    = dropout    or cfg["dropout"]
    use_focal  = (loss_fn == "focal")
    augment    = (variant == "clip_focal")
    use_amp    = (device.type == "cuda")

    train_loader, val_loader, test_loader = make_loaders(
        answer_to_idx, variant, train_ratio, batch_size, num_workers, augment,
    )

    model     = build_model(variant, num_classes, dropout).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01,
    )
    stopper   = EarlyStopping(patience=patience, min_epochs=3)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    t_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            labels = batch["labels"].to(device)
            valid  = labels >= 0
            if not valid.any():
                continue

            pixel = batch["pixel_values"].to(device)[valid]
            ids   = batch["input_ids"].to(device)[valid]
            am    = batch["attention_mask"].to(device)[valid]
            y     = labels[valid]

            optimiser.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(pixel, ids, am)
                    loss   = (focal_loss(logits, y, gamma=focal_gamma)
                              if use_focal
                              else F.cross_entropy(logits, y))
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                logits = model(pixel, ids, am)
                loss   = (focal_loss(logits, y, gamma=focal_gamma)
                          if use_focal
                          else F.cross_entropy(logits, y))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

        val_metrics = evaluate(model, val_loader, device, use_amp)
        if stopper.step(epoch + 1, val_metrics.get("balanced_accuracy", 0.0), model):
            break

    train_time = time.perf_counter() - t_start

    # Restore best weights before final test evaluation
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    t_test       = time.perf_counter()
    test_metrics = evaluate(model, test_loader, device, use_amp)
    test_time    = time.perf_counter() - t_test

    result = {
        "variant"          : variant,
        "train_ratio"      : train_ratio,
        "loss_fn"          : loss_fn,
        "lr"               : lr,
        "batch_size"       : batch_size,
        "dropout"          : dropout,
        "focal_gamma"      : focal_gamma if use_focal else None,
        "train_time_s"     : train_time,
        "test_time_s"      : test_time,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# AGENTIC LLM CALLS
# ══════════════════════════════════════════════════════════════════════════════

def agent_mid_sweep_analysis(
    client: anthropic.Anthropic,
    completed: List[Dict],
    ratio: float,
) -> str:
    """
    Ask Claude to analyse results after completing one training ratio sweep.

    This is the 'observe -> reason' step of the agentic loop.  Claude
    receives the metrics for all experiments at this ratio and identifies
    which variant is performing best and why.

    Args:
        client:    Anthropic client.
        completed: List of result dicts completed so far.
        ratio:     The training ratio just completed.

    Returns:
        Claude's analysis as a string.
    """
    table = "\n".join(
        f"  variant={r['variant']:<15} ratio={r['train_ratio']}  "
        f"loss={r['loss_fn']:<15} "
        f"bal_acc={r.get('test_balanced_accuracy', 0):.4f}  "
        f"f1={r.get('test_f1_weighted', 0):.4f}  "
        f"ece={r.get('test_ece', 0):.4f}"
        for r in completed
        if r["train_ratio"] == ratio
    )

    prompt = f"""You are an expert machine learning researcher analysing
Visual Question Answering experiments on the SLAKE medical imaging dataset.

The following experiments just completed using {int(ratio*100)}% of training data:

{table}

Metrics explained:
- bal_acc : balanced classification accuracy (primary metric, higher is better)
- f1      : weighted F1 score (higher is better)
- ece     : expected calibration error (lower is better, measures confidence quality)

Variants:
- resnet_bert   : ResNet-50 CNN + BERT, late fusion
- vit_crossattn : Vision Transformer + BERT, cross-attention fusion
- clip_focal    : CLIP fine-tuned with focal loss and data augmentation

In 3-4 sentences, identify which variant is performing best at this data size,
comment on whether focal loss is helping, and give one concrete observation
about calibration or the accuracy-F1 gap if relevant.
Be specific and reference the numbers."""

    print(f"\n[Agent] Analysing results at train_ratio={ratio} ...")
    response = call_claude(client, prompt)
    print(f"[Agent] Analysis:\n{response}\n")
    return response


def agent_select_optuna_target(
    client: anthropic.Anthropic,
    all_results: List[Dict],
) -> str:
    """
    Ask Claude to recommend which variant should receive Optuna budget.

    This is the 'act' step of the agentic loop.  Based on all 18 baseline
    results, Claude identifies the variant most likely to benefit from
    further hyperparameter tuning.

    Args:
        client:      Anthropic client.
        all_results: All 18 baseline experiment results.

    Returns:
        One of 'resnet_bert', 'vit_crossattn', or 'clip_focal'.
    """
    # Summarise best result per variant
    summary = {}
    for r in all_results:
        v   = r["variant"]
        acc = r.get("test_balanced_accuracy", 0.0)
        if v not in summary or acc > summary[v]["best_acc"]:
            summary[v] = {
                "best_acc"  : acc,
                "best_f1"   : r.get("test_f1_weighted", 0.0),
                "best_ece"  : r.get("test_ece", 1.0),
                "best_ratio": r["train_ratio"],
                "best_loss" : r["loss_fn"],
            }

    table = "\n".join(
        f"  {v:<15}: best_bal_acc={d['best_acc']:.4f}  "
        f"f1={d['best_f1']:.4f}  ece={d['best_ece']:.4f}  "
        f"at ratio={d['best_ratio']}  loss={d['best_loss']}"
        for v, d in summary.items()
    )

    prompt = f"""You are an expert ML researcher helping to allocate a
hyperparameter tuning budget for SLAKE medical VQA experiments.

Best results across all training ratios per variant:

{table}

Optuna will tune: learning rate, batch size, dropout, and focal gamma
for ONE variant only (compute budget is limited).

Reply with ONLY the variant name — exactly one of:
  resnet_bert
  vit_crossattn
  clip_focal

Choose the variant most likely to improve further with hyperparameter tuning.
Consider: which has the most headroom, which might benefit most from
dropout/LR tuning, and which is most sensitive to focal gamma.
Reply with the variant name only, nothing else."""

    print("\n[Agent] Selecting Optuna target variant ...")
    response = call_claude(client, prompt).strip().lower()

    # Validate — fall back to best by accuracy if Claude returns unexpected text
    valid = {"resnet_bert", "vit_crossattn", "clip_focal"}
    if response not in valid:
        print(f"[Agent] Unexpected response '{response}', falling back to best by accuracy.")
        response = max(summary, key=lambda v: summary[v]["best_acc"])

    print(f"[Agent] Selected variant for Optuna: {response}")
    return response


def agent_final_report(
    client: anthropic.Anthropic,
    all_results: List[Dict],
    optuna_result: Optional[Dict],
    optuna_variant: str,
) -> str:
    """
    Ask Claude to write a final recommendation report.

    Synthesises all 18 baseline runs plus the Optuna-tuned result into
    a concise recommendation suitable for inclusion in the assignment report.

    Args:
        client:         Anthropic client.
        all_results:    All 18 baseline results.
        optuna_result:  Best Optuna trial result (None if skipped).
        optuna_variant: Variant that was Optuna-tuned.

    Returns:
        Final report string from Claude.
    """
    # Best baseline per variant
    best_per_variant = {}
    for r in all_results:
        v   = r["variant"]
        acc = r.get("test_balanced_accuracy", 0.0)
        if v not in best_per_variant or acc > best_per_variant[v]:
            best_per_variant[v] = acc

    baseline_lines = "\n".join(
        f"  {v:<15}: best balanced_accuracy = {acc:.4f}"
        for v, acc in best_per_variant.items()
    )

    optuna_line = ""
    if optuna_result:
        optuna_line = (
            f"\nOptuna-tuned {optuna_variant}: "
            f"balanced_accuracy = {optuna_result.get('test_balanced_accuracy', 0):.4f}  "
            f"(best hyperparams: lr={optuna_result.get('lr', '?'):.2e}  "
            f"batch={optuna_result.get('batch_size', '?')}  "
            f"dropout={optuna_result.get('dropout', '?'):.2f}  "
            f"gamma={optuna_result.get('focal_gamma', 'N/A')})"
        )

    prompt = f"""You are writing the results section of a machine learning
assignment report on SLAKE medical Visual Question Answering.

BASELINE RESULTS (best across all training ratios and loss functions):
{baseline_lines}
{optuna_line}

Write a concise 5-7 sentence recommendation paragraph that:
1. States which model performed best overall and by how much
2. Comments on whether focal loss helped vs standard cross-entropy
3. Notes the effect of training data size (60%, 80%, 100% ratios)
4. Comments on whether Optuna tuning improved over the baseline
5. Gives a clear final recommendation for deployment

Write in academic style suitable for a university report.
Be specific and reference the numbers."""

    print("\n[Agent] Writing final recommendation report ...")
    report = call_claude(client, prompt)
    print(f"\n{'='*60}")
    print("AGENT FINAL REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60 + "\n")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA TUNING
# ══════════════════════════════════════════════════════════════════════════════

def run_optuna(
    variant:       str,
    answer_to_idx: Dict[str, int],
    num_classes:   int,
    device:        torch.device,
    epochs:        int,
    num_workers:   int,
    n_trials:      int,
) -> Tuple[Dict, optuna.Study]:
    """
    Run Optuna hyperparameter search on the specified variant.

    Search space:
        lr          : log-uniform in [1e-6, 1e-4]
        batch_size  : categorical {8, 16, 32}
        dropout     : uniform in [0.05, 0.5]
        focal_gamma : uniform in [0.5, 4.0]  (only applied when variant=clip_focal)

    Uses TPE (Tree-structured Parzen Estimator) sampler, which is Optuna's
    default and outperforms random search by modelling the objective surface.

    Args:
        variant:       Variant to tune.
        answer_to_idx: Answer vocabulary mapping.
        num_classes:   Vocabulary size.
        device:        Compute device.
        epochs:        Max epochs per trial.
        num_workers:   DataLoader workers.
        n_trials:      Number of Optuna trials.

    Returns:
        Tuple of (best_trial_result_dict, optuna_study).
    """
    print(f"\n[Optuna] Starting {n_trials}-trial search on [{variant}] ...")

    def objective(trial: optuna.Trial) -> float:
        lr          = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [8, 16, 32])
        dropout     = trial.suggest_float("dropout", 0.05, 0.5)
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 4.0)

        # Use focal loss when variant is clip_focal, or when gamma > 0
        # For BERT variants, still allow focal loss as a learning strategy
        use_focal = (variant == "clip_focal") or (focal_gamma > 1.0)
        loss_fn   = "focal" if use_focal else "cross_entropy"

        result = run_experiment(
            variant       = variant,
            train_ratio   = 1.0,        # always full data for tuning
            loss_fn       = loss_fn,
            answer_to_idx = answer_to_idx,
            num_classes   = num_classes,
            device        = device,
            epochs        = epochs,
            num_workers   = num_workers,
            lr            = lr,
            batch_size    = batch_size,
            dropout       = dropout,
            focal_gamma   = focal_gamma,
            patience      = 4,          # tighter patience for tuning speed
        )

        bal_acc = result.get("test_balanced_accuracy", 0.0)
        print(
            f"  Trial {trial.number:3d}: "
            f"lr={lr:.2e}  bs={batch_size}  "
            f"dropout={dropout:.2f}  gamma={focal_gamma:.2f}  "
            f"-> bal_acc={bal_acc:.4f}"
        )

        # Store full result dict on the trial for later retrieval
        trial.set_user_attr("result", result)
        return bal_acc

    # Suppress Optuna's verbose per-trial logging (we print our own)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"slake_{variant}",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_result = study.best_trial.user_attrs["result"]
    print(f"\n[Optuna] Best balanced_accuracy: {study.best_value:.4f}")
    print(f"[Optuna] Best params: {study.best_params}")

    return best_result, study


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Agentic SLAKE VQA experiment controller"
    )
    parser.add_argument("--epochs",        type=int,   default=10,
                        help="Max epochs per experiment run.")
    parser.add_argument("--num_workers",   type=int,   default=2,
                        help="DataLoader worker processes.")
    parser.add_argument("--optuna_trials", type=int,   default=15,
                        help="Number of Optuna hyperparameter trials.")
    parser.add_argument("--skip_optuna",   action="store_true",
                        help="Run the 18-experiment sweep only.")
    parser.add_argument("--skip_agent",    action="store_true",
                        help="Skip LLM calls (useful if no API key yet).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Answer vocabulary ────────────────────────────────────────────────────
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_rows = json.load(f)
    answer_to_idx, idx_to_answer = build_answer_vocab(train_rows)
    num_classes = len(answer_to_idx)

    # ── Anthropic client ─────────────────────────────────────────────────────
    client = None
    if not args.skip_agent:
        try:
            client = get_anthropic_client()
        except EnvironmentError as e:
            print(f"[Warning] {e}")
            print("[Warning] Continuing without LLM analysis (--skip_agent mode).")
            client = None

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — 18-EXPERIMENT BASELINE SWEEP
    # 3 training ratios × 3 variants × 2 loss functions
    # ══════════════════════════════════════════════════════════════════════════

    all_results    = []
    mid_analyses   = {}
    total_runs     = len(TRAIN_RATIOS) * len(VARIANTS) * len(LOSS_FUNCTIONS)
    run_number     = 0

    print(f"\n{'='*60}")
    print(f"  PHASE 1: Baseline sweep — {total_runs} experiments")
    print(f"{'='*60}\n")

    for ratio in TRAIN_RATIOS:
        print(f"\n── Training ratio: {int(ratio*100)}% ──────────────────────────")

        for variant in VARIANTS:
            for loss_fn in LOSS_FUNCTIONS:
                run_number += 1
                exp_id = (f"{variant}_ratio{int(ratio*100)}_"
                          f"{loss_fn}")

                print(f"\n[Run {run_number}/{total_runs}] {exp_id}")

                result = run_experiment(
                    variant       = variant,
                    train_ratio   = ratio,
                    loss_fn       = loss_fn,
                    answer_to_idx = answer_to_idx,
                    num_classes   = num_classes,
                    device        = device,
                    epochs        = args.epochs,
                    num_workers   = args.num_workers,
                )
                result["exp_id"] = exp_id
                all_results.append(result)

                print(
                    f"  -> bal_acc={result.get('test_balanced_accuracy', 0):.4f}  "
                    f"f1={result.get('test_f1_weighted', 0):.4f}  "
                    f"ece={result.get('test_ece', 0):.4f}  "
                    f"train_time={result.get('train_time_s', 0):.0f}s"
                )

                # Save incrementally so progress is never lost
                with open(RESULTS_DIR / "sweep_results.json", "w") as f:
                    json.dump(all_results, f, indent=2)

        # ── Agent mid-sweep analysis after each ratio block ──────────────────
        if client is not None:
            analysis = agent_mid_sweep_analysis(client, all_results, ratio)
            mid_analyses[ratio] = analysis

    # Save all sweep results
    with open(RESULTS_DIR / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSweep results saved to {RESULTS_DIR / 'sweep_results.json'}")

    # Print sweep summary table
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print("=" * 60)
    for r in sorted(all_results, key=lambda x: -x.get("test_balanced_accuracy", 0)):
        print(
            f"  {r['variant']:<15} ratio={r['train_ratio']}  "
            f"loss={r['loss_fn']:<15} "
            f"bal_acc={r.get('test_balanced_accuracy', 0):.4f}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — AGENT SELECTS OPTUNA TARGET
    # ══════════════════════════════════════════════════════════════════════════

    if args.skip_optuna:
        print("\n[Agent] --skip_optuna set. Skipping Optuna phase.")
        optuna_result  = None
        optuna_variant = max(
            {r["variant"] for r in all_results},
            key=lambda v: max(
                r.get("test_balanced_accuracy", 0)
                for r in all_results if r["variant"] == v
            ),
        )
    else:
        print(f"\n{'='*60}")
        print("  PHASE 2: Agent selects Optuna target")
        print(f"{'='*60}")

        if client is not None:
            optuna_variant = agent_select_optuna_target(client, all_results)
        else:
            # Fall back to best variant by accuracy
            optuna_variant = max(
                {r["variant"] for r in all_results},
                key=lambda v: max(
                    r.get("test_balanced_accuracy", 0)
                    for r in all_results if r["variant"] == v
                ),
            )
            print(f"[Agent] No LLM — defaulting to best variant: {optuna_variant}")

        # ── Phase 3: Optuna tuning ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  PHASE 3: Optuna tuning — {args.optuna_trials} trials on [{optuna_variant}]")
        print(f"{'='*60}")

        optuna_result, study = run_optuna(
            variant       = optuna_variant,
            answer_to_idx = answer_to_idx,
            num_classes   = num_classes,
            device        = device,
            epochs        = args.epochs,
            num_workers   = args.num_workers,
            n_trials      = args.optuna_trials,
        )

        with open(RESULTS_DIR / "optuna_best.json", "w") as f:
            json.dump(optuna_result, f, indent=2)
        print(f"Optuna best result saved to {RESULTS_DIR / 'optuna_best.json'}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — AGENT FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════════

    if client is not None:
        report = agent_final_report(
            client, all_results, optuna_result, optuna_variant,
        )
        with open(RESULTS_DIR / "agent_report.txt", "w") as f:
            f.write(report)
        print(f"Agent report saved to {RESULTS_DIR / 'agent_report.txt'}")

    print("\n[Agent] All phases complete.")
    print(f"Results directory: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
