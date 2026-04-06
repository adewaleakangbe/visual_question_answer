################################################################################
# slake/metrics.py
#
# Evaluation metrics for SLAKE VQA multi-class classification.
#
# All required assignment metrics are computed in a single evaluation pass:
#   - Exact match accuracy
#   - Balanced classification accuracy
#   - Weighted F1 score
#   - Mean Reciprocal Rank (top-1)
#   - Expected Calibration Error
#   - Mean cross-entropy loss
################################################################################

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predictions by their confidence score and measures the average
    absolute gap between mean confidence and mean accuracy within each bin.
    A perfectly calibrated model has ECE = 0.

    Args:
        confidences: Model confidence for the predicted class, shape (N,).
        correct:     Binary correctness array (1=correct), shape (N,).
        n_bins:      Number of equally-spaced confidence bins in [0, 1].

    Returns:
        ECE as a float in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(confidences)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    """
    Run inference on a DataLoader and compute all assignment metrics.

    Handles the label=-1 mask (out-of-vocabulary answers) automatically —
    masked samples are excluded from all metric calculations.

    Metrics returned:
        accuracy          : Exact match accuracy across all predictions.
        balanced_accuracy : Mean per-class recall (handles class imbalance).
        f1_weighted       : Weighted F1 score (accounts for class frequency).
        mrr               : Mean Reciprocal Rank at rank-1.
        ece               : Expected Calibration Error.
        loss              : Mean cross-entropy loss per sample.

    Args:
        model:   Trained model (will be set to eval mode).
        loader:  DataLoader for the evaluation split.
        device:  Compute device (cpu or cuda).
        use_amp: Whether to run inference under automatic mixed precision.

    Returns:
        Dict mapping metric name -> float value.
    """
    model.eval()

    all_labels = []
    all_preds  = []
    all_confs  = []
    total_loss = 0.0
    n_samples  = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            valid  = labels >= 0
            if not valid.any():
                continue

            pixel = batch["pixel_values"].to(device)[valid]
            ids   = batch["input_ids"].to(device)[valid]
            am    = batch["attention_mask"].to(device)[valid]
            y     = labels[valid]

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(pixel, ids, am)
            else:
                logits = model(pixel, ids, am)

            loss  = F.cross_entropy(logits, y, reduction="sum")
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confs = probs.max(dim=1).values

            total_loss += loss.item()
            n_samples  += y.size(0)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    if n_samples == 0:
        return {}

    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    confs_arr  = np.array(all_confs)
    correct    = (preds_arr == labels_arr).astype(float)

    # Exact match accuracy
    accuracy = float(correct.mean())

    # Balanced accuracy: mean per-class recall
    # Each class contributes equally regardless of how many samples it has.
    classes      = np.unique(labels_arr)
    per_class    = [correct[labels_arr == c].mean() for c in classes]
    balanced_acc = float(np.mean(per_class))

    # Weighted F1
    f1 = float(f1_score(
        labels_arr, preds_arr, average="weighted", zero_division=0
    ))

    # MRR at rank-1: equals accuracy for top-1 predictions
    mrr = float(correct.mean())

    # ECE
    ece = compute_ece(confs_arr, correct)

    # Average loss
    avg_loss = total_loss / n_samples

    return {
        "accuracy"         : accuracy,
        "balanced_accuracy": balanced_acc,
        "f1_weighted"      : f1,
        "mrr"              : mrr,
        "ece"              : ece,
        "loss"             : avg_loss,
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """
    Pretty-print a metrics dictionary to stdout.

    Args:
        metrics: Dict of metric name -> float value.
        prefix:  Optional label shown in the header (e.g. variant name).
    """
    header = f"[{prefix}]" if prefix else "Results"
    print(f"\n{header}")
    print("─" * 45)
    for k, v in metrics.items():
        print(f"  {k:<25}: {v:.4f}")
    print("─" * 45 + "\n")
