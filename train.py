################################################################################
# train.py
#
# Main training script for SLAKE multi-class VQA.
#
# Trains one of three model variants and evaluates it on the test set.
# All results and checkpoints are saved under checkpoints/<variant>/.
#
# Usage:
#   python train.py --variant resnet_bert
#   python train.py --variant vit_crossattn
#   python train.py --variant clip_focal
#
# Data layout expected:
#   SLAKE/
#     train.json
#     validation.json
#     test.json
#     imgs/
#
# Author: Student implementation for CMP9137M Advanced Machine Learning
################################################################################

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, CLIPProcessor

from slake.dataset   import SlakeDataset, build_answer_vocab
from slake.metrics   import evaluate, print_metrics
from slake.models    import CLIPFocalModel, ResNetBertModel, ViTCrossAttentionModel, focal_loss
from slake.stopping  import EarlyStopping
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
CKPT_ROOT  = Path("checkpoints")

# ══════════════════════════════════════════════════════════════════════════════
# VARIANT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Each entry defines the hyperparameters and flags for a single variant.
# Keeping them here makes it easy to compare variants at a glance and
# ensures the training loop itself stays generic.

VARIANT_CONFIG = {
    "resnet_bert": {
        "description" : "ResNet-50 + BERT, late fusion, cross-entropy",
        "use_focal"   : False,
        "augment"     : False,
        "use_clip"    : False,
        "lr"          : 2e-5,
        "batch_size"  : 16,
        "patience"    : 5,
        "min_epochs"  : 3,
    },
    "vit_crossattn": {
        "description" : "ViT-B/16 + BERT, cross-attention fusion, cross-entropy",
        "use_focal"   : False,
        "augment"     : False,
        "use_clip"    : False,
        "lr"          : 1e-5,
        "batch_size"  : 8,
        "patience"    : 5,
        "min_epochs"  : 3,
    },
    "clip_focal": {
        "description" : "CLIP fine-tuned, focal loss + augmentation",
        "use_focal"   : True,
        "augment"     : True,
        "use_clip"    : True,
        "lr"          : 5e-6,
        "batch_size"  : 16,
        "patience"    : 7,
        "min_epochs"  : 3,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_model(variant: str, num_classes: int) -> torch.nn.Module:
    """
    Instantiate the model for the requested variant.

    Args:
        variant:     One of 'resnet_bert', 'vit_crossattn', 'clip_focal'.
        num_classes: Number of answer classes from the training vocabulary.

    Returns:
        Initialised nn.Module.
    """
    if variant == "resnet_bert":
        return ResNetBertModel(num_classes)
    if variant == "vit_crossattn":
        return ViTCrossAttentionModel(num_classes)
    if variant == "clip_focal":
        return CLIPFocalModel(num_classes)
    raise ValueError(f"Unknown variant: {variant!r}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_variant(
    model:         torch.nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    optimiser:     torch.optim.Optimizer,
    device:        torch.device,
    num_epochs:    int,
    use_focal:     bool,
    use_amp:       bool,
    early_stopper: EarlyStopping,
    variant_name:  str,
):
    """
    Training loop with validation, early stopping, and AMP support.

    Each epoch:
      1. Forward pass on training batches with chosen loss function.
      2. Backward pass with gradient clipping (max norm = 1.0).
      3. Validation evaluation on the full val set.
      4. Early stopping check on balanced accuracy.

    Args:
        model:          Model to train (modified in place).
        train_loader:   DataLoader for training split.
        val_loader:     DataLoader for validation split.
        optimiser:      Parameter optimiser.
        device:         Compute device.
        num_epochs:     Maximum epochs to run.
        use_focal:      Use focal loss (True for clip_focal variant).
        use_amp:        Enable automatic mixed precision on CUDA.
        early_stopper:  EarlyStopping instance.
        variant_name:   Label used in progress output.

    Returns:
        Tuple of (total_training_seconds: float, epochs_run: int).
    """
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_name = "FocalLoss" if use_focal else "CrossEntropy"
    print(f"\nTraining [{variant_name}]  loss={loss_name}  AMP={use_amp}")

    t_start    = time.perf_counter()
    epochs_run = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n            = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in bar:
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
                    loss   = (focal_loss(logits, y)
                              if use_focal
                              else F.cross_entropy(logits, y))
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                logits = model(pixel, ids, am)
                loss   = (focal_loss(logits, y)
                          if use_focal
                          else F.cross_entropy(logits, y))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            running_loss += loss.item() * y.size(0)
            n            += y.size(0)
            bar.set_postfix(loss=f"{running_loss / max(n, 1):.4f}")

        epochs_run = epoch + 1
        avg_loss   = running_loss / max(n, 1)
        print(f"  Epoch {epoch+1:3d}/{num_epochs}  train_loss={avg_loss:.4f}")

        # Validate and check early stopping
        val_metrics = evaluate(model, val_loader, device, use_amp)
        bal_acc     = val_metrics.get("balanced_accuracy", 0.0)
        val_loss    = val_metrics.get("loss", 0.0)
        print(f"  val balanced_acc={bal_acc:.4f}  val_loss={val_loss:.4f}")

        if early_stopper.step(epoch + 1, bal_acc, model):
            break

    total_seconds = time.perf_counter() - t_start
    print(f"\nFinished [{variant_name}]: {total_seconds:.1f}s  ({epochs_run} epochs)")
    return total_seconds, epochs_run


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SLAKE multi-class VQA training")
    parser.add_argument(
        "--variant", type=str, default="resnet_bert",
        choices=list(VARIANT_CONFIG.keys()),
        help="Which model variant to train.",
    )
    parser.add_argument("--epochs",      type=int, default=20,
                        help="Maximum training epochs.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes.")
    args = parser.parse_args()

    cfg    = VARIANT_CONFIG[args.variant]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"\n{'='*55}")
    print(f"  Variant : {args.variant}")
    print(f"  Desc    : {cfg['description']}")
    print(f"  Device  : {device}  |  AMP: {use_amp}")
    print(f"{'='*55}\n")

    # ── Answer vocabulary ────────────────────────────────────────────────────
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_rows = json.load(f)
    answer_to_idx, idx_to_answer = build_answer_vocab(train_rows)
    num_classes = len(answer_to_idx)

    # ── Tokeniser / processor ────────────────────────────────────────────────
    clip_proc = None
    tokenizer = None

    if cfg["use_clip"]:
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_tf  = get_eval_transform()   # CLIP processor handles its own resize
        val_tf    = get_eval_transform()
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_tf  = get_train_transform(augment=cfg["augment"])
        val_tf    = get_eval_transform()

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("Loading datasets ...")
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

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=args.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=args.num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=args.num_workers, pin_memory=pin,
    )

    # ── Model & optimiser ────────────────────────────────────────────────────
    model     = build_model(args.variant, num_classes).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=0.01,
    )
    early_stopper = EarlyStopping(
        patience=cfg["patience"], min_epochs=cfg["min_epochs"],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    train_seconds, epochs_run = train_one_variant(
        model, train_loader, val_loader, optimiser,
        device, args.epochs,
        use_focal=cfg["use_focal"],
        use_amp=use_amp,
        early_stopper=early_stopper,
        variant_name=args.variant,
    )

    # Restore best weights found during training
    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)
        print("Restored best model weights.")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    out_dir = CKPT_ROOT / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "variant"         : args.variant,
        "num_classes"     : num_classes,
        "answer_to_idx"   : answer_to_idx,
        "idx_to_answer"   : idx_to_answer,
        "train_seconds"   : train_seconds,
        "epochs_run"      : epochs_run,
    }, ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    t0           = time.perf_counter()
    test_metrics = evaluate(model, test_loader, device, use_amp)
    test_time    = time.perf_counter() - t0

    test_metrics["train_time_s"] = train_seconds
    test_metrics["test_time_s"]  = test_time
    test_metrics["epochs_run"]   = float(epochs_run)

    print_metrics(test_metrics, prefix=f"TEST  {args.variant}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Metrics saved -> {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
