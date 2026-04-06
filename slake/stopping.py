################################################################################
# slake/stopping.py
#
# Early stopping based on validation balanced accuracy.
#
# Monitors a validation metric (higher is better) and signals the training
# loop to stop when the metric has not improved for `patience` epochs.
# The best model state seen during training is cached internally and can
# be restored after stopping.
################################################################################

import copy

import torch.nn as nn


class EarlyStopping:
    """
    Early stopping monitor for validation balanced accuracy.

    Tracks the best metric seen so far and caches the corresponding model
    weights.  Once the metric has not improved for `patience` consecutive
    epochs (after `min_epochs` have elapsed), signals the training loop
    to stop.

    Usage:
        stopper = EarlyStopping(patience=5, min_epochs=3)
        for epoch in range(max_epochs):
            val_metrics = evaluate(model, val_loader, device)
            bal_acc = val_metrics["balanced_accuracy"]
            if stopper.step(epoch + 1, bal_acc, model):
                break
        if stopper.best_state is not None:
            model.load_state_dict(stopper.best_state)

    Args:
        patience:   Epochs to wait after the last improvement before stopping.
        min_delta:  Minimum absolute improvement to count as progress.
        min_epochs: Do not stop before this many epochs have elapsed.
    """

    def __init__(
        self,
        patience:   int   = 5,
        min_delta:  float = 1e-4,
        min_epochs: int   = 3,
    ):
        self.patience   = patience
        self.min_delta  = min_delta
        self.min_epochs = min_epochs
        self.best       = -float("inf")
        self.wait       = 0
        self.best_state = None
        self.improved   = False

    def step(self, epoch: int, score: float, model: nn.Module) -> bool:
        """
        Evaluate the current metric and decide whether to stop.

        Args:
            epoch: Current epoch number (1-indexed).
            score: Validation balanced accuracy for this epoch.
            model: Model whose state_dict is cached on improvement.

        Returns:
            True  -> training should stop.
            False -> training should continue.
        """
        self.improved = False

        if score > self.best + self.min_delta:
            self.best       = score
            self.wait       = 0
            self.improved   = True
            self.best_state = copy.deepcopy(model.state_dict())
            print(f"  [EarlyStopping] New best balanced_accuracy: {self.best:.4f}")
        else:
            self.wait += 1
            print(
                f"  [EarlyStopping] No improvement for {self.wait}/{self.patience} epochs"
            )

        if epoch >= self.min_epochs and self.wait >= self.patience:
            print(
                f"  [EarlyStopping] Stopping at epoch {epoch}. "
                f"Best balanced_accuracy: {self.best:.4f}"
            )
            return True

        return False
