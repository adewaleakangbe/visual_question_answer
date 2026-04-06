################################################################################
# slake/transforms.py
#
# Image transform pipelines for training and evaluation.
#
# Training transforms optionally include data augmentation (Variant 3).
# Evaluation transforms are deterministic — resize and normalise only.
#
# ImageNet normalisation statistics are used because all vision backbones
# (ResNet-50, ViT-B/16) were pretrained on ImageNet.
################################################################################

from torchvision import transforms


def get_train_transform(augment: bool = False) -> transforms.Compose:
    """
    Build the training image transform pipeline.

    When augment=True (used by Variant 3 / clip_focal), applies random
    horizontal flip, colour jitter, and random resized crop to improve
    generalisation on the limited SLAKE training set.

    Standard ImageNet normalisation is applied in all cases so that
    pretrained backbone weights receive inputs in their expected range.

    Args:
        augment: Whether to include stochastic augmentation ops.

    Returns:
        A torchvision Compose transform pipeline.
    """
    ops = [transforms.Resize((224, 224))]

    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            ),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(ops)


def get_eval_transform() -> transforms.Compose:
    """
    Deterministic evaluation transform — resize and normalise only.

    Used for validation and test sets across all variants.

    Returns:
        A torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
