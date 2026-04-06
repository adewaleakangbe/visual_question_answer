################################################################################
# slake/dataset.py
#
# Answer vocabulary builder and SLAKE VQA dataset.
#
# The dataset loads samples from the SLAKE JSON files and returns
# (image, question_tokens, label) triples for multi-class classification.
# Each label is an integer index into the training answer vocabulary.
#
# Samples whose answers do not appear in the vocabulary are assigned
# label -1.  The training loop filters these out via the valid mask.
################################################################################

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


def build_answer_vocab(
    train_samples: List[dict],
) -> Tuple[Dict[str, int], List[str]]:
    """
    Build a closed answer vocabulary from training samples.

    Collects every unique answer string in the training set and assigns
    each a stable integer index (alphabetically sorted for reproducibility
    across runs).

    Only training samples are used to build the vocabulary — validation
    and test answers that fall outside this set are labelled -1 at load
    time.

    Args:
        train_samples: List of dicts loaded from train.json.  Each dict
                       must have an "answer" key.

    Returns:
        answer_to_idx: Maps answer string -> class index (int).
        idx_to_answer: Maps class index -> answer string (list).
    """
    idx_to_answer = sorted({s["answer"] for s in train_samples})
    answer_to_idx = {a: i for i, a in enumerate(idx_to_answer)}
    print(f"Answer vocabulary size: {len(idx_to_answer)}")
    return answer_to_idx, idx_to_answer


class SlakeDataset(Dataset):
    """
    SLAKE VQA dataset for multi-class answer prediction.

    Each sample is a (image, question) pair labelled with the index of
    the correct answer in the training vocabulary.

    Supports two question encoding modes:
      - BERT tokeniser  : used by Variants 1 (resnet_bert) and 2 (vit_crossattn).
      - CLIP processor  : used by Variant 3 (clip_focal); handles image and
                          text jointly inside __getitem__.

    Args:
        json_path:       Path to train / validation / test JSON file.
        imgs_root:       Path to the SLAKE imgs/ directory.
        answer_to_idx:   Answer vocabulary mapping built from training data.
        image_transform: torchvision transform applied to each image.
                         Ignored when clip_processor is provided.
        tokenizer:       HuggingFace BERT tokeniser.  None for CLIP variant.
        clip_processor:  CLIPProcessor instance.  None for BERT variants.
        max_q_len:       Maximum token length for BERT question encoding.
        skip_oov:        If True, drop samples with out-of-vocabulary answers
                         at load time (useful for val / test sets).
    """

    def __init__(
        self,
        json_path: Path,
        imgs_root: Path,
        answer_to_idx: Dict[str, int],
        image_transform,
        tokenizer=None,
        clip_processor=None,
        max_q_len: int = 64,
        skip_oov: bool = False,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.imgs_root        = imgs_root
        self.answer_to_idx    = answer_to_idx
        self.image_transform  = image_transform
        self.tokenizer        = tokenizer
        self.clip_processor   = clip_processor
        self.max_q_len        = max_q_len
        self.samples: List[dict] = []

        skipped = 0
        for row in raw:
            img_path = imgs_root / row["img_name"]
            if not img_path.is_file():
                skipped += 1
                continue

            ans   = row["answer"]
            label = answer_to_idx.get(ans, -1)

            if label == -1 and skip_oov:
                skipped += 1
                continue

            self.samples.append({
                "path"    : img_path,
                "question": row["question"],
                "label"   : label,
                "answer"  : ans,
            })

        print(
            f"  Loaded {len(self.samples)} samples "
            f"from {Path(json_path).name}  (skipped {skipped})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")

        # ── CLIP variant ─────────────────────────────────────────────────────
        # CLIPProcessor handles image resizing and text tokenisation jointly.
        if self.clip_processor is not None:
            proc = self.clip_processor(
                text=[s["question"]],
                images=img,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
            return {
                "pixel_values"  : proc["pixel_values"].squeeze(0),
                "input_ids"     : proc["input_ids"].squeeze(0),
                "attention_mask": proc["attention_mask"].squeeze(0),
                "labels"        : torch.tensor(s["label"], dtype=torch.long),
            }

        # ── BERT variants ─────────────────────────────────────────────────────
        img = self.image_transform(img)
        enc = self.tokenizer(
            s["question"],
            padding="max_length",
            truncation=True,
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        return {
            "pixel_values"  : img,
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels"        : torch.tensor(s["label"], dtype=torch.long),
        }
