################################################################################
# predict.py
#
# Inference script for trained SLAKE VQA models.
#
# Loads a saved checkpoint and predicts the top-k answers for a given
# image and question, with confidence scores.
#
# Usage:
#   python predict.py \
#       --checkpoint checkpoints/resnet_bert/model.pt \
#       --image SLAKE/imgs/xmlab0/source.jpg \
#       --question "What organ is shown in this image?"
#
# Author: Student implementation for CMP9137M Advanced Machine Learning
################################################################################

import argparse
import json

import torch
from PIL import Image
from transformers import BertTokenizer, CLIPProcessor

from slake.models    import CLIPFocalModel, ResNetBertModel, ViTCrossAttentionModel
from slake.transforms import get_eval_transform


def load_checkpoint(ckpt_path: str, device: torch.device):
    """
    Load a saved model checkpoint and reconstruct the model.

    Args:
        ckpt_path: Path to the .pt checkpoint file.
        device:    Compute device to load weights onto.

    Returns:
        Tuple of (model, idx_to_answer, variant).
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    variant       = ckpt["variant"]
    num_classes   = ckpt["num_classes"]
    idx_to_answer = ckpt["idx_to_answer"]

    if variant == "resnet_bert":
        model = ResNetBertModel(num_classes)
    elif variant == "vit_crossattn":
        model = ViTCrossAttentionModel(num_classes)
    elif variant == "clip_focal":
        model = CLIPFocalModel(num_classes)
    else:
        raise ValueError(f"Unknown variant in checkpoint: {variant!r}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded [{variant}] checkpoint from {ckpt_path}")
    return model, idx_to_answer, variant


def predict(
    model,
    idx_to_answer,
    variant: str,
    image_path: str,
    question: str,
    device: torch.device,
    top_k: int = 5,
):
    """
    Predict the top-k answers for an image-question pair.

    Args:
        model:        Trained model in eval mode.
        idx_to_answer: List mapping class index -> answer string.
        variant:      Model variant string (determines tokeniser).
        image_path:   Path to the input image file.
        question:     Question string.
        device:       Compute device.
        top_k:        Number of top answers to return.

    Returns:
        List of (answer, confidence) tuples ranked by confidence.
    """
    img = Image.open(image_path).convert("RGB")

    if variant == "clip_focal":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        proc = processor(
            text=[question],
            images=img,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        pixel = proc["pixel_values"].to(device)
        ids   = proc["input_ids"].to(device)
        am    = proc["attention_mask"].to(device)
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        transform = get_eval_transform()
        pixel = transform(img).unsqueeze(0).to(device)
        enc   = tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        am  = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(pixel, ids, am)
        probs  = torch.softmax(logits, dim=1)[0]

    k = min(top_k, len(idx_to_answer))
    top_probs, top_idx = torch.topk(probs, k=k)

    results = [
        (idx_to_answer[i.item()], p.item())
        for i, p in zip(top_idx, top_probs)
    ]

    print(f"\nQuestion : {question}")
    print(f"Image    : {image_path}")
    print(f"\nTop-{k} predicted answers:")
    for rank, (answer, conf) in enumerate(results, 1):
        print(f"  {rank}. {answer:<30}  confidence: {conf:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SLAKE VQA inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model .pt checkpoint.")
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to input image file.")
    parser.add_argument("--question",   type=str, required=True,
                        help="Question string.")
    parser.add_argument("--top_k",      type=int, default=5,
                        help="Number of top answers to display.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_answer, variant = load_checkpoint(args.checkpoint, device)

    predict(
        model, idx_to_answer, variant,
        args.image, args.question,
        device, top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
