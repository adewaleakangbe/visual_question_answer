################################################################################
# slake/models.py
#
# Three multimodal model architectures for SLAKE VQA.
#
# All three models take (pixel_values, input_ids, attention_mask) as input
# and output raw logits of shape (B, num_classes).  The answer vocabulary
# size (num_classes) is determined at runtime from the training data.
#
# ─────────────────────────────────────────────────────────────────────────────
# VARIANT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
#
#  resnet_bert   |  ResNet-50 + BERT  |  Late concatenation  |  Cross-entropy
#  vit_crossattn |  ViT-B/16 + BERT  |  Cross-attention      |  Cross-entropy
#  clip_focal    |  CLIP (ViT-B/32)  |  Joint VLM space      |  Focal loss
#
# Architecture differences:
#   - resnet_bert uses a CNN (ResNet-50) vision backbone.
#   - vit_crossattn uses a Vision Transformer (ViT-B/16) and fuses
#     modalities via multi-head cross-attention.
#   - clip_focal uses a single joint vision-language backbone (CLIP)
#     pretrained with a contrastive image-text objective.
#
# Fusion differences:
#   - Late fusion (resnet_bert): each modality is encoded independently
#     and concatenated only at the classifier input.
#   - Cross-attention (vit_crossattn): the question CLS token queries over
#     all image patch tokens, attending to relevant visual regions.
#   - Joint embedding (clip_focal): CLIP projects image and text into a
#     shared semantic space before classification.
#
# References:
#   - He et al. "Deep Residual Learning." CVPR 2016.  (ResNet)
#   - Dosovitskiy et al. "An Image is Worth 16x16 Words." ICLR 2021.  (ViT)
#   - Radford et al. "Learning Transferable Visual Models from Natural
#     Language Supervision." ICML 2021.  (CLIP)
#   - Devlin et al. "BERT." NAACL 2019.
#   - Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017.
################################################################################

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from transformers import BertModel, CLIPModel, ViTModel


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT 1 — ResNet-50 + BERT, late fusion
# ══════════════════════════════════════════════════════════════════════════════

class ResNetBertModel(nn.Module):
    """
    Variant 1: ResNet-50 CNN + BERT with late fusion.

    Vision path  : ResNet-50 (ImageNet pretrained), global average pool
                   -> 2048-dimensional feature vector.
    Language path: BERT-base, [CLS] pooler output
                   -> 768-dimensional feature vector.
    Fusion       : Concatenate (2816-d) -> MLP classifier.

    Late fusion is the simplest multimodal strategy — each modality is
    encoded independently and combined only at the final classification
    stage.  This serves as the architectural baseline.

    Args:
        num_classes: Number of answer classes in the vocabulary.
        bert_name:   HuggingFace model identifier for BERT.
    """

    def __init__(
        self,
        num_classes: int,
        bert_name: str = "bert-base-uncased",
    ):
        super().__init__()

        # Vision: ResNet-50 without the final FC layer -> (B, 2048, 1, 1)
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        self.bert = BertModel.from_pretrained(bert_name)

        d_img = 2048
        d_txt = self.bert.config.hidden_size   # 768
        d_in  = d_img + d_txt                  # 2816

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_in, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        img_feat = self.cnn(pixel_values).flatten(1)          # (B, 2048)
        txt_feat = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output                                        # (B, 768)
        fused = torch.cat([img_feat, txt_feat], dim=-1)       # (B, 2816)
        return self.classifier(fused)


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT 2 — ViT-B/16 + BERT, cross-attention fusion
# ══════════════════════════════════════════════════════════════════════════════

class ViTCrossAttentionModel(nn.Module):
    """
    Variant 2: ViT-B/16 + BERT with cross-attention fusion.

    Vision path  : ViT-Base/16 (pretrained) produces a sequence of 197
                   patch tokens (768-d each) via its last hidden state.
    Language path: BERT-base [CLS] token (768-d) acts as the query.
    Fusion       : Multi-head cross-attention where the question CLS token
                   queries over all image patch tokens.  The attended
                   output is residually connected, then classified.

    Cross-attention allows the model to selectively attend to the image
    regions most relevant to the question, rather than treating the image
    as a single flat vector.  This produces richer multimodal
    representations than late fusion.

    Args:
        num_classes: Number of answer classes.
        vit_name:    HuggingFace model identifier for ViT.
        bert_name:   HuggingFace model identifier for BERT.
        num_heads:   Number of cross-attention heads.
    """

    def __init__(
        self,
        num_classes: int,
        vit_name:  str = "google/vit-base-patch16-224",
        bert_name: str = "bert-base-uncased",
        num_heads: int = 8,
    ):
        super().__init__()
        self.vit  = ViTModel.from_pretrained(vit_name)
        self.bert = BertModel.from_pretrained(bert_name)

        d_v = self.vit.config.hidden_size    # 768
        d_t = self.bert.config.hidden_size   # 768

        # Project text CLS token into vision space
        # (same dimensionality here but explicit for clarity)
        self.text_proj = nn.Linear(d_t, d_v)

        # Cross-attention: query=text CLS, key/value=ViT patch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_v,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_v)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_v, 512),
            nn.GELU(),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Vision: full sequence of patch tokens
        patch_tokens = self.vit(
            pixel_values=pixel_values
        ).last_hidden_state                                     # (B, 197, 768)

        # Language: CLS token only as the cross-attention query
        cls_token = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0:1, :]                         # (B, 1, 768)

        query = self.text_proj(cls_token)                      # (B, 1, 768)

        # Question attends over image patch tokens
        attended, _ = self.cross_attn(
            query=query,
            key=patch_tokens,
            value=patch_tokens,
        )                                                       # (B, 1, 768)

        # Residual connection + layer norm for training stability
        attended = self.norm(attended + query)                  # (B, 1, 768)
        pooled   = attended.squeeze(1)                         # (B, 768)

        return self.classifier(pooled)


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT 3 — CLIP fine-tuned, focal loss + augmentation
# ══════════════════════════════════════════════════════════════════════════════

class CLIPFocalModel(nn.Module):
    """
    Variant 3: Fine-tuned CLIP with focal loss and data augmentation.

    Uses OpenAI's CLIP (ViT-B/32) dual-encoder backbone, pretrained on
    400M image-text pairs with a contrastive objective.  Fine-tuning
    CLIP end-to-end on SLAKE specialises the shared embedding space for
    medical VQA.

    Vision path  : CLIP image encoder -> 512-d normalised embedding.
    Language path: CLIP text encoder  -> 512-d normalised embedding.
    Fusion       : Concatenate (1024-d) -> MLP classifier.

    Learning strategy differences from Variants 1 & 2:
      - Focal loss (gamma=2) to down-weight the dominant "yes"/"no"
        answers and focus training on rarer medical answers.
      - Image augmentation (flip, colour jitter, random crop) applied
        during training to reduce overfitting on the small SLAKE set.

    Args:
        num_classes: Number of answer classes.
        clip_name:   HuggingFace model identifier for CLIP.
    """

    def __init__(
        self,
        num_classes: int,
        clip_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        d = self.clip.config.projection_dim   # 512

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # Concatenate normalised image and text projections
        z = torch.cat([out.image_embeds, out.text_embeds], dim=-1)  # (B, 1024)
        return self.classifier(z)


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS  (used by Variant 3)
# ══════════════════════════════════════════════════════════════════════════════

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Focal Loss for imbalanced multi-class classification.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Down-weights easy, common answers (e.g. "yes", "no") so that the
    model concentrates gradient on harder, rarer medical answers such as
    specific organ names or numerical quantities.

    When gamma=0 this reduces exactly to standard cross-entropy.
    gamma=2 is the value recommended by Lin et al. (2017).

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017.

    Args:
        logits:  Raw model outputs of shape (B, num_classes).
        targets: Ground truth class indices of shape (B,).
        gamma:   Focusing parameter.  Higher values down-weight easy
                 examples more aggressively.
        weight:  Optional per-class loss weights (torch.Tensor).

    Returns:
        Scalar mean focal loss.
    """
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()
