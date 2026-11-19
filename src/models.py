import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.fusion_modules import (
    PureDeepHead,
    ProjectedConcatFusion,
    GatedCrossModalFusion,
    CrossModalTokenAttentionFusion,
)


class HybridMedicalClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        fusion_type: str = "none",
        hand_dim: int = 0,
        embed_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.fusion_type = fusion_type.lower()
        self.hand_dim = hand_dim

        if self.backbone_name == "convnext_tiny":
            weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            base = tv_models.convnext_tiny(weights=weights)
            self.frozen_stages = base.features[:6]
            self.trainable_stage = base.features[6:]
            self.post_norm = nn.Identity()
            self.avgpool = base.avgpool
            self.deep_dim = 768
        elif self.backbone_name == "swin_t":
            weights = tv_models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            base = tv_models.swin_t(weights=weights)
            self.frozen_stages = base.features[:6]
            self.trainable_stage = base.features[6:]
            self.post_norm = base.norm
            self.avgpool = base.avgpool
            self.deep_dim = 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        for p in self.frozen_stages.parameters():
            p.requires_grad = False
        self.frozen_stages.eval()

        if self.fusion_type == "none":
            self.head = PureDeepHead(self.deep_dim, embed_dim, num_classes, dropout)
        elif self.fusion_type == "projected_concat":
            self.head = ProjectedConcatFusion(self.deep_dim, hand_dim, embed_dim, num_classes, dropout)
        elif self.fusion_type == "gated_attention":
            self.head = GatedCrossModalFusion(self.deep_dim, hand_dim, embed_dim, num_classes, dropout)
        elif self.fusion_type == "token_attention":
            self.head = CrossModalTokenAttentionFusion(
                self.deep_dim, hand_dim, embed_dim, num_heads=4, num_classes=num_classes, dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen low-level stages in eval mode during training
        self.frozen_stages.eval()
        return self

    def extract_deep_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.frozen_stages(x)
        h = self.trainable_stage(h)
        if self.backbone_name == "swin_t":
            h = self.post_norm(h).permute(0, 3, 1, 2)
        return torch.flatten(self.avgpool(h), 1)

    def forward(self, x: torch.Tensor, hand_feat: torch.Tensor = None):
        return self.head(self.extract_deep_features(x), hand_feat)

    def get_param_groups(self, backbone_lr: float = 1e-4, head_lr: float = 5e-4, weight_decay: float = 1e-2):
        backbone_params = [
            p for p in list(self.trainable_stage.parameters()) + list(self.post_norm.parameters())
            if p.requires_grad
        ]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
