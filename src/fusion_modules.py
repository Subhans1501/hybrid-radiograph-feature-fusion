import torch
import torch.nn as nn
import torch.nn.functional as F


class PureDeepHead(nn.Module):
    def __init__(self, deep_dim: int, embed_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(deep_dim),
            nn.Dropout(dropout),
            nn.Linear(deep_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, deep_feat: torch.Tensor, hand_feat: torch.Tensor = None):
        emb = self.proj(deep_feat)
        return self.classifier(emb), emb, None


class ProjectedConcatFusion(nn.Module):
    """Projects deep and handcrafted vectors into L2-normalized 256-D space before concatenation."""

    def __init__(self, deep_dim: int, hand_dim: int, embed_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.deep_proj = nn.Sequential(
            nn.LayerNorm(deep_dim), nn.Dropout(dropout),
            nn.Linear(deep_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.hand_proj = nn.Sequential(
            nn.LayerNorm(hand_dim), nn.Dropout(dropout),
            nn.Linear(hand_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, deep_feat: torch.Tensor, hand_feat: torch.Tensor):
        scale = (256 ** 0.5) * 0.25
        f_d = F.normalize(self.deep_proj(deep_feat), p=2, dim=-1) * scale
        f_h = F.normalize(self.hand_proj(hand_feat), p=2, dim=-1) * scale
        emb = self.fusion_mlp(torch.cat([f_d, f_h], dim=-1)) + f_d
        return self.classifier(emb), emb, None


class GatedCrossModalFusion(nn.Module):
    """Per-channel sigmoid gate balancing handcrafted texture cues and bilinear interactions."""

    def __init__(self, deep_dim: int, hand_dim: int, embed_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.deep_proj = nn.Sequential(
            nn.LayerNorm(deep_dim), nn.Dropout(dropout),
            nn.Linear(deep_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.hand_proj = nn.Sequential(
            nn.LayerNorm(hand_dim), nn.Dropout(dropout),
            nn.Linear(hand_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.gate_net = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.interaction_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.out_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(embed_dim, num_classes))

    def forward(self, deep_feat: torch.Tensor, hand_feat: torch.Tensor):
        f_d = self.deep_proj(deep_feat)
        f_h = self.hand_proj(hand_feat)
        alpha = self.gate_net(torch.cat([f_d, f_h], dim=-1))
        interaction = self.interaction_proj(f_d * f_h)
        emb = self.out_norm(f_d + alpha * f_h + (1.0 - alpha) * 0.25 * interaction)
        return self.classifier(emb), emb, alpha.mean(dim=-1)


class CrossModalTokenAttentionFusion(nn.Module):
    """4-head self-attention across [deep, handcrafted] modality tokens + SE channel gating."""

    def __init__(self, deep_dim: int, hand_dim: int, embed_dim: int = 256,
                 num_heads: int = 4, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.deep_proj = nn.Sequential(
            nn.LayerNorm(deep_dim), nn.Dropout(dropout),
            nn.Linear(deep_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.hand_proj = nn.Sequential(
            nn.LayerNorm(hand_dim), nn.Dropout(dropout),
            nn.Linear(hand_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.mod_pos = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.fuse_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.se = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim), nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(embed_dim, num_classes))

    def forward(self, deep_feat: torch.Tensor, hand_feat: torch.Tensor):
        f_d = self.deep_proj(deep_feat)
        f_h = self.hand_proj(hand_feat)
        tokens = torch.stack([f_d, f_h], dim=1) + self.mod_pos
        normed = self.norm1(tokens)
        attn_out, attn_weights = self.mhsa(normed, normed, normed, need_weights=True, average_attn_weights=True)
        tokens = tokens + attn_out
        joined = self.fuse_proj(tokens.reshape(tokens.size(0), -1))
        emb = self.out_norm(f_d + joined * self.se(joined))
        # Weight that deep token places on handcrafted token
        return self.classifier(emb), emb, attn_weights[:, 0, 1]
