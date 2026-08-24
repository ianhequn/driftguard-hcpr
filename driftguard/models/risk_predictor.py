"""Small task-risk predictors for Round 2."""

from __future__ import annotations

import torch
from torch import nn


class TinyRiskCNN(nn.Module):
    """Spatial residual risk model kept intentionally small for edge deployment."""

    def __init__(self, metadata_dim: int = 5):
        super().__init__()
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(24 + metadata_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, patches: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        encoded = self.patch_encoder(patches).flatten(1)
        return self.head(torch.cat([encoded, metadata], dim=1)).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
