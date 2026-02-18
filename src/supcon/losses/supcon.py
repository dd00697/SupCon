from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [N, D]")
        if labels.ndim != 1:
            raise ValueError("labels must have shape [N]")

        n = features.shape[0]
        if labels.shape[0] != n:
            raise ValueError("features and labels must have same first dimension")

        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        device = features.device
        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(dtype=features.dtype, device=device)

        self_mask = torch.eye(n, device=device, dtype=features.dtype)
        logits_mask = 1.0 - self_mask
        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_counts = positive_mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts

        return -mean_log_prob_pos.mean()