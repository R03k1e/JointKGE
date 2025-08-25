"""
Abstract base class plus concrete implementations of loss functions
for knowledge-graph embedding models.

All losses accept:
    p_score : torch.Tensor (batch,)
        Positive triple scores.
    n_score : torch.Tensor (batch, neg)
        Negative triple scores.
    alpha   : float
        Temperature / scaling factor for softmax over negatives.
    subsampling_weight : torch.Tensor (batch,)
        Per-sample weights (e.g., for subsampling correction).
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Base loss definition
# ------------------------------------------------------------------
class Loss(nn.Module, ABC):
    """
    Abstract base loss class.

    Any concrete subclass must implement `forward`.
    """

    def __init__(self, margin: float = 2.5):
        super().__init__()
        self.margin = margin

    @abstractmethod
    def forward(self, p_score, n_score, alpha, subsampling_weight):
        """
        Compute loss.

        Returns
        -------
        p_loss : torch.Tensor
            Positive term loss (for logging).
        n_loss : torch.Tensor
            Negative term loss (for logging).
        loss   : torch.Tensor
            Scalar total loss for back-prop.
        """
        raise NotImplementedError


# ------------------------------------------------------------------
# Margin-based ranking loss
# ------------------------------------------------------------------
class MarginLoss(Loss):
    """
    Standard max-margin / hinge loss.

    loss = max(0, p_score + margin - n_score)
    """

    def __init__(self, margin: float = 0.0):
        super().__init__(margin=margin)

    def forward(self, p_score, n_score, alpha, subsampling_weight):
        # Compute per-sample margin violation
        loss = p_score + self.margin - n_score
        loss = torch.max(loss, torch.zeros_like(loss)).sum()

        # Optional logging terms
        p_loss = p_score.sum()
        n_loss = n_score.sum()

        return p_loss, n_loss, loss


# ------------------------------------------------------------------
# Softmax-negative-sampling logistic loss
# ------------------------------------------------------------------
class LogisticLoss(Loss):
    """
    Logistic loss with softmax over negative samples.

    Positive term:  -log σ(p_score)
    Negative term:  Σ_i  softmax_i * -log σ(-n_score_i)

    The negative weighting is controlled by `alpha`.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__(margin=margin)  # margin unused, kept for API consistency

    def forward(self, p_score, n_score, alpha, subsampling_weight):
        # Positive part
        p_score = -p_score
        p_log_prob = F.logsigmoid(p_score).squeeze(dim=1)

        # Negative part with softmax weights
        softmax = F.softmax(n_score * alpha, dim=1).detach()
        n_log_prob = torch.sum(softmax * F.logsigmoid(-n_score), dim=-1)

        # Weighted average via subsampling weights
        p_loss = (subsampling_weight * p_log_prob).sum() / subsampling_weight.sum()
        n_loss = (subsampling_weight * n_log_prob).sum() / subsampling_weight.sum()

        # Combine and negate (maximize log-likelihood)
        loss = (-p_loss.mean() - n_loss.mean()) / 2

        return p_loss, n_loss, loss
