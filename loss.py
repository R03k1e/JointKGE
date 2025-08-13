from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module, ABC):
    def __init__(self, margin=2.5):
        super().__init__()
        self.margin = margin
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class MarginLoss(Loss):
    def __init__(self, margin=0.0):
        super().__init__(margin = margin)
    def forward(self, p_score, n_score, alpha, subsampling_weight):
        loss = p_score + self.margin - n_score
        loss = torch.max(loss, torch.zeros_like(loss)).sum()
        p_loss = p_score.sum()
        n_loss = n_score.sum()
        return p_loss, n_loss, loss


class LogisticLoss(Loss):
    def __init__(self, margin=0.0):
        super().__init__(margin = margin)
    def forward(self, p_score, n_score, alpha, subsampling_weight):
        p_score = -p_score
        n_score = -n_score
        p_score = F.logsigmoid(p_score).squeeze(dim = 1)
        softmax = nn.Softmax(dim=1)(n_score * alpha).detach()
        n_score = torch.sum(softmax * (F.logsigmoid(-n_score)), dim=-1)
        p_loss = (subsampling_weight * p_score).sum() / subsampling_weight.sum()
        n_loss = (subsampling_weight * n_score).sum() / subsampling_weight.sum()
        loss = (-p_loss.mean()-n_loss.mean())/2
        return p_loss, n_loss, loss
