"""
Knowledge-graph embedding models for both instance-level triples and type-level pairs.
Each model follows a common pattern:

    forward(self, h / ent, r / type, t / ent_type, batch_type) -> scores
    embed(...) -> embeddings  (helper to handle batch shapes)

All models inherit from nn.Module and expose:
    loss            : chosen loss function
    parameter_list  : dict(str -> nn.Parameter) for easy weight sharing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import LogisticLoss, MarginLoss
from data_helper import BatchType
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'          # debug-friendly CUDA calls


# ------------------------------------------------------------------
# RotatE on instance triples (h,r,t)
# ------------------------------------------------------------------
class ins_model_rotate(nn.Module):
    """
    RotatE for instance triples.
    Relation vectors are interpreted as rotations in complex space.
    """

    def __init__(self, ent_emb, rel_emb, l1, margin_ins, embedding_range):
        super().__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.embedding_range = embedding_range
        self.loss = LogisticLoss(margin_ins)
        self.l1 = l1
        self.gamma = nn.Parameter(torch.tensor([margin_ins]), requires_grad=False)
        self.parameter_list = {'ent': self.ent_emb, 'rel': self.rel_emb}

    # ---------- forward -------------------------------------------------
    def forward(self, h, r, t, batch_type):
        h_e, r_e_r, r_e_i, t_e = self.embed(h, r, t, batch_type)

        # Split complex dimensions
        h_e_r, h_e_i = torch.chunk(h_e, 2, dim=2)
        t_e_r, t_e_i = torch.chunk(t_e, 2, dim=2)

        # Rotate head or tail depending on corruption side
        if batch_type == BatchType.HEAD_BATCH:
            score_r = r_e_r * t_e_r + r_e_i * t_e_i
            score_i = r_e_r * t_e_i - r_e_i * t_e_r
            score_r -= h_e_r
            score_i -= h_e_i
        else:
            score_r = h_e_r * r_e_r - h_e_i * r_e_i
            score_i = h_e_r * r_e_i + h_e_i * r_e_r
            score_r -= t_e_r
            score_i -= t_e_i

        # Distance in complex plane
        score = torch.stack([score_r, score_i], dim=0).norm(dim=0).sum(dim=2)
        return -(self.gamma.item() - score)

    # ---------- embedding helper ----------------------------------------
    def embed(self, h, r, t, batch_type):
        pi = 3.14159265358979323846
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            bs, neg = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(bs, neg, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            bs, neg = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(bs, neg, -1)

        # Phase in radians
        r_e_r = r_e / (self.embedding_range / pi)
        r_e_i = torch.sin(r_e_r)
        r_e_r = torch.cos(r_e_r)
        return h_e, r_e_r, r_e_i, t_e


# ------------------------------------------------------------------
# TransE on instance triples
# ------------------------------------------------------------------
class ins_model_transe(nn.Module):
    def __init__(self, ent_emb, rel_emb, l1, margin_ins):
        super().__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.loss = MarginLoss(margin_ins)
        self.l1 = l1
        self.parameter_list = {'ent': self.ent_emb, 'rel': self.rel_emb}

    def forward(self, h, r, t, batch_type):
        h_e, r_e, t_e = self.embed(h, r, t, batch_type)

        # L2-normalize embeddings (common practice)
        h_e = F.normalize(h_e, p=2, dim=-1)
        r_e = F.normalize(r_e, p=2, dim=-1)
        t_e = F.normalize(t_e, p=2, dim=-1)

        score = torch.norm(h_e + r_e - t_e, p=self.l1, dim=-1)
        return score

    def embed(self, h, r, t, batch_type):
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            bs, neg = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(bs, neg, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            bs, neg = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(bs, neg, -1)
        return h_e, r_e, t_e


# ------------------------------------------------------------------
# TransE on entity-type pairs
# ------------------------------------------------------------------
class type_model_transe(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_type):
        super().__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        self.loss = MarginLoss(margin_type)
        self.l1 = l1
        self.parameter_list = {'ent': self.ent_emb, 'type': self.type_emb}

    def forward(self, ent, ent_type, batch_type):
        ent_e, type_e = self.embed(ent, ent_type, batch_type)
        ent_e = F.normalize(ent_e, p=2, dim=-1)
        type_e = F.normalize(type_e, p=2, dim=-1)
        score = torch.norm(ent_e - type_e, p=self.l1, dim=-1)
        return score

    def embed(self, ent, ent_type, batch_type):
        if batch_type == BatchType.SINGLE:
            ent_e = self.ent_emb(ent).unsqueeze(1)
            type_e = self.type_emb(ent_type).unsqueeze(1)
        else:
            bs, neg = ent_type.size(0), ent_type.size(1)
            ent_e = self.ent_emb(ent).unsqueeze(1)
            type_e = self.type_emb(ent_type.view(-1)).view(bs, neg, -1)
        return ent_e, type_e


# ------------------------------------------------------------------
# RotatE on entity-type pairs
# ------------------------------------------------------------------
class type_model_rotate(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_pair):
        super().__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        self.gamma = nn.Parameter(torch.tensor([margin_pair]), requires_grad=False)
        self.loss = LogisticLoss(margin_pair)
        self.l1 = l1
        self.parameter_list = {'ent': self.ent_emb, 'type': self.type_emb}

    def forward(self, ent, ent_type, batch_type):
        e, t = self.embed(ent, ent_type, batch_type)
        e_r, e_i = torch.chunk(e, 2, dim=2)
        t_r, t_i = torch.chunk(t, 2, dim=2)

        # Direct difference in complex plane
        score_r = e_r - t_r
        score_i = e_i - t_i
        score = torch.stack([score_r, score_i], dim=0).norm(dim=0).sum(dim=2)
        return -(self.gamma.item() - score)

    def embed(self, ent, ent_type, batch_type):
        if batch_type == BatchType.SINGLE:
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(ent_type).unsqueeze(1)
        else:
            bs, neg = ent_type.size(0), ent_type.size(1)
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(ent_type.view(-1)).view(bs, neg, -1)
        return e, t


# ------------------------------------------------------------------
# HAKE on instance triples
# ------------------------------------------------------------------
class HAKE_ins(nn.Module):
    """
    Phase + modulus decomposition (HAKE) for instance triples.
    """

    def __init__(self, ent_emb, rel_emb, l1, margin_ins, embedding_range, hidden_dim):
        super().__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.embedding_range = nn.Parameter(torch.tensor([embedding_range]), requires_grad=False)
        self.loss = LogisticLoss(margin_ins)
        self.l1 = l1
        self.gamma = nn.Parameter(torch.tensor([margin_ins]), requires_grad=False)

        self.phase_weight = nn.Parameter(torch.tensor([[0.5 * embedding_range]]))
        self.modulus_weight = nn.Parameter(torch.tensor([[1.0]]))
        self.pi = 3.14159265358979323846

        self.parameter_list = {'ent': self.ent_emb, 'rel': self.rel_emb}

        # Special initialization for modulus & bias
        nn.init.ones_(self.rel_emb.weight[:, hidden_dim:2 * hidden_dim])
        nn.init.zeros_(self.rel_emb.weight[:, 2 * hidden_dim:3 * hidden_dim])

    def forward(self, h, r, t, batch_type):
        head, rel, tail = self.embed(h, r, t, batch_type)

        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_rel, mod_rel, bias_rel = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        # Phase in radians
        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_rel = phase_rel / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        # Composition rules
        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_rel - phase_tail)
        else:
            phase_score = (phase_head + phase_rel) - phase_tail

        # Modulus & bias
        mod_rel = torch.abs(mod_rel)
        bias_rel = torch.clamp(bias_rel, max=1)
        indicator = bias_rel < -mod_rel
        bias_rel[indicator] = -mod_rel[indicator]

        mod_score = mod_head * (mod_rel + bias_rel) - mod_tail * (1 - bias_rel)
        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        mod_score = torch.norm(mod_score, dim=2) * self.modulus_weight

        return -(self.gamma.item() - (phase_score + mod_score))

    def embed(self, h, r, t, batch_type):
        # Same shape logic as other models
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            bs, neg = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(bs, neg, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            bs, neg = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(bs, neg, -1)
        return h_e, r_e, t_e


# ------------------------------------------------------------------
# HAKE on entity-type pairs
# ------------------------------------------------------------------
class HAKE_type(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_pair, embedding_range, embedding_range_type):
        super().__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        self.loss = LogisticLoss(margin_pair)
        self.l1 = l1
        self.gamma = nn.Parameter(torch.tensor([margin_pair]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.tensor([embedding_range]), requires_grad=False)
        self.embedding_range_type = nn.Parameter(torch.tensor([embedding_range_type]), requires_grad=False)

        self.phase_weight = nn.Parameter(torch.tensor([[0.5 * embedding_range]]))
        self.modulus_weight = nn.Parameter(torch.tensor([[1.0]]))
        self.pi = 3.14159265358979323846

        self.parameter_list = {'ent': self.ent_emb, 'type': self.type_emb}

    def forward(self, ent, ent_type, batch_type):
        e, t = self.embed(ent, ent_type, batch_type)

        # Split into modulus and phase
        phase_ent, mod_ent = torch.chunk(e, 2, dim=2)
        phase_type, mod_type = torch.chunk(t, 2, dim=2)

        phase_ent = phase_ent / (self.embedding_range.item() / self.pi)
        phase_type = phase_type / (self.embedding_range_type.item() / self.pi)

        # Simple difference (no composition)
        mod_score = mod_ent - mod_type
        mod_score = torch.norm(mod_score, dim=2) * self.modulus_weight

        phase_score = phase_ent - phase_type
        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight

        return -(self.gamma.item() - (phase_score + mod_score))

    def embed(self, ent, ent_type, batch_type):
        if batch_type == BatchType.SINGLE:
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(ent_type).unsqueeze(1)
        else:
            bs, neg = ent_type.size(0), ent_type.size(1)
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(ent_type.view(-1)).view(bs, neg, -1)
        return e, t
