import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import LogisticLoss, MarginLoss
from data_helper import BatchType
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class ins_model_rotate(nn.Module):
    def __init__(self, ent_emb, rel_emb, l1, margin_ins, embedding_range):
        super(ins_model_rotate, self).__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.embedding_range = embedding_range
        self.loss = LogisticLoss(margin_ins)
        self.l1 = l1
        self.gamma = nn.Parameter(
            torch.Tensor([margin_ins]),
            requires_grad=False
        )
        self.parameter_list = {
            'ent': self.ent_emb,
            'rel': self.rel_emb
        }

    def forward(self, h, r, t, batch_type):
        h_e, r_e_r, r_e_i, t_e = self.embed(h, r, t, batch_type)
        h_e_r, h_e_i = torch.chunk(h_e, 2, dim=2)
        t_e_r, t_e_i = torch.chunk(t_e, 2, dim=2)
        if batch_type == BatchType.HEAD_BATCH:
            score_r = r_e_r * t_e_r + r_e_i * t_e_i
            score_i = r_e_r * t_e_i - r_e_i * t_e_r
            score_r = score_r - h_e_r
            score_i = score_i - h_e_i
        else:
            score_r = h_e_r * r_e_r - h_e_i * r_e_i
            score_i = h_e_r * r_e_i + h_e_i * r_e_r
            score_r = score_r - t_e_r
            score_i = score_i - t_e_i
        score = torch.stack([score_r, score_i], dim=0)
        score = score.norm(dim=0)
        score = -(self.gamma.item() - score.sum(dim=2))
        return score

    def embed(self, h, r, t, batch_type):
        pi = 3.14159265358979323846
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            batch_size, negative_sample_size = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(batch_size, negative_sample_size, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            batch_size, negative_sample_size = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(batch_size, negative_sample_size, -1)
        r_e_r = r_e / (self.embedding_range / pi)
        r_e_i = torch.sin(r_e_r)
        r_e_r = torch.cos(r_e_r)
        return h_e, r_e_r, r_e_i, t_e


class ins_model_transe(nn.Module):
    def __init__(self, ent_emb, rel_emb, l1, margin_ins):
        super(ins_model_transe, self).__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.loss = MarginLoss(margin_ins)
        self.l1 = l1
        self.parameter_list = {
            'ent': self.ent_emb,
            'rel': self.rel_emb
        }

    def forward(self, h, r, t, batch_type):
        h_e, r_e, t_e = self.embed(h, r, t, batch_type)
        norm_h_e = F.normalize(h_e, p=2, dim=-1)
        norm_r_e = F.normalize(r_e, p=2, dim=-1)
        norm_t_e = F.normalize(t_e, p=2, dim=-1)
        score = torch.norm(norm_h_e + norm_r_e - norm_t_e, p=self.l1, dim=-1)
        return score

    def embed(self, h, r, t, batch_type):
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            batch_size, negative_sample_size = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(batch_size, negative_sample_size, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            batch_size, negative_sample_size = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(batch_size, negative_sample_size, -1)
        return h_e, r_e, t_e


class type_model_transe(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_type):
        super(type_model_transe, self).__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        self.loss = MarginLoss(margin_type)
        self.l1 = l1
        self.parameter_list = {
            'ent': self.ent_emb,
            'type': self.type_emb
        }

    def forward(self, ent, ent_type, batch_type):
        ent_e, type_e = self.embed(ent, ent_type, batch_type)
        norm_ent_e = F.normalize(ent_e, p=2, dim=-1)
        norm_type_e = F.normalize(type_e, p=2, dim=-1)
        l2_norm = torch.norm(norm_ent_e - norm_type_e, self.l1, -1)
        return l2_norm

    def embed(self, ent, type, batch_type):
        if batch_type == BatchType.SINGLE:
            ent_e = self.ent_emb(ent).unsqueeze(1)
            type_e = self.type_emb(type).unsqueeze(1)
        else:
            batch_size, negative_sample_size = type.size(0), type.size(1)
            ent_e = self.ent_emb(ent).unsqueeze(1)
            type_e = self.type_emb(type.view(-1)).view(batch_size, negative_sample_size, -1)
        return ent_e, type_e


class type_model_rotate(nn.Module):
    def __init__(self, ent_emb, type_emb,
                 l1, margin_pair):
        super(type_model_rotate, self).__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        self.gamma = nn.Parameter(
            torch.Tensor([margin_pair]),
            requires_grad=False
        )
        self.loss = LogisticLoss(margin_pair)
        self.l1 = l1
        self.parameter_list = {
            'ent': self.ent_emb,
            'type': self.type_emb
        }

    def forward(self, ent, type, batch_type):
        e, t = self.embed(ent, type, batch_type)
        e_r, e_i = torch.chunk(e, 2, dim=2)
        t_r, t_i = torch.chunk(t, 2, dim=2)
        score_r = e_r - t_r
        score_i = e_i - t_i
        score = torch.stack([score_r, score_i], dim=0)
        score = score.norm(dim=0)
        score = -(self.gamma.item() - score.sum(dim=2))
        return score

    def embed(self, ent, type, batch_type):
        if batch_type == BatchType.SINGLE:
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(type).unsqueeze(1)
        else:
            batch_size, negative_sample_size = type.size(0), type.size(1)
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(type.view(-1)).view(batch_size, negative_sample_size, -1)
        return e, t


class HAKE_ins(nn.Module):
    def __init__(self, ent_emb, rel_emb, l1, margin_ins, embedding_range, hidden_dim):
        super(HAKE_ins, self).__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.embedding_range = nn.Parameter(
            torch.Tensor([embedding_range]),
            requires_grad=False
        )
        self.loss = LogisticLoss(margin_ins)
        self.l1 = l1

        self.gamma = nn.Parameter(
            torch.Tensor([margin_ins]),
            requires_grad=False
        )
        self.parameter_list = {
            'ent': self.ent_emb,
            'rel': self.rel_emb
        }
        nn.init.ones_(tensor=self.rel_emb.weight[:, hidden_dim:2 * hidden_dim])
        nn.init.zeros_(tensor=self.rel_emb.weight[:, 2 * hidden_dim:3 * hidden_dim])
        phase_weight = 0.5
        modulus_weight = 1.0
        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))
        self.pi = 3.14159265358979323846

    def forward(self, h, r, t, batch_type):
        head, rel, tail = self.embed(h, r, t, batch_type)
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)
        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)
        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail
        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        mod_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        mod_score = torch.norm(mod_score, dim=2) * self.modulus_weight
        return -(self.gamma.item() - (phase_score + mod_score))

    def embed(self, h, r, t, batch_type):
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            batch_size, negative_sample_size = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(batch_size, negative_sample_size, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            batch_size, negative_sample_size = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(batch_size, negative_sample_size, -1)
        return h_e, r_e, t_e

class HAKE_type(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_pair, embedding_range, embedding_range_type):
        super(HAKE_type, self).__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        self.loss = LogisticLoss(margin_pair)
        self.l1 = l1
        self.gamma = nn.Parameter(
            torch.Tensor([margin_pair]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([embedding_range]),
            requires_grad=False
        )
        self.embedding_range_type = nn.Parameter(
            torch.Tensor([embedding_range_type]),
            requires_grad=False
        )
        self.parameter_list = {
            'ent': self.ent_emb,
            'type': self.type_emb,
        }
        phase_weight = 0.5
        modulus_weight = 1.0
        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))
        self.pi = 3.14159265358979323846

    def forward(self, ent, type, batch_type):
        e, t = self.embed(ent, type, batch_type)
        phase_ent, mod_ent = torch.chunk(e, 2, dim=2)
        phase_type, mod_type = torch.chunk(t, 2, dim=2)
        phase_ent = phase_ent / (self.embedding_range.item() / self.pi)
        phase_type = phase_type / (self.embedding_range_type.item() / self.pi)
        mod_score = mod_ent - mod_type
        mod_score = torch.norm(mod_score, dim=2) * self.modulus_weight
        phase_score = phase_ent - phase_type
        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        return -(self.gamma.item() - (phase_score + mod_score))

    def embed(self, ent, type, batch_type):
        if batch_type == BatchType.SINGLE:
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(type).unsqueeze(1)
        else:
            batch_size, negative_sample_size = type.size(0), type.size(1)
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(type.view(-1)).view(batch_size, negative_sample_size, -1)
        return e, t


class ins_model_transD(nn.Module):
    def __init__(self, ent_emb, rel_emb, l1, margin_ins, hidden_dim):
        super(ins_model_transD, self).__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.rel_proj = nn.Linear(rel_emb.size(1), hidden_dim)  # Projection matrix for relations
        self.loss = MarginLoss(margin_ins)
        self.l1 = l1
        self.hidden_dim = hidden_dim
        self.parameter_list = {
            'ent': self.ent_emb,
            'rel': self.rel_emb,
            'proj': self.rel_proj
        }

    def forward(self, h, r, t, batch_type):
        h_e, r_e, t_e = self.embed(h, r, t, batch_type)
        score = self.calculate_score(h_e, r_e, t_e)
        return score

    def calculate_score(self, h_e, r_e, t_e):
        # TransD scoring function
        norm_h_e = F.normalize(h_e, p=2, dim=-1)
        norm_r_e = F.normalize(r_e, p=2, dim=-1)
        norm_t_e = F.normalize(t_e, p=2, dim=-1)
        score = torch.norm(norm_h_e + self.rel_proj(r_e) - norm_t_e, p=self.l1, dim=-1)
        return score

    def embed(self, h, r, t, batch_type):
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            batch_size, negative_sample_size = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(batch_size, negative_sample_size, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            batch_size, negative_sample_size = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(batch_size, negative_sample_size, -1)
        return h_e, r_e, t_e

class type_model_transD(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_type, dim):
        super(type_model_transD, self).__init__()
        self.ent_emb = ent_emb  # Entity embedding matrix
        self.type_emb = type_emb  # Type embedding matrix
        self.loss = MarginLoss(margin_type)
        self.l1 = l1
        self.dim = dim  # The type-specific dimension
        self.parameter_list = {
            'ent': self.ent_emb,
            'type': self.type_emb
        }

    def forward(self, ent, ent_type, batch_type):
        ent_e, type_e = self.embed(ent, ent_type, batch_type)
        norm_ent_e = F.normalize(ent_e, p=2, dim=-1)
        norm_type_e = F.normalize(type_e, p=2, dim=-1)
        l2_norm = torch.norm(norm_ent_e - norm_type_e, self.l1, -1)
        return l2_norm

    def embed(self, ent, type, batch_type):
        if batch_type == BatchType.SINGLE:
            ent_e = self.ent_emb(ent).unsqueeze(1)
            type_e = self.type_emb(type).unsqueeze(1)
        else:
            batch_size, negative_sample_size = type.size(0), type.size(1)
            ent_e = self.ent_emb(ent).unsqueeze(1)
            type_e = self.type_emb(type.view(-1)).view(batch_size, negative_sample_size, -1)

        # TransD: Add type-specific dimension
        ent_e = ent_e + self.type_emb(type).unsqueeze(0).repeat(ent_e.size(0), 1, 1)  # Add type vector to entity

        return ent_e, type_e


class CompoundE_ins(nn.Module):
    def __init__(self, ent_emb, rel_emb, l1, margin_ins, embedding_range):
        super(CompoundE_ins, self).__init__()
        self.loss = LogisticLoss(margin_ins)
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.embedding_range = embedding_range
        self.loss = LogisticLoss(margin_ins)
        self.l1 = l1
        self.gamma = nn.Parameter(
            torch.Tensor([margin_ins]),
            requires_grad=False
        )
        self.parameter_list = {
            'ent': self.ent_emb,
            'rel': self.rel_emb
        }

    def forward(self, h, r, t, batch_type):
        head, rel, tail = self.embed(h, r, t, batch_type)

        tail_scale, tail_translate, theta = torch.chunk(rel, 3, dim=2)
        theta, _ = torch.chunk(theta, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        pi = 3.14159265358979323846

        theta = theta / (self.embedding_range / pi)

        re_rotation = torch.cos(theta)
        im_rotation = torch.sin(theta)

        re_rotation = re_rotation.unsqueeze(-1)
        im_rotation = im_rotation.unsqueeze(-1)

        tail = tail.view((tail.shape[0], tail.shape[1], -1, 2))

        tail_r = torch.cat((re_rotation * tail[:, :, :, 0:1], im_rotation * tail[:, :, :, 0:1]), dim=-1)
        tail_r += torch.cat((-im_rotation * tail[:, :, :, 1:], re_rotation * tail[:, :, :, 1:]), dim=-1)

        tail_r = tail_r.view((tail_r.shape[0], tail_r.shape[1], -1))

        tail_r += tail_translate
        tail_r *= tail_scale

        score = head - tail_r
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def embed(self, h, r, t, batch_type):
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            batch_size, negative_sample_size = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(batch_size, negative_sample_size, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            batch_size, negative_sample_size = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(batch_size, negative_sample_size, -1)
        return h_e, r_e, t_e

class CompoundE_type(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_pair, embedding_range, embedding_range_type):
        super(CompoundE_type, self).__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        self.l1 = l1
        self.margin_pair = margin_pair
        self.embedding_range = nn.Parameter(
            torch.Tensor([embedding_range]),
            requires_grad=False
        )
        self.embedding_range_type = nn.Parameter(
            torch.Tensor([embedding_range_type]),
            requires_grad=False
        )
        self.gamma = nn.Parameter(
            torch.Tensor([margin_pair]),
            requires_grad=False
        )
        self.modulus_weight = nn.Parameter(torch.Tensor([1.0]))
        self.phase_weight = nn.Parameter(torch.Tensor([0.5]))
        self.pi = 3.14159265358979323846

    def forward(self, ent, type, batch_type):
        e, t = self.embed(ent, type, batch_type)
        # Split the embedding into real and imaginary parts for entities and types
        real_ent, imag_ent = torch.chunk(e, 2, dim=2)
        real_type, imag_type = torch.chunk(t, 2, dim=2)

        # Normalize the embeddings
        real_ent = real_ent / (self.embedding_range.item() / self.pi)
        imag_ent = imag_ent / (self.embedding_range.item() / self.pi)
        real_type = real_type / (self.embedding_range_type.item() / self.pi)
        imag_type = imag_type / (self.embedding_range_type.item() / self.pi)

        # Calculate the modulus score
        mod_score = (real_ent - real_type) ** 2 + (imag_ent - imag_type) ** 2
        mod_score = torch.sqrt(mod_score) * self.modulus_weight

        # Calculate the phase score
        phase_score = torch.cos(real_ent - real_type) * torch.cos(imag_ent - imag_type)
        phase_score = torch.sum(phase_score, dim=2) * self.phase_weight

        # Combine the scores
        score = -(self.gamma.item() - (phase_score + mod_score))
        return score

    def embed(self, ent, type, batch_type):
        if batch_type == 0:  # Assuming 0 represents 'single' batch type
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(type).unsqueeze(1)
        else:
            # 检查type张量的维度，如果小于2，则增加一个维度
            if type.dim() < 2:
                type = type.unsqueeze(1)  # 增加一个维度以表示负样本大小为1
            # 获取批次大小和负样本大小
            batch_size, negative_sample_size = type.size(0), type.size(1)
            e = self.ent_emb(ent).unsqueeze(1)
            # 如果负样本大小大于1，需要展开type张量以匹配每个实体的负样本
            if negative_sample_size > 1:
                t = self.type_emb(type.view(-1)).view(batch_size, negative_sample_size, -1)
            else:
                t = self.type_emb(type).unsqueeze(1)  # 负样本大小为1，直接增加一个维度
        return e, t

class ins_model_DistMult(nn.Module):
    def __init__(self, ent_emb, rel_emb, l1, margin_ins):
        super(ins_model_DistMult, self).__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.loss = LogisticLoss(margin_ins)
        self.l1 = l1
        self.parameter_list = {
            'ent': self.ent_emb,
            'rel': self.rel_emb
        }

    def forward(self, h, r, t, batch_type):
        h_e, r_e, t_e = self.embed(h, r, t, batch_type)
        # DistMult评分函数: sum(h * r * t)
        score = torch.sum(h_e * r_e * t_e, dim=2)
        return score

    def embed(self, h, r, t, batch_type):
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            batch_size, negative_sample_size = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(batch_size, negative_sample_size, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            batch_size, negative_sample_size = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(batch_size, negative_sample_size, -1)
        return h_e, r_e, t_e


class type_model_DistMult(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_pair):
        super(type_model_DistMult, self).__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        self.loss = LogisticLoss(margin_pair)
        self.l1 = l1
        self.parameter_list = {
            'ent': self.ent_emb,
            'type': self.type_emb
        }

    def forward(self, ent, ent_type, batch_type):
        e, t = self.embed(ent, ent_type, batch_type)
        # 类型模型评分函数: sum(e * t)
        score = torch.sum(e * t, dim=2)
        return score

    def embed(self, ent, type, batch_type):
        if batch_type == BatchType.SINGLE:
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(type).unsqueeze(1)
        else:
            batch_size, negative_sample_size = type.size(0), type.size(1)
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(type.view(-1)).view(batch_size, negative_sample_size, -1)
        return e, t


class ins_model_transD(nn.Module):
    def __init__(self, ent_emb, rel_emb, l1, margin_ins, hidden_dim):
        super(ins_model_transD, self).__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        # 实体和关系的投影矩阵
        self.ent_proj = nn.Embedding(ent_emb.num_embeddings, hidden_dim)
        self.rel_proj = nn.Embedding(rel_emb.num_embeddings, hidden_dim)

        # 初始化投影矩阵
        nn.init.xavier_uniform_(self.ent_proj.weight)
        nn.init.xavier_uniform_(self.rel_proj.weight)

        self.loss = MarginLoss(margin_ins)
        self.l1 = l1
        self.hidden_dim = hidden_dim
        self.parameter_list = {
            'ent': self.ent_emb,
            'rel': self.rel_emb,
            'ent_proj': self.ent_proj,
            'rel_proj': self.rel_proj
        }

    def forward(self, h, r, t, batch_type):
        h_e, r_e, t_e = self.embed(h, r, t, batch_type)
        # 计算投影后的嵌入 - 使用高效实现
        h_p = self.project(h, h_e, self.ent_proj)
        r_p = self.project(r, r_e, self.rel_proj)
        t_p = self.project(t, t_e, self.ent_proj)

        # 计算得分
        score = torch.norm(h_p + r_p - t_p, p=self.l1, dim=-1)
        return score

    def project(self, indices, embeddings, proj_layer):
        # 获取投影向量
        proj_vec = proj_layer(indices)
        # 高效计算投影: h_p = h + (w_h · h) * w_r
        # 避免构建大矩阵
        dot_product = (proj_vec * embeddings).sum(dim=-1, keepdim=True)
        projected = embeddings + dot_product * proj_vec
        return projected

    def embed(self, h, r, t, batch_type):
        # 保持原有实现
        if batch_type == BatchType.SINGLE:
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            batch_size, negative_sample_size = h.size(0), h.size(1)
            h_e = self.ent_emb(h).view(batch_size, negative_sample_size, -1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).unsqueeze(1)
        elif batch_type == BatchType.TAIL_BATCH:
            batch_size, negative_sample_size = t.size(0), t.size(1)
            h_e = self.ent_emb(h).unsqueeze(1)
            r_e = self.rel_emb(r).unsqueeze(1)
            t_e = self.ent_emb(t).view(batch_size, negative_sample_size, -1)
        return h_e, r_e, t_e


class type_model_transD(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_pair, hidden_dim):
        super(type_model_transD, self).__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        # 实体和类型的投影矩阵
        self.ent_proj = nn.Embedding(ent_emb.num_embeddings, hidden_dim)
        self.type_proj = nn.Embedding(type_emb.num_embeddings, hidden_dim)

        # 初始化投影矩阵
        nn.init.xavier_uniform_(self.ent_proj.weight)
        nn.init.xavier_uniform_(self.type_proj.weight)

        self.loss = MarginLoss(margin_pair)
        self.l1 = l1
        self.hidden_dim = hidden_dim
        self.parameter_list = {
            'ent': self.ent_emb,
            'type': self.type_emb,
            'ent_proj': self.ent_proj,
            'type_proj': self.type_proj
        }

    def forward(self, ent, ent_type, batch_type):
        e, t = self.embed(ent, ent_type, batch_type)
        # 计算投影后的嵌入 - 使用高效实现
        e_p = self.project(ent, e, self.ent_proj)
        t_p = self.project(ent_type, t, self.type_proj)

        # 计算得分
        score = torch.norm(e_p - t_p, p=self.l1, dim=-1)
        return score

    def project(self, indices, embeddings, proj_layer):
        # 获取投影向量
        proj_vec = proj_layer(indices)
        # 高效计算投影: e_p = e + (w_e · e) * w_t
        # 避免构建大矩阵
        dot_product = (proj_vec * embeddings).sum(dim=-1, keepdim=True)
        projected = embeddings + dot_product * proj_vec
        return projected

    def embed(self, ent, type, batch_type):
        # 保持原有实现
        if batch_type == BatchType.SINGLE:
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(type).unsqueeze(1)
        else:
            batch_size, negative_sample_size = type.size(0), type.size(1)
            e = self.ent_emb(ent).unsqueeze(1)
            t = self.type_emb(type.view(-1)).view(batch_size, negative_sample_size, -1)
        return e, t


class ins_model_ComplEx(nn.Module):
    def __init__(self, ent_emb, rel_emb, l1, margin_ins):
        super(ins_model_ComplEx, self).__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.loss = LogisticLoss(margin_ins)
        self.l1 = l1
        self.parameter_list = {
            'ent': self.ent_emb,
            'rel': self.rel_emb
        }

    def forward(self, h, r, t, batch_type):
        # 根据batch_type调整输入维度
        if batch_type == BatchType.SINGLE:
            h = h.unsqueeze(0) if h.dim() == 0 else h
            r = r.unsqueeze(0) if r.dim() == 0 else r
            t = t.unsqueeze(0) if t.dim() == 0 else t

        h_real, h_imag = self.get_complex_embeddings(h, self.ent_emb, batch_type)
        r_real, r_imag = self.get_complex_embeddings(r, self.rel_emb, batch_type)
        t_real, t_imag = self.get_complex_embeddings(t, self.ent_emb, batch_type)

        # ComplEx评分函数: Re(<h, r, conjugate(t)>)
        score_real = h_real * r_real * t_real + h_real * r_imag * t_imag + \
                     h_imag * r_real * t_imag - h_imag * r_imag * t_real
        score = torch.sum(score_real, dim=-1)
        return score

    def get_complex_embeddings(self, indices, embedding_layer, batch_type):
        """获取复数嵌入的实部和虚部"""
        # 确保索引至少有一维
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        # 获取嵌入
        if batch_type == BatchType.SINGLE:
            emb = embedding_layer(indices).unsqueeze(1)
        elif batch_type == BatchType.HEAD_BATCH:
            if indices.dim() == 1:
                # 关系索引 (batch_size)
                emb = embedding_layer(indices).unsqueeze(1)
            else:
                # 头实体索引 (batch_size, negative_sample_size)
                emb = embedding_layer(indices.view(-1)).view(indices.size(0), indices.size(1), -1)
        elif batch_type == BatchType.TAIL_BATCH:
            if indices.dim() == 1:
                # 关系索引 (batch_size)
                emb = embedding_layer(indices).unsqueeze(1)
            else:
                # 尾实体索引 (batch_size, negative_sample_size)
                emb = embedding_layer(indices.view(-1)).view(indices.size(0), indices.size(1), -1)

        # 将嵌入拆分为实部和虚部
        dim = emb.shape[-1] // 2
        real = emb[..., :dim]
        imag = emb[..., dim:]
        return real, imag


class type_model_ComplEx(nn.Module):
    def __init__(self, ent_emb, type_emb, l1, margin_pair):
        super(type_model_ComplEx, self).__init__()
        self.ent_emb = ent_emb
        self.type_emb = type_emb
        self.loss = LogisticLoss(margin_pair)
        self.l1 = l1
        self.parameter_list = {
            'ent': self.ent_emb,
            'type': self.type_emb
        }

    def forward(self, ent, ent_type, batch_type):
        # 根据batch_type调整输入维度
        if batch_type == BatchType.SINGLE:
            ent = ent.unsqueeze(0) if ent.dim() == 0 else ent
            ent_type = ent_type.unsqueeze(0) if ent_type.dim() == 0 else ent_type

        e_real, e_imag = self.get_complex_embeddings(ent, self.ent_emb, batch_type)
        t_real, t_imag = self.get_complex_embeddings(ent_type, self.type_emb, batch_type)

        # 类型模型评分函数: Re(<e, conjugate(t)>)
        score_real = e_real * t_real + e_imag * t_imag
        score = torch.sum(score_real, dim=-1)
        return score

    def get_complex_embeddings(self, indices, embedding_layer, batch_type):
        """获取复数嵌入的实部和虚部"""
        # 确保索引至少有一维
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        # 获取嵌入
        if batch_type == BatchType.SINGLE:
            emb = embedding_layer(indices).unsqueeze(1)
        else:
            if indices.dim() == 1:
                # 单个实体或类型
                emb = embedding_layer(indices).unsqueeze(1)
            else:
                # 类型索引 (batch_size, negative_sample_size)
                emb = embedding_layer(indices.view(-1)).view(indices.size(0), indices.size(1), -1)

        # 将嵌入拆分为实部和虚部
        dim = emb.shape[-1] // 2
        real = emb[..., :dim]
        imag = emb[..., dim:]
        return real, imag