import logging
import torch
import torch.nn as nn
from model import ins_model_transe, ins_model_rotate, type_model_transe, type_model_rotate, HAKE_ins, HAKE_type, CompoundE_ins,CompoundE_type,ins_model_DistMult,type_model_DistMult,type_model_transD,ins_model_transD,ins_model_ComplEx,type_model_ComplEx
from optimizer import Optimizer
from utils import Monitor
from tqdm import tqdm
import numpy as np
from earlystopper import EarlyStopper
from evaluator import Evaluator
import pandas as pd
from pathlib import Path
from data_helper import BatchType

import importlib

log = logging.getLogger(__name__)


class Trainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.GeoKG = None
        self.training_results = []
        self.eval_results = []
        self.step = 0

    def gen_batch_ins(self, batch_num, batch_size, neg_rate, forever=False, shuffle=True):
        kg = self.GeoKG.triplets_withtype["train"]
        while True:
            if shuffle:
                np.random.shuffle(kg)
            for i in range(batch_num):
                start = i * batch_size
                batch = kg[start: start + batch_size]
                if self.step % 2 == 0:
                    batch_type = BatchType.HEAD_BATCH
                else:
                    batch_type = BatchType.TAIL_BATCH
                self.step += 1
                neg_batch_triplets, subsampling_weights = self.GeoKG.corrupt_batch_triplets_ins(neg_rate, batch,
                                                                                                    batch_type)
                batch_array = np.asarray([[x.h, x.r, x.t] for x in batch])
                h_batch, r_batch, t_batch = batch_array[:, 0], batch_array[:, 1], batch_array[:, 2]
                if batch_type == BatchType.HEAD_BATCH:
                    neg_h_batch, neg_r_batch, neg_t_batch = torch.stack(neg_batch_triplets), batch_array[:,
                                                                                             1], batch_array[:, 2]
                elif batch_type == BatchType.TAIL_BATCH:
                    neg_h_batch, neg_r_batch, neg_t_batch = batch_array[:, 0], batch_array[:, 1], torch.stack(
                        neg_batch_triplets)
                yield h_batch, r_batch, t_batch, neg_h_batch, neg_r_batch, neg_t_batch, subsampling_weights, batch_type
            if not forever:
                break

    def gen_batch_type(self, batch_num, batch_size, neg_rate, forever=False, shuffle=True):
        kg_type = self.GeoKG.triplets_type["train"]
        while True:
            if shuffle:
                np.random.shuffle(kg_type)
            for i in range(batch_num):
                start = i * batch_size
                batch = kg_type[start: start + batch_size]
                if self.step % 2 == 0:
                    batch_type = BatchType.HEAD_BATCH
                else:
                    batch_type = BatchType.TAIL_BATCH
                self.step += 1
                neg_batch_triplets, subsampling_weights = self.GeoKG.corrupt_batch_triplets_type(neg_rate, batch,
                                                                                                batch_type)
                batch_array_type = np.asarray([[x.h, x.r, x.t] for x in batch])
                h_batch, r_batch, t_batch = batch_array_type[:, 0], batch_array_type[:, 1], batch_array_type[:, 2]
                if batch_type == BatchType.HEAD_BATCH:
                    neg_h_batch, neg_r_batch, neg_t_batch = torch.stack(neg_batch_triplets), batch_array_type[:,
                                                                                             1], batch_array_type[:, 2]
                elif batch_type == BatchType.TAIL_BATCH:
                    neg_h_batch, neg_r_batch, neg_t_batch = batch_array_type[:, 0], batch_array_type[:, 1], torch.stack(
                        neg_batch_triplets)
                yield h_batch, r_batch, t_batch, neg_h_batch, neg_r_batch, neg_t_batch, subsampling_weights, batch_type
            if not forever:
                break

    def gen_batch_pair(self, batch_num, batch_size, neg_rate, forever=False, shuffle=True):
        kg_pair = self.GeoKG.pairs_all["train"]
        while True:
            if shuffle:
                np.random.shuffle(kg_pair)
            for i in range(batch_num):
                start = i * batch_size
                batch_pair = kg_pair[start: start + batch_size]
                batch_array_pair = np.asarray([[x.ent, x.type] for x in batch_pair])
                ent_batch, type_batch = batch_array_pair[:, 0], batch_array_pair[:, 1]
                neg_batch_pair = self.GeoKG.corrupt_batch_pairs(neg_rate, batch_pair)
                neg_ent_batch, neg_type_batch = batch_array_pair[:, 0], torch.stack(neg_batch_pair, dim=0)
                yield ent_batch, type_batch, neg_ent_batch, neg_type_batch
            if not forever:
                break

    def summary(self, args):
        summary = ["", "------------------Global Setting--------------------"]
        maxspace = len(max(args.__dict__.keys())) + 20
        for key, val in args.__dict__.items():
            if len(key) < maxspace:
                for _ in range(maxspace - len(key)):
                    key = ' ' + key
            summary.append("%s : %s" % (key, val))
        summary.append("---------------------------------------------------")
        summary.append("")
        log.info("\n".join(summary))

    def train_batch_transe(self, batch, model, optimizer, alpha, flag):
        if flag == 'ins':
            pos_pred = model(batch['h'], batch['r'], batch['t'], batch_type=BatchType.SINGLE)
            neg_pred = model(batch['neg_h'], batch['neg_r'], batch['neg_t'], batch_type=batch['batch_type'])
        elif flag == 'type':
            pos_pred = model(batch['h_type_ins'], batch['r_type_ins'], batch['t_type_ins'], batch_type=BatchType.SINGLE)
            neg_pred = model(batch['neg_h_type_ins'], batch['neg_r_type_ins'], batch['neg_t_type_ins'],
                             batch_type=batch['batch_type'])
        pos_loss, neg_loss, loss = model.loss(pos_pred, neg_pred, alpha, subsampling_weight=batch['subsampling_weight'])
        optimizer.backprop(loss)
        return pos_loss, neg_loss, loss

    def train_batch_rotate_hake(self, batch, model, optimizer, alpha, flag):
        if flag == 'ins':
            pos_pred = model(batch['h'], batch['r'], batch['t'], batch_type=BatchType.SINGLE)
            neg_pred = model(batch['neg_h'], batch['neg_r'], batch['neg_t'], batch_type=batch['batch_type'])
        elif flag == 'type':
            pos_pred = model(batch['h_type_ins'], batch['r_type_ins'], batch['t_type_ins'], batch_type=BatchType.SINGLE)
            neg_pred = model(batch['neg_h_type_ins'], batch['neg_r_type_ins'], batch['neg_t_type_ins'],
                             batch_type=batch['batch_type'])
        pos_loss, neg_loss, loss = model.loss(pos_pred, neg_pred, alpha, subsampling_weight=batch['subsampling_weight'])
        optimizer.backprop(loss)
        return pos_loss, neg_loss, loss

    def train_batch_pair(self, batch, model, optimizer, alpha):
        pos_pred = model(batch['ent_pair'], batch['type_pair'], batch_type=BatchType.SINGLE)
        neg_pred = model(batch['neg_ent_pair'], batch['neg_type_pair'], batch_type=None)
        pos_loss, neg_loss, loss = model.loss(pos_pred, neg_pred, alpha,
                                              subsampling_weight=torch.ones(pos_pred.size(dim=0)).to(self.device))
        optimizer.backprop(loss)
        return pos_loss, neg_loss, loss


class Join_trainer(Trainer):
    def __init__(self, args):
        super(Join_trainer, self).__init__()
        self.args = args
        self.method = args.method
        self.epochs = args.epochs
        self.dim_ins = args.dim_ins
        self.dim_type = args.dim_type
        self.batch_size_ins = args.batch_size_ins
        self.batch_size_type = args.batch_size_type
        self.batch_size_pair = args.batch_size_pair
        self.model_save_path = Path(args.model_save_path)
        self.l1 = args.l1
        self.margin_ins = args.margin_ins
        self.margin_type = args.margin_type
        self.margin_pair = args.margin_pair
        self.fold_ins = args.fold_ins
        self.fold_type = args.fold_type
        self.fold_pair = args.fold_pair
        self.lr = args.lr
        self.weight_G = args.weight_G
        self.weight_J = args.weight_J
        self.device = args.device
        self.test_step = args.test_step
        self.best_metric = None
        self.save_times = args.save_times
        self.save_path = self.model_save_path / self.method / self.save_times

    def build(self, data):
        self.summary(self.args)
        self.GeoKG = data
        if self.method == 'transe':
            self.ent_embs = nn.Embedding(self.GeoKG.geoent_num, self.dim_ins)
            self.rel_embs = nn.Embedding(self.GeoKG.georel_num, self.dim_ins)
            self.type_embs_ent = nn.Embedding(self.GeoKG.geotype_ent_num, self.dim_type)
            self.type_embs_rel = nn.Embedding(self.GeoKG.geotype_rel_num, self.dim_type)

            nn.init.xavier_uniform_(self.ent_embs.weight)
            nn.init.xavier_uniform_(self.rel_embs.weight)
            nn.init.xavier_uniform_(self.type_embs_ent.weight)
            nn.init.xavier_uniform_(self.type_embs_rel.weight)
            self.ins_model = ins_model_transe(self.ent_embs, self.rel_embs, self.l1, self.margin_ins)
            self.type_model = ins_model_transe(self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type)
            self.pair_model = type_model_transe(self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair)

        elif self.method == 'rotate':
            self.ent_embs = nn.Embedding(self.GeoKG.geoent_num, self.dim_ins * 2)
            self.rel_embs = nn.Embedding(self.GeoKG.georel_num, self.dim_ins)
            self.type_embs_ent = nn.Embedding(self.GeoKG.geotype_ent_num, self.dim_type * 2)
            self.type_embs_rel = nn.Embedding(self.GeoKG.geotype_rel_num, self.dim_type)
            self.embedding_range = (self.margin_ins + 2.0) / self.dim_ins
            self.embedding_range_type = (self.margin_type + 2.0) / self.dim_type
            nn.init.uniform_(self.ent_embs.weight, -self.embedding_range, self.embedding_range)
            nn.init.uniform_(self.rel_embs.weight, -self.embedding_range, self.embedding_range)
            nn.init.uniform_(self.type_embs_ent.weight, -self.embedding_range_type, self.embedding_range_type)
            nn.init.uniform_(self.type_embs_rel.weight, -self.embedding_range_type, self.embedding_range_type)

            self.ins_model = ins_model_rotate(self.ent_embs, self.rel_embs, self.l1, self.margin_ins,
                                              self.embedding_range)
            self.type_model = ins_model_rotate(self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type,
                                               self.embedding_range_type)
            self.pair_model = type_model_rotate(self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair)

        elif self.method == 'HAKE':
            self.ent_embs = nn.Embedding(self.GeoKG.geoent_num, self.dim_ins * 2)
            self.rel_embs = nn.Embedding(self.GeoKG.georel_num, self.dim_ins * 3)
            self.type_embs_ent = nn.Embedding(self.GeoKG.geotype_ent_num, self.dim_type * 2)
            self.type_embs_rel = nn.Embedding(self.GeoKG.geotype_rel_num, self.dim_type * 3)
            self.embedding_range = (self.margin_ins + 2.0) / self.dim_ins
            self.embedding_range_type = (self.margin_type + 2.0) / self.dim_type
            nn.init.uniform_(self.ent_embs.weight, -self.embedding_range, self.embedding_range)
            nn.init.uniform_(self.rel_embs.weight, -self.embedding_range, self.embedding_range)
            nn.init.uniform_(self.type_embs_ent.weight, -self.embedding_range_type, self.embedding_range_type)
            nn.init.uniform_(self.type_embs_rel.weight, -self.embedding_range_type, self.embedding_range_type)

            self.ins_model = HAKE_ins(self.ent_embs, self.rel_embs, self.l1, self.margin_ins, self.embedding_range,
                                      self.dim_ins)
            self.type_model = HAKE_ins(self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type,
                                       self.embedding_range_type, self.dim_type)
            self.pair_model = HAKE_type(self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair,
                                        self.embedding_range, self.embedding_range_type)


        elif self.method == 'CompoundE':
                self.ent_embs = nn.Embedding(self.GeoKG.geoent_num, self.dim_ins)
                self.rel_embs = nn.Embedding(self.GeoKG.georel_num,
                                             self.dim_ins * 3)  # CompoundE uses 3 times the embedding size for relations
                self.type_embs_ent = nn.Embedding(self.GeoKG.geotype_ent_num, self.dim_type)
                self.type_embs_rel = nn.Embedding(self.GeoKG.geotype_rel_num,
                                                  self.dim_type * 3)  # CompoundE uses 3 times the embedding size for relation types

                # Initialize embeddings with uniform distribution within the specified range
                self.embedding_range = (self.margin_ins + 2.0) / self.dim_ins
                self.embedding_range_type = (self.margin_type + 2.0) / self.dim_type
                nn.init.uniform_(self.ent_embs.weight, -self.embedding_range, self.embedding_range)
                nn.init.uniform_(self.rel_embs.weight, -self.embedding_range, self.embedding_range)
                nn.init.uniform_(self.type_embs_ent.weight, -self.embedding_range_type, self.embedding_range_type)
                nn.init.uniform_(self.type_embs_rel.weight, -self.embedding_range_type, self.embedding_range_type)

                # Initialize CompoundE models for instance, type, and pair
                self.ins_model = CompoundE_ins(self.ent_embs, self.rel_embs, self.l1, self.margin_ins,
                                               self.embedding_range)
                self.type_model = CompoundE_ins(self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type,
                                                self.embedding_range_type)
                self.pair_model = HAKE_type(self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair,
                                            self.embedding_range, self.embedding_range_type)

        elif self.method == 'DistMult':
            # 初始化嵌入层
            self.ent_embs = nn.Embedding(self.GeoKG.geoent_num, self.dim_ins)
            self.rel_embs = nn.Embedding(self.GeoKG.georel_num, self.dim_ins)
            self.type_embs_ent = nn.Embedding(self.GeoKG.geotype_ent_num, self.dim_type)
            self.type_embs_rel = nn.Embedding(self.GeoKG.geotype_rel_num, self.dim_type)

            # Xavier初始化
            nn.init.xavier_uniform_(self.ent_embs.weight)
            nn.init.xavier_uniform_(self.rel_embs.weight)
            nn.init.xavier_uniform_(self.type_embs_ent.weight)
            nn.init.xavier_uniform_(self.type_embs_rel.weight)

            # 创建DistMult模型
            self.ins_model = ins_model_DistMult(self.ent_embs, self.rel_embs, self.l1, self.margin_ins)
            self.type_model = ins_model_DistMult(self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type)
            self.pair_model = type_model_DistMult(self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair)

        elif self.method == 'transD':
            self.ent_embs = nn.Embedding(self.GeoKG.geoent_num, self.dim_ins)
            self.rel_embs = nn.Embedding(self.GeoKG.georel_num, self.dim_ins)
            self.type_embs_ent = nn.Embedding(self.GeoKG.geotype_ent_num, self.dim_type)
            self.type_embs_rel = nn.Embedding(self.GeoKG.geotype_rel_num, self.dim_type)

            nn.init.xavier_uniform_(self.ent_embs.weight)
            nn.init.xavier_uniform_(self.rel_embs.weight)
            nn.init.xavier_uniform_(self.type_embs_ent.weight)
            nn.init.xavier_uniform_(self.type_embs_rel.weight)

            # 使用TransD模型
            self.ins_model = ins_model_transD(
                self.ent_embs, self.rel_embs, self.l1, self.margin_ins, self.dim_ins
            )
            self.type_model = ins_model_transD(
                self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type, self.dim_type
            )
            self.pair_model = type_model_transD(
                self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair, self.dim_type
            )
        elif self.method == 'ComplEx':
            # 复数嵌入维度是实际维度的一半
            complex_dim_ins = self.dim_ins // 2
            complex_dim_type = self.dim_type // 2

            # 初始化嵌入层
            self.ent_embs = nn.Embedding(self.GeoKG.geoent_num, complex_dim_ins * 2)  # 实部和虚部
            self.rel_embs = nn.Embedding(self.GeoKG.georel_num, complex_dim_ins * 2)  # 实部和虚部
            self.type_embs_ent = nn.Embedding(self.GeoKG.geotype_ent_num, complex_dim_type * 2)
            self.type_embs_rel = nn.Embedding(self.GeoKG.geotype_rel_num, complex_dim_type * 2)

            # Xavier初始化
            nn.init.xavier_uniform_(self.ent_embs.weight)
            nn.init.xavier_uniform_(self.rel_embs.weight)
            nn.init.xavier_uniform_(self.type_embs_ent.weight)
            nn.init.xavier_uniform_(self.type_embs_rel.weight)

            # 创建ComplEx模型
            self.ins_model = ins_model_ComplEx(self.ent_embs, self.rel_embs, self.l1, self.margin_ins)
            self.type_model = ins_model_ComplEx(
                self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type
            )
            self.pair_model = type_model_ComplEx(
                self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair
            )

        if self.args.load_model != None:
            if self.method == 'HAKE':
                self.ins_model = self.loadmodel_HAKE(flag='ins')
                self.type_model = self.loadmodel_HAKE(flag='type')
                self.pair_model = self.loadmodel_HAKE(flag='pair')
            elif self.method == 'rotate':
                self.ins_model = self.loadmodel_rotate(flag='ins')
                self.type_model = self.loadmodel_rotate(flag='type')
                self.pair_model = self.loadmodel_rotate(flag='pair')
                self.ins_model = self.loadmodel_transe(flag='ins')
                self.type_model = self.loadmodel_transe(flag='type')
                self.pair_model = self.loadmodel_transe(flag='pair')
            elif self.method == 'transD':
                self.ins_model = self.loadmodel_transD(flag='ins')
                self.type_model = self.loadmodel_transD(flag='type')
                self.pair_model = self.loadmodel_transD(flag='pair')
            elif self.method == 'CompoundE':
                self.ins_model = self.loadmodel_CompoundE(flag='ins')
                self.type_model = self.loadmodel_CompoundE(flag='type')
                self.pair_model = self.loadmodel_CompoundE(flag='pair')
            elif self.method == 'DistMult':
                self.ins_model = self.loadmodel_DistMult(flag='ins')
                self.type_model = self.loadmodel_DistMult(flag='type')
                self.pair_model = self.loadmodel_DistMult(flag='pair')
            elif self.method == 'transD':
                self.ins_model = self.loadmodel_DistMult(flag='ins')
                self.type_model = self.loadmodel_DistMult(flag='type')
                self.pair_model = self.loadmodel_DistMult(flag='pair')
            elif self.method == 'ComplEx':
                self.ins_model = self.loadmodel_DistMult(flag='ins')
                self.type_model = self.loadmodel_DistMult(flag='type')
                self.pair_model = self.loadmodel_DistMult(flag='pair')
        self.ins_model.to(self.device)
        self.type_model.to(self.device)
        self.pair_model.to(self.device)

        self.ins_optimizer = Optimizer(model=self.ins_model, learning_rate=self.lr,
                                       type='Adam'),
        self.type_optimizer = Optimizer(model=self.type_model, learning_rate=self.lr * self.weight_G,
                                        type='Adam'),
        self.pair_optimizer = Optimizer(model=self.pair_model, learning_rate=self.lr * self.weight_J, type='Adam'),
        self.evaluator = Evaluator(self.ins_model, self.type_model, self.pair_model, self.GeoKG, self.args)
        monitor = Monitor.FILTERED_MEAN_RANK_REL
        self.early_stopper = EarlyStopper(self.args.patience, monitor)

    def train_one_epoch(self, cur_epoch):
        acc_ins_loss = 0
        acc_type_loss = 0
        acc_pair_loss = 0
        num_batch1 = len(self.GeoKG.triplets_withtype["train"]) // self.batch_size_ins
        num_batch2 = len(self.GeoKG.triplets_type["train"]) // self.batch_size_type
        if num_batch2 < 1:
            num_batch2 = 1
        num_batch3 = len(self.GeoKG.pairs_all["train"]) // self.batch_size_pair
        if num_batch3 < 1:
            num_batch3 = 1
        if cur_epoch <= 1:
            print('num_batch =', num_batch1)
        self.ins_model.train()
        self.type_model.train()
        self.pair_model.train()
        order = self.args.training_order.split(',')
        for item in order:
            if item == '0':
                # G-ins training
                progress1 = tqdm(range(self.fold_ins * num_batch1), ncols=150, ascii=True)
                batches1 = self.gen_batch_ins(num_batch1, self.batch_size_ins, self.args.neg_rate, forever=True)
                for _ in progress1:
                    data = next(batches1)
                    batch = dict()
                    batch['h'] = torch.LongTensor(data[0]).to(self.device)
                    batch['r'] = torch.LongTensor(data[1]).to(self.device)
                    batch['t'] = torch.LongTensor(data[2]).to(self.device)
                    batch['neg_h'] = torch.LongTensor(np.array(data[3])).to(self.device)
                    batch['neg_r'] = torch.LongTensor(np.array(data[4])).to(self.device)
                    batch['neg_t'] = torch.LongTensor(np.array(data[5])).to(self.device)
                    batch['subsampling_weight'] = torch.FloatTensor(data[6]).to(self.device)
                    batch['batch_type'] = data[7]
                    if self.method == 'transe':
                        pos_loss, neg_loss, loss_ins = self.train_batch_transe(batch=batch, model=self.ins_model,
                                                                               optimizer=self.ins_optimizer[0],
                                                                               alpha=self.args.alpha,
                                                                               flag='ins')
                    else:
                        pos_loss, neg_loss, loss_ins = self.train_batch_rotate_hake(batch=batch, model=self.ins_model,
                                                                                    optimizer=self.ins_optimizer[0],
                                                                                    alpha=self.args.alpha,
                                                                                    flag='ins')
                    acc_ins_loss += loss_ins.item()
                    progress1.set_description('acc_ins_loss: %f, cur_ins_loss: %f' % (acc_ins_loss, loss_ins))
                progress1.close()

            elif item == '1':
                # G-type training
                progress2 = tqdm(range(self.fold_type * num_batch2), ncols=150, ascii=True)
                batches2 = self.gen_batch_type(num_batch2, self.batch_size_type, self.args.neg_rate, forever=True)
                for _ in progress2:
                    data = next(batches2)
                    batch = dict()
                    batch['h_type_ins'] = torch.LongTensor(data[0]).to(self.device)
                    batch['r_type_ins'] = torch.LongTensor(data[1]).to(self.device)
                    batch['t_type_ins'] = torch.LongTensor(data[2]).to(self.device)
                    batch['neg_h_type_ins'] = torch.LongTensor(np.array(data[3])).to(self.device)
                    batch['neg_r_type_ins'] = torch.LongTensor(np.array(data[4])).to(self.device)
                    batch['neg_t_type_ins'] = torch.LongTensor(np.array(data[5])).to(self.device)
                    batch['subsampling_weight'] = torch.FloatTensor(data[6]).to(self.device)  # 32位浮点类型 torch.cat()
                    batch['batch_type'] = data[7]
                    if self.method == 'transe':
                        pos_loss, neg_loss, loss_type = self.train_batch_transe(batch=batch, model=self.type_model,
                                                                                optimizer=self.type_optimizer[0],
                                                                                alpha=self.args.alpha,
                                                                                flag='type')
                    else:
                        pos_loss, neg_loss, loss_type = self.train_batch_rotate_hake(batch=batch, model=self.type_model,
                                                                                     optimizer=self.type_optimizer[0],
                                                                                     alpha=self.args.alpha,
                                                                                     flag='type')
                    acc_type_loss += loss_type.item()
                    progress2.set_description('acc_type_loss: %f, cur_type_loss: %f' % (acc_type_loss, loss_type))
                progress2.close()

            elif item == '2':
                # P-type training
                progress3 = tqdm(range(self.fold_pair * num_batch3), ncols=150, ascii=True)
                batches3 = self.gen_batch_pair(num_batch3, self.batch_size_pair, self.args.neg_rate, forever=True)
                for _ in progress3:
                    data = next(batches3)
                    batch = dict()
                    batch['ent_pair'] = torch.LongTensor(data[0]).to(self.device)
                    batch['type_pair'] = torch.LongTensor(data[1]).to(self.device)
                    batch['neg_ent_pair'] = torch.LongTensor(data[2]).to(self.device)
                    batch['neg_type_pair'] = data[3].to(self.device)
                    if self.method == 'transe':
                        pos_loss, neg_loss, loss_pair = self.train_batch_pair(batch=batch, model=self.pair_model,
                                                                              optimizer=self.pair_optimizer[0],
                                                                              alpha=self.args.alpha)
                    else:
                        pos_loss, neg_loss, loss_pair = self.train_batch_pair(batch=batch, model=self.pair_model,
                                                                              optimizer=self.pair_optimizer[0],
                                                                              alpha=self.args.alpha)
                    acc_pair_loss += loss_pair.item()
                    progress3.set_description('acc_pair_loss: %f, cur_pair_loss: %f' % (acc_pair_loss, loss_pair))
                progress3.close()

        self.training_results.append([cur_epoch, acc_ins_loss, acc_type_loss, acc_pair_loss])
        return acc_ins_loss, acc_type_loss, acc_pair_loss

    def join_train(self):
        self.monitor = Monitor.FILTERED_MEAN_RANK_REL
        for cur_epoch in range(self.epochs):
            ins_loss, type_loss, pair_loss = self.train_one_epoch(cur_epoch)
            if np.isnan(ins_loss) or np.isnan(type_loss) or np.isnan(pair_loss):
                print("Training collapsed.")
                return
            tot_loss = ins_loss + type_loss + pair_loss
            log.info('Join model Epoch [%s/%s]: %s', cur_epoch, self.epochs, tot_loss)

            if cur_epoch % self.test_step == 0:
                self.ins_model.eval()
                with torch.no_grad():
                    metrics = self.evaluator.mini_test(cur_epoch)
                    self.eval_results.append([cur_epoch,
                                      metrics['mr_rel'], metrics['fmr_rel'], metrics['mrr_rel'], metrics['fmrr_rel'],
                                      metrics['hits@1_rel'], metrics['filtered_hits@1_rel'], metrics['hits@3_rel'],
                                      metrics['filtered_hits@3_rel'],
                                      metrics['hits@5_rel'], metrics['filtered_hits@5_rel'], metrics['hits@10_rel'],
                                      metrics['filtered_hits@10_rel']])
                    if self.best_metric is None:
                        self.best_metric = metrics
                        self.save_model()
                    else:
                        if self.monitor == Monitor.MEAN_RANK_REL or self.monitor == Monitor.FILTERED_MEAN_RANK_REL:
                            is_better = self.best_metric[self.monitor.value] > metrics[self.monitor.value]
                        else:
                            is_better = self.best_metric[self.monitor.value] < metrics[self.monitor.value]
                        if is_better:
                            self.save_model()
                            self.best_metric = metrics
                    self.export_embeddings()
        self.ins_model.eval()
        self.type_model.eval()
        self.pair_model.eval()
        self.save_model()
        with torch.no_grad():
            metrics = self.evaluator.full_test(cur_epoch)
            self.eval_results.append([cur_epoch,
                                      metrics['mr_rel'], metrics['fmr_rel'], metrics['mrr_rel'], metrics['fmrr_rel'],
                                      metrics['hits@1_rel'], metrics['filtered_hits@1_rel'], metrics['hits@3_rel'],
                                      metrics['filtered_hits@3_rel'],
                                      metrics['hits@5_rel'], metrics['filtered_hits@5_rel'], metrics['hits@10_rel'],
                                      metrics['filtered_hits@10_rel']])
        self.evaluator.metric.save_test_summary(self.save_path)
        self.export_embeddings()
        self.save_training_result()
        self.save_evaluator_result()
        print("Done!")

    def save_model(self):
        saved_path = self.save_path
        saved_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.ins_model.state_dict(), str(saved_path / "ins_model.vec.pt"))
        torch.save(self.type_model.state_dict(), str(saved_path / "type_model.vec.pt"))
        torch.save(self.pair_model.state_dict(), str(saved_path / "pair_model.vec.pt"))
        save_path_config = saved_path / "config.npy"
        np.save(save_path_config, self.args)

    def save_training_result(self):
        df = pd.DataFrame(self.training_results, columns=['Epochs', 'ins_Loss', 'type_loss', 'pair_Loss'])
        with open(str(self.save_path / ('_Training_results_' + self.save_times + '.csv')), 'w', encoding='utf-8') as fh:
            df.to_csv(fh)

    def save_evaluator_result(self):
        df = pd.DataFrame(self.eval_results, columns=['epoch', 'mr_rel', 'fmr_rel', 'mrr_rel', 'fmrr_rel', 'hits@1_rel',
                                                      'filtered_hits@1_rel',
                                                      'hits@3_rel', 'filtered_hits@3_rel', 'hits@5_rel',
                                                      'filtered_hits@5_rel',
                                                      'hits@10_rel', 'filtered_hits@10_rel'])
        df = df.T
        with open(str(self.save_path / ('_Eval_results_' + self.save_times + '.csv')), 'w', encoding='utf-8') as fh:
            df.to_csv(fh)

    def export_embeddings(self):
        save_path = self.save_path
        save_path.mkdir(parents=True, exist_ok=True)

        idx2ent = self.GeoKG.geoent_tokens
        idx2rel = self.GeoKG.georel_tokens
        idx2type_ent = self.GeoKG.geotype_ent_tokens
        idx2type_rel = self.GeoKG.geotype_rel_tokens
        with open(str(save_path / "ent_labels.tsv"), 'w', encoding='utf-8') as l_export_file:
            for label in idx2ent.values():
                l_export_file.write(label + "\n")

        with open(str(save_path / "rel_labels.tsv"), 'w', encoding='utf-8') as l_export_file:
            for label in idx2rel.values():
                l_export_file.write(label + "\n")

        with open(str(save_path / "type_ent_labels.tsv"), 'w', encoding='utf-8') as l_export_file:
            for label in idx2type_ent.values():
                l_export_file.write(label + "\n")

        with open(str(save_path / "type_rel_labels.tsv"), 'w', encoding='utf-8') as l_export_file:
            for label in idx2type_rel.values():
                l_export_file.write(label + "\n")

        ins_embed = self.ins_model.parameter_list
        type_embed = self.type_model.parameter_list
        for para in ins_embed:
            all_ids = list(range(0, int(ins_embed[para].weight.shape[0])))
            stored_name = para
            if len(ins_embed[para].weight.shape) == 2:
                all_embs = ins_embed[para].weight.detach().detach().cpu().numpy()
                with open(str(save_path / ("%s.tsv" % stored_name)), 'w', encoding='utf-8') as v_export_file:
                    for idx in all_ids:
                        v_export_file.write("\t".join([str(x) for x in all_embs[idx]]) + "\n")
        for para in type_embed:
            all_ids = list(range(0, int(type_embed[para].weight.shape[0])))
            stored_name = para
            if len(type_embed[para].weight.shape) == 2:
                all_embs = type_embed[para].weight.detach().detach().cpu().numpy()
                with open(str(save_path / ("type_%s.tsv" % stored_name)), 'w', encoding='utf-8') as v_export_file:
                    for idx in all_ids:
                        v_export_file.write("\t".join([str(x) for x in all_embs[idx]]) + "\n")

    def loadmodel_HAKE(self, flag):
        try:
            model_obj = getattr(importlib.import_module("model"), 'HAKE_ins')
            model_obj_pair = getattr(importlib.import_module("model"), 'HAKE_type')
            if flag == 'ins':
                model = model_obj(self.ent_embs, self.rel_embs, self.l1, self.margin_ins, self.embedding_range,
                                  self.dim_ins)
                model.load_state_dict(torch.load(self.args.load_model + '/ins_model.vec.pt'))
            elif flag == 'type':
                model = model_obj(self.type_embs_ent, self.type_embs_rel, self.l1,
                                  self.margin_type, self.embedding_range_type, self.dim_type)
                model.load_state_dict(torch.load(self.args.load_model + '/type_model.vec.pt'))
            elif flag == 'pair':
                model = model_obj_pair(self.ent_embs, self.type_embs_ent,
                                       self.l1, self.margin_pair, self.embedding_range, self.embedding_range_type)
                model.load_state_dict(torch.load(self.args.load_model + '/pair_model.vec.pt'))
            model.eval()
            return model
        except ModuleNotFoundError:
            log.error("%s model has not been implemented." % (self.method))

    def loadmodel_rotate(self, flag):
        try:
            model_obj = getattr(importlib.import_module("model"), 'ins_model_rotate')
            model_obj_pair = getattr(importlib.import_module("model"), 'type_model_rotate')
            if flag == 'ins':
                model = model_obj(self.ent_embs, self.rel_embs, self.l1, self.margin_ins, self.embedding_range)
                model.load_state_dict(torch.load(self.args.load_model + '/ins_model.vec.pt'))
            elif flag == 'type':
                model = model_obj(self.type_embs_ent, self.type_embs_rel, self.l1,
                                  self.margin_type, self.embedding_range_type)
                model.load_state_dict(torch.load(self.args.load_model + '/type_model.vec.pt'))
            elif flag == 'pair':
                model = model_obj_pair(self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair)
                model.load_state_dict(torch.load(self.args.load_model + '/pair_model.vec.pt'))
            model.eval()
            return model
        except ModuleNotFoundError:
            log.error("%s model has not been implemented." % (self.method))

    def loadmodel_transe(self, flag):
        try:
            model_obj = getattr(importlib.import_module("model"), 'ins_model_transe')
            model_obj_pair = getattr(importlib.import_module("model"), 'type_model_transe')
            if flag == 'ins':
                model = model_obj(self.ent_embs, self.rel_embs, self.l1, self.margin_ins)
                model.load_state_dict(torch.load(self.args.load_model + '/ins_model.vec.pt'))
            elif flag == 'type':
                model = model_obj(self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type)
                model.load_state_dict(torch.load(self.args.load_model + '/type_model.vec.pt'))
            elif flag == 'pair':
                model = model_obj_pair(self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair)
                model.load_state_dict(torch.load(self.args.load_model + '/pair_model.vec.pt'))
            model.eval()
            return model
        except ModuleNotFoundError:
            log.error("%s model has not been implemented." % (self.method))


    def loadmodel_CompoundE(self, flag):
        try:
            model_obj = getattr(importlib.import_module("model"), 'ins_model_transD')
            model_obj_pair = getattr(importlib.import_module("model"), 'type_model_transD')
            if flag == 'ins':
                model = model_obj(self.ent_embs, self.rel_embs, self.l1, self.margin_ins)
                model.load_state_dict(torch.load(self.args.load_model + '/ins_model.vec.pt'))
            elif flag == 'type':
                model = model_obj(self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type)
                model.load_state_dict(torch.load(self.args.load_model + '/type_model.vec.pt'))
            elif flag == 'pair':
                model = model_obj_pair(self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair)
                model.load_state_dict(torch.load(self.args.load_model + '/pair_model.vec.pt'))
            model.eval()
            return model
        except ModuleNotFoundError:
            log.error("%s model has not been implemented." % (self.method))

    def loadmodel_DistMult(self, flag):
        try:
            model_obj = getattr(importlib.import_module("model"), 'ins_model_DistMult')
            model_obj_pair = getattr(importlib.import_module("model"), 'type_model_DistMult')
            if flag == 'ins':
                model = model_obj(self.ent_embs, self.rel_embs, self.l1, self.margin_ins)
                model.load_state_dict(torch.load(self.args.load_model + '/ins_model.vec.pt'))
            elif flag == 'type':
                model = model_obj(self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type)
                model.load_state_dict(torch.load(self.args.load_model + '/type_model.vec.pt'))
            elif flag == 'pair':
                model = model_obj_pair(self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair)
                model.load_state_dict(torch.load(self.args.load_model + '/pair_model.vec.pt'))
            model.eval()
            return model
        except ModuleNotFoundError:
            log.error("DistMult model has not been implemented.")

    def loadmodel_transD(self, flag):
        try:
            model_obj_ins = getattr(importlib.import_module("model"), 'ins_model_transD')
            model_obj_type = getattr(importlib.import_module("model"), 'type_model_transD')

            if flag == 'ins':
                model = model_obj_ins(
                    self.ent_embs, self.rel_embs, self.l1, self.margin_ins, self.dim_ins
                )
                model.load_state_dict(torch.load(self.args.load_model + '/ins_model.vec.pt'))
            elif flag == 'type':
                model = model_obj_ins(
                    self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type, self.dim_type
                )
                model.load_state_dict(torch.load(self.args.load_model + '/type_model.vec.pt'))
            elif flag == 'pair':
                model = model_obj_type(
                    self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair, self.dim_type
                )
                model.load_state_dict(torch.load(self.args.load_model + '/pair_model.vec.pt'))
            model.eval()
            return model
        except ModuleNotFoundError:
            log.error("TransD model has not been implemented.")

    def loadmodel_ComplEx(self, flag):
        try:
            model_obj_ins = getattr(importlib.import_module("model"), 'ins_model_ComplEx')
            model_obj_type = getattr(importlib.import_module("model"), 'type_model_ComplEx')

            if flag == 'ins':
                model = model_obj_ins(
                    self.ent_embs, self.rel_embs, self.l1, self.margin_ins
                )
                model.load_state_dict(torch.load(self.args.load_model + '/ins_model.vec.pt'))
            elif flag == 'type':
                model = model_obj_ins(
                    self.type_embs_ent, self.type_embs_rel, self.l1, self.margin_type
                )
                model.load_state_dict(torch.load(self.args.load_model + '/type_model.vec.pt'))
            elif flag == 'pair':
                model = model_obj_type(
                    self.ent_embs, self.type_embs_ent, self.l1, self.margin_pair
                )
                model.load_state_dict(torch.load(self.args.load_model + '/pair_model.vec.pt'))
            model.eval()
            return model
        except ModuleNotFoundError:
            log.error("ComplEx model has not been implemented.")

    def infer_rels(self, h_name, t_name, topk=5):
        idx2ent = self.GeoKG.geoent_tokens
        idx2rel = self.GeoKG.georel_tokens
        h = list(idx2ent.values()).index(h_name)
        t = list(idx2ent.values()).index(t_name)
        rels, dis = self.evaluator.test_rel_rank(h, t, topk)
        rels = rels.detach().cpu().numpy()
        dis = dis.detach().cpu().numpy()
        logs = [
            "",
            "(head,tail)->({},{}) :: Inferred rels->({})".format(h, t, ",".join([str(i) for i in rels])),
            "",
            "head: %s" % idx2ent[h],
            "tail: %s" % idx2ent[t],
        ]
        rel_withdis = zip(rels, dis)
        for idx, (rel, dis1) in enumerate(rel_withdis):
            logs.append("%s, distance: %f" % (idx2rel[rel], dis1))
        log.info("\n".join(logs))
        return {rel: idx2rel[rel] for rel in rels}