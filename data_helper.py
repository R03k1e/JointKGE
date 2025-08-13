import logging
import copy
from collections import defaultdict
import numpy as np
import torch
from enum import Enum

log = logging.getLogger(__name__)


class BatchType(Enum):
    HEAD_BATCH = 0
    TAIL_BATCH = 1
    SINGLE = 2


class Triple_only:
    def __init__(self, h, r, t):
        self.h = h
        self.r = r
        self.t = t

    def set_ids(self, h, r, t):
        self.h = h
        self.r = r
        self.t = t

    def set_value(self, type, value):
        if type == 'h':
            self.h = value
        if type == 't':
            self.t = value


class Pair:
    def __init__(self, ent, type):
        self.ent = ent
        self.type = type

    def set_ids(self, ent, type):
        self.ent = ent
        self.type = type

    def set_value(self, mark, value):
        if mark == 'type':
            self.type = value


class Triple_withtype:
    def __init__(self, h, r, t, h_type, t_type):
        self.h = h
        self.r = r
        self.t = t
        self.h_type = h_type
        self.t_type = t_type

    def set_ids(self, h, r, t, h_type, t_type):
        self.h = h
        self.r = r
        self.t = t
        self.h_type = h_type
        self.t_type = t_type

    def set_value(self, type, value):
        if type == 'h':
            self.h = value
        if type == 't':
            self.t = value
        if type == 'h_type':
            self.h_type = value
        if type == 't_type':
            self.t_type = value


class GeoKG():
    def __init__(self):
        self.geoent_tokens = {}
        self.geoent_index = {}
        self.geoent_num = 0
        self.geoent_forupdate = set([])
        self.geoent_head_count = defaultdict(int)
        self.geoent_tail_count = defaultdict(int)
        self.geoent_count = defaultdict(int)
        self.georel_tokens = {}
        self.georel_index = {}
        self.georel_num = 0
        self.georel_forupdate = set([])
        self.georel_count = defaultdict(int)
        self.geotype_ent_tokens = {}
        self.geotype_rel_tokens = {}
        self.geotype_ent_num = 0
        self.geotype_rel_num = 0
        self.geoent_forupdate_type = set([])
        self.georel_forupdate_type = set([])
        self.geotype_head_count = defaultdict(int)
        self.geotype_tail_count = defaultdict(int)
        self.geotype_ent_count = defaultdict(int)
        self.geotype_rel_count = defaultdict(int)

        self.triplets_withtype = {'train': [], 'test': [], 'valid': []}
        self.triplets_record_withtype = set([])
        self.triplets_type = {'train': [], 'test': [], 'valid': []}
        self.triplets_record_type = set([])
        self.triplets_count = defaultdict(int)
        self.triplets_type_count = defaultdict(int)
        self.pairs = {'train': [], 'test': [], 'valid': []}
        self.pairs_all = {'train': [], 'test': [], 'valid': []}
        self.pairs_record = set([])
        self.pair_count = defaultdict(int)

        self.ent_relation = defaultdict(set)
        self.relation_headentity = defaultdict(set)
        self.relation_tailentity = defaultdict(set)

        self.hr_t_idx = defaultdict(set)
        self.tr_h_idx = defaultdict(set)
        self.ht_r_idx = defaultdict(set)
        self.hr_t_type_idx = defaultdict(set)
        self.tr_h_type_idx = defaultdict(set)
        self.hr_t_freq = {}
        self.tr_h_freq = {}
        self.hr_t_type_freq = {}
        self.tr_h_type_freq = {}
        self.ent_type_pairs = {}
        self.triple2type = {'train': [], 'test': [], 'valid': []}
        self.triple2type_type = {'train': [], 'test': [], 'valid': []}
        self.ent_type_pairs_train = defaultdict(set)

        self.type = {}
        self.type_embed = {}
        self.type_embed_padded = np.array([0])
        self.type_length = 100
        self.type_index = np.array([0])

    def prepare_data(self, paths_ins, paths_type, paths_pair):
        for key, path in paths_type.items():
            self.load_triplets_onlytype(key, path)
        for key, path in paths_ins.items():
            self.load_triplets(key, path)
        for key, path in paths_pair.items():
            self.load_pairs(key, path)
        self.load_ent_rel_type_idx()
        for key, path in paths_ins.items():
            self.load_triplets_idx(key)
        for key, path in paths_type.items():
            self.load_typetriplets_idx(key)
        for key, path in paths_pair.items():
            self.load_pairs_idx(key)
        del self.pairs
        print("loaded all sample data!")

    def load_ent_rel_type_idx(self):
        self.geoent_tokens = {k: v for k, v in enumerate(np.sort(list(self.geoent_count.keys())))}
        self.georel_tokens = {k: v for k, v in enumerate(np.sort(list(self.georel_count.keys())))}
        self.geotype_ent_tokens = {k: v for k, v in enumerate(np.sort(list(self.geoent_forupdate_type)))}
        self.geotype_rel_tokens = {k: v for k, v in enumerate(np.sort(list(self.georel_forupdate_type)))}

        self.geoent_index = {v: k for k, v in self.geoent_tokens.items()}
        self.georel_index = {v: k for k, v in self.georel_tokens.items()}
        self.geotype_ent_index = {v: k for k, v in self.geotype_ent_tokens.items()}
        self.geotype_rel_index = {v: k for k, v in self.geotype_rel_tokens.items()}

        self.geoent_num = len(self.geoent_tokens)
        self.georel_num = len(self.georel_tokens)
        self.geotype_ent_num = len(self.geotype_ent_tokens)
        self.geotype_rel_num = len(self.geotype_rel_tokens)

    def load_triplets_idx(self, key):
        init_cnt = 3
        for t in self.triplets_withtype[key]:
            head = self.geoent_index[t.h]
            rel = self.georel_index[t.r]
            tail = self.geoent_index[t.t]
            if key == 'train':
                if (head, rel) not in self.hr_t_freq.keys():
                    self.hr_t_freq[(head, rel)] = init_cnt
                if (tail, rel) not in self.tr_h_freq.keys():
                    self.tr_h_freq[(tail, rel)] = init_cnt
                self.hr_t_idx[(head, rel)].add(tail)
                self.tr_h_idx[(tail, rel)].add(head)
                self.hr_t_freq[(head, rel)] += 1
                self.tr_h_freq[(tail, rel)] += 1
            t.set_ids(head, rel, tail)

    def load_typetriplets_idx(self, key):
        init_cnt = 3
        for t in self.triplets_type[key]:
            head = self.geotype_ent_index[t.h]
            rel = self.geotype_rel_index[t.r]
            tail = self.geotype_ent_index[t.t]
            if key == 'train':
                self.hr_t_type_idx[(head, rel)].add(tail)
                self.tr_h_type_idx[(tail, rel)].add(head)
                if (head, rel) not in self.hr_t_type_freq.keys():
                    self.hr_t_type_freq[(head, rel)] = init_cnt
                if (tail, rel) not in self.tr_h_type_freq.keys():
                    self.tr_h_type_freq[(tail, rel)] = init_cnt
                self.hr_t_type_freq[(head, rel)] += 1
                self.tr_h_type_freq[(tail, rel)] += 1
            t.set_ids(head, rel, tail)

    def load_pairs_idx(self, key):
        for t in self.pairs[key]:
            if t.type == '[]':
                self.pairs[key].remove(t)
                continue
            if 'establishment' == t.type[0]:
                t.type.reverse()
                if 'natural_feature' in t.type:
                    if len(t.type) > 2:
                        del (t.type[0])
            while '' in t.type:
                t.type.remove('')
        triple2type_temp = []
        for tr in self.triple2type[key]:
            m = copy.deepcopy(tr)
            m1 = m[0].type[0] + '-' + m[1].type[0]
            triple2type_temp.append(m1)
        self.triple2type_type[key] = triple2type_temp
        for t in self.pairs[key]:
            s = copy.deepcopy(t)
            if self.geoent_index[s.ent] in self.ent_type_pairs.keys():
                self.ent_type_pairs[self.geoent_index[s.ent]].update([x for x in s.type])
            else:
                self.ent_type_pairs.setdefault(self.geoent_index[s.ent], set()).update([x for x in s.type])
            if key == 'train':
                if self.geoent_index[s.ent] in self.ent_type_pairs_train.keys():
                    self.ent_type_pairs_train[self.geoent_index[s.ent]].add(self.geotype_ent_index[s.type[0]])
                else:
                    self.ent_type_pairs_train.setdefault(self.geoent_index[s.ent], set()).add(
                        self.geotype_ent_index[s.type[0]])
            t.set_ids(self.geoent_index[s.ent], self.geotype_ent_index[s.type[0]])
            self.pairs_all[key].append(t)
        log.info('Loaded alltype pairs from %s : %s pairs', key, len(self.pairs_all[key]))

    def load_triplets_onlytype(self, key, path):
        triplets_type = []
        entities = set()
        relations = set()
        with open(path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):  # 使用enumerate记录行号，从1开始计数
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                li = line.split(' ')
                if len(li) != 3:
                    raise ValueError(f"Invalid triplet format at line {line_number}: {line}")
                # 继续处理正常逻辑
                self.triplets_type_count[(li[0], li[1], li[2])] += 1
                t_ins = Triple_only(li[0], li[1], li[2])
                triplets_type.append(t_ins)
                self.triplets_record_type.add(t_ins)
                entities.update([li[0], li[2]])
                relations.add(li[1])
                self.geotype_head_count[li[0]] += 1
                self.geotype_tail_count[li[2]] += 1
                self.geotype_ent_count[li[0]] += 1
                self.geotype_ent_count[li[2]] += 1
                self.geotype_rel_count[li[1]] += 1
        self.geoent_forupdate_type.update(entities)
        self.georel_forupdate_type.update(relations)
        self.triplets_type[key] = triplets_type
        log.info('Loaded type triplets from %s : %s triplets, %s ents, %s rels', key, len(triplets_type),
                 len(entities),
                 len(relations))

    def load_triplets(self, key, path):
        triplets_withtype = []
        entities = set()
        relations = set()

        with open(path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):  # 使用enumerate记录行号，从1开始计数
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                li = line.split(' ')
                if len(li) < 5:
                    raise ValueError(f"Invalid triplet format at line {line_number}: {line}")
                # 继续处理正常逻辑
                t = Triple_only(li[0].strip(), li[1].strip(), li[2].strip())
                self.triplets_count[(li[0].strip(), li[1].strip(), li[2].strip())] += 1
                self.geoent_head_count[li[0].strip()] += 1
                self.geoent_tail_count[li[2].strip()] += 1
                self.geoent_count[li[0].strip()] += 1
                self.geoent_count[li[2].strip()] += 1
                self.georel_count[li[1].strip()] += 1
                triplets_withtype.append(t)
                self.triplets_record_withtype.add(t)
                entities.update([li[0].strip(), li[2].strip()])
                relations.add(li[1].strip())
                self.ent_relation.setdefault(li[0].strip(), set()).add((li[1].strip(), li[3].strip(), 'head'))
                self.ent_relation.setdefault(li[2].strip(), set()).add((li[1].strip(), li[4].strip(), 'tail'))
                self.relation_headentity.setdefault(li[1].strip(), set()).add((li[0].strip(), li[3].strip()))
                self.relation_tailentity.setdefault(li[1].strip(), set()).add((li[2].strip(), li[4].strip()))
        self.geoent_forupdate.update(entities)
        self.georel_forupdate.update(relations)
        self.triplets_withtype[key] = triplets_withtype
        log.info('Loaded triplets from %s : %s triplets, %s ents, %s rels', key, len(triplets_withtype),
                 len(entities), len(relations))

    def load_pairs(self, key, path):
        geo_type = set()
        geo_ent = set()
        pairs = []
        triplets_type = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                li = line.split(' ')
                h_type = li[3].strip()
                t_type = li[4].strip()
                self.pair_count[(li[0].strip(), h_type)] += 1
                self.pair_count[(li[2].strip(), t_type)] += 1
                if len(h_type) > 2:
                    subject_multitype = eval(
                        (h_type[:h_type.find(']') + 1] + h_type[h_type.find(']') + 1:].replace(",",
                                                                                               "', '") + "'").replace(
                            "']", '') + ']')
                else:
                    subject_multitype = h_type
                if len(t_type) > 2:
                    object_multitype = eval(
                        (t_type[:t_type.find(']') + 1] + t_type[t_type.find(']') + 1:].replace(",",
                                                                                               "', '") + "'").replace(
                            "']", '') + ']')
                else:
                    object_multitype = t_type
                pair_h = Pair(li[0].strip(), subject_multitype)
                pair_t = Pair(li[2].strip(), object_multitype)
                pairs.extend([pair_h, pair_t])
                triplets_type.append((pair_h, pair_t))
                self.pairs_record.update([pair_h, pair_t])
                geo_ent.update([li[0].strip(), li[2].strip()])
                geo_type.update(subject_multitype, object_multitype)
        self.pairs[key] = pairs
        self.triple2type[key] = triplets_type
        log.info('Loaded pairs from %s : %s pairs, %s ents, %s types', key, len(pairs), len(geo_ent), len(geo_type))

    def corrupt_batch_triplets_ins(self, neg_rate, data, batch_type):
        positive_triplets = {(t.h, t.r, t.t): 1 for t in data}
        neg_batch_tr = []
        subsampling_weights = []
        for t in data:
            neg_triplets = []
            head, rel, tail = t.h, t.r, t.t
            subsampling_weight = self.hr_t_freq[(head, rel)] + self.tr_h_freq[(tail, rel)]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
            for _ in range(neg_rate):
                if batch_type == BatchType.TAIL_BATCH:
                    idx_replace = np.random.randint(self.geoent_num)
                    while (t.h, t.r, idx_replace) in positive_triplets:
                        idx_replace = np.random.randint(self.geoent_num)
                else:
                    idx_replace = np.random.randint(self.geoent_num)
                    while (idx_replace, t.r, t.t) in positive_triplets:
                        idx_replace = np.random.randint(self.geoent_num)
                neg_triplets.append(idx_replace)
            subsampling_weights.append(subsampling_weight)
            neg_batch_tr.append(torch.from_numpy(np.array(neg_triplets)))
        return neg_batch_tr, subsampling_weights

    def corrupt_batch_triplets_type(self, neg_rate, data, batch_type):
        neg_batch_triplets = []
        subsampling_weights = []
        for t in data:
            head, rel, tail = t.h, t.r, t.t
            subsampling_weight = self.hr_t_type_freq[(head, rel)] + self.tr_h_type_freq[(tail, rel)]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
            neg_triplets = []
            neg_size = 0
            while neg_size < neg_rate:
                neg_triplets_tmp = np.random.randint(self.geotype_ent_num, size=neg_rate * 2)
                if batch_type == BatchType.HEAD_BATCH:
                    mask = np.in1d(
                        neg_triplets_tmp,
                        self.tr_h_type_idx[(tail, rel)],
                        assume_unique=True,
                        invert=True
                    )
                elif batch_type == BatchType.TAIL_BATCH:
                    mask = np.in1d(
                        neg_triplets_tmp,
                        self.hr_t_type_idx[(head, rel)],
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError('Invalid BatchType: {}'.format(batch_type))
                neg_triplets_tmp = neg_triplets_tmp[mask]
                neg_triplets.append(neg_triplets_tmp)
                neg_size += neg_triplets_tmp.size
            neg_triplets = np.concatenate(neg_triplets)[:neg_rate]
            neg_batch_triplets.append(torch.from_numpy(neg_triplets))
            subsampling_weights.append(subsampling_weight)
        return neg_batch_triplets, subsampling_weights

    def corrupt_batch_pairs(self, neg_rate, data):
        neg_batch_pairs = []
        for t in data:
            ent, type = t.ent, t.type
            neg_triplets = []
            neg_size = 0
            while neg_size < neg_rate:
                neg_triplets_tmp = np.random.randint(self.geotype_ent_num, size=neg_rate * 2)
                mask = np.in1d(
                    neg_triplets_tmp,
                    self.ent_type_pairs_train[ent],
                    assume_unique=True,
                    invert=True
                )
                neg_triplets_tmp = neg_triplets_tmp[mask]
                neg_triplets.append(neg_triplets_tmp)
                neg_size += neg_triplets_tmp.size
            neg_triplets = np.concatenate(neg_triplets)[:neg_rate]
            neg_batch_pairs.append(torch.from_numpy(neg_triplets.astype(np.int64)))
        return neg_batch_pairs