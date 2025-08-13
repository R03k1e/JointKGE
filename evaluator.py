from tqdm import tqdm
import numpy as np
import timeit
import torch
import logging
import os
from pathlib import Path
from data_helper import BatchType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class Metric:

    def __init__(self, data, args):
        self.mr = {}
        self.fmr = {}
        self.mrr = {}
        self.fmrr = {}

        self.mr_rel = {}
        self.fmr_rel = {}
        self.mrr_rel = {}
        self.fmrr_rel = {}

        self.mr_type = {}
        self.mrr_type = {}
        self.fmr_type = {}
        self.fmrr_type = {}

        self.mr_pair = {}
        self.mrr_pair = {}
        self.fmr_pair = {}
        self.fmrr_pair = {}

        self.hit = {}
        self.fhit = {}
        self.hit_rel = {}
        self.fhit_rel = {}
        self.type_hit = {}
        self.type_fhit = {}
        self.pair_hit = {}
        self.pair_fhit = {}

        self.data = data
        self.args = args
        self.epoch = None
        self.reset()
        self.hr_t_idx = data.hr_t_idx
        self.tr_h_idx = data.tr_h_idx
        self.ht_r_idx = data.ht_r_idx
        self.hr_t_type_idx = data.hr_t_type_idx
        self.tr_h_type_idx = data.tr_h_type_idx
        self.ent_type_pairs_train = data.ent_type_pairs_train

    def reset(self):
        self.rank_head = []
        self.rank_tail = []
        self.rank_rel = []
        self.rank_head_type = []
        self.rank_tail_type = []
        self.rank_type = []
        self.f_rank_head = []
        self.f_rank_tail = []
        self.f_rank_rel = []
        self.f_rank_head_type = []
        self.f_rank_tail_type = []
        self.f_rank_type = []
        self.epoch = None
        self.start_time = timeit.default_timer()

    def get_head_rank(self, head_candidate, h, r, t):
        hranks = []
        fhranks = []
        for i in range(len(head_candidate)):
            hrank = 0
            fhrank = 0
            for j in range(len(head_candidate[i])):
                val = head_candidate[i][j]
                j += 1
                if val != h[i]:
                    hrank += 1
                    fhrank += 1
                    if val in self.tr_h_idx[(t[i], r[i])]:
                        fhrank -= 1

                else:
                    break
            hranks.append(hrank)
            fhranks.append(fhrank)
        return hranks, fhranks

    def get_tail_rank(self, tail_candidate, h, r, t):
        tranks = []
        ftranks = []
        for i in range(len(tail_candidate)):
            trank = 0
            ftrank = 0
            for j in range(len(tail_candidate[i])):
                val = tail_candidate[i][j]
                j += 1
                if val != t[i]:
                    trank += 1
                    ftrank += 1
                    if val in self.hr_t_idx[(h[i], r[i])]:
                        ftrank -= 1
                else:
                    break
            tranks.append(trank)
            ftranks.append(ftrank)
        return tranks, ftranks

    def get_rel_rank(self, rel_candidate, h, r, t):
        rranks = []
        frranks = []
        for i in range(len(rel_candidate)):
            rrank = 0
            frrank = 0
            for j in range(len(rel_candidate[i])):
                val = rel_candidate[i][j]
                j += 1
                if val != r[i]:
                    rrank += 1
                    frrank += 1
                    if val in self.ht_r_idx[(h[i], t[i])]:
                        frrank -= 1
                else:
                    break
            rranks.append(rrank)
            frranks.append(frrank)
        return rranks, frranks

    def get_headtype_rank(self, head_candidate, h, r, t):
        htype_ranks = []
        fhtype_ranks = []
        for i in range(len(head_candidate)):
            hrank = 0
            fhrank = 0
            for j in range(len(head_candidate[i])):
                val = head_candidate[i][j]
                j += 1
                if val != h[i]:
                    hrank += 1
                    fhrank += 1
                    if val in self.tr_h_type_idx[(t[i], r[i])]:
                        fhrank -= 1
                else:
                    break
            htype_ranks.append(hrank)
            fhtype_ranks.append(fhrank)
        return htype_ranks, fhtype_ranks

    def get_tailtype_rank(self, tail_candidate, h, r, t):
        ttype_ranks = []
        fttype_ranks = []
        for i in range(len(tail_candidate)):
            trank = 0
            ftrank = 0
            for j in range(len(tail_candidate[i])):
                val = tail_candidate[i][j]
                j += 1
                if val != t[i]:
                    trank += 1
                    ftrank += 1
                    if val in self.hr_t_type_idx[(h[i], r[i])]:
                        ftrank -= 1
                else:
                    break
            ttype_ranks.append(trank)
            fttype_ranks.append(ftrank)
        return ttype_ranks, fttype_ranks

    def get_type_rank(self, type_candidate, ent, ent_type):
        ranks = []
        franks = []
        for i in range(len(type_candidate)):
            rank = 0
            frank = 0
            for j in range(len(type_candidate[i])):
                val = type_candidate[i][j]
                j += 1
                if val != ent_type[i]:
                    rank += 1
                    frank += 1
                    if ent[i] in self.ent_type_pairs_train.keys():
                        if val in self.ent_type_pairs_train.get(ent[i]):
                            frank -= 1
                else:
                    break
            ranks.append(rank)
            franks.append(frank)
        return ranks, franks

    def append_result(self, res_ins):
        predict_rel = res_ins[0]
        h, r, t = res_ins[1], res_ins[2], res_ins[3]
        self.epoch = res_ins[4][0]
        r_rank, f_r_rank = self.get_rel_rank(predict_rel, h, r, t)
        self.rank_rel.append(r_rank)
        self.f_rank_rel.append(f_r_rank)

    def settle(self):
        rel_ranks = np.asarray(self.rank_rel, dtype=np.float32) + 1
        rel_franks = np.asarray(self.f_rank_rel, dtype=np.float32) + 1

        ranks_ins_rel = rel_ranks.ravel()
        franks_ins_rel = rel_franks.ravel()
        self.mr_rel[self.epoch] = np.mean(ranks_ins_rel)
        self.mrr_rel[self.epoch] = np.mean(np.reciprocal(ranks_ins_rel))
        self.fmr_rel[self.epoch] = np.mean(franks_ins_rel)
        self.fmrr_rel[self.epoch] = np.mean(np.reciprocal(franks_ins_rel))

        for hit in self.args.hits:
            self.hit_rel[(self.epoch, hit)] = np.mean(ranks_ins_rel <= hit, dtype=np.float32)
            self.fhit_rel[(self.epoch, hit)] = np.mean(franks_ins_rel <= hit, dtype=np.float32)

    def display_summary(self):
        stop_time = timeit.default_timer()
        test_results = []
        test_results.append('')
        test_results.append(
            "------Test Results : Epoch: %d --- time: %.2f------------" % (self.epoch, stop_time - self.start_time))
        test_results.append("------Test Results for ins_rel------------")
        test_results.append('--# of entities, # of relations: %d, %d' % (self.data.geoent_num, self.data.georel_num))
        test_results.append(
            '--mr,  filtered mr             : %.4f, %.4f' % (self.mr_rel[self.epoch], self.fmr_rel[self.epoch]))
        test_results.append(
            '--mrr, filtered mrr            : %.4f, %.4f' % (self.mrr_rel[self.epoch], self.fmrr_rel[self.epoch]))
        for hit in self.args.hits:
            test_results.append(
                '--hits_rel%d                        : %.4f ' % (hit, (self.hit_rel[(self.epoch, hit)])))
            test_results.append(
                '--filtered hits_rel%d               : %.4f ' % (hit, (self.fhit_rel[(self.epoch, hit)])))
        test_results.append("---------------------------------------------------------")
        test_results.append('')
        log.info("\n".join(test_results))

    def save_test_summary(self, path):
        if not os.path.isdir(str(path)):
            os.makedirs(str(path))
        files = os.listdir(str(path))
        l = len([f for f in files if self.args.method in f if 'Testing' in f])
        with open(str(Path(path) / (self.args.method + '_summary_' + str(l) + '.txt')), 'w') as fh:
            fh.write('----------------summary----------------\n')
            for key, val in self.args.__dict__.items():
                if 'gpu' in key:
                    continue
                if 'knowledge_graph' in key:
                    continue
                if not isinstance(val, str):
                    if isinstance(val, list):
                        v_tmp = '['
                        for i, v in enumerate(val):
                            if i == 0:
                                v_tmp += str(v)
                            else:
                                v_tmp += ',' + str(v)
                        v_tmp += ']'
                        val = v_tmp
                    else:
                        val = str(val)
                fh.write(key + ':' + val + '\n')
            fh.write('-----------------------------------------\n')
            fh.write("Total Training triplets   :%d\n" % len(self.data.triplets_withtype['train']))
            fh.write("Total validation triplets    :%d\n" % len(self.data.triplets_withtype['valid']))
            fh.write("Total Testing triplets :%d\n" % len(self.data.triplets_withtype['test']))
            fh.write("Total Entities           :%d\n" % self.data.geoent_num)
            fh.write("Total Relations          :%d\n" % self.data.georel_num)
            fh.write("---------------------------------------------")

    def get_curr_scores(self):
        scores = {'mr_rel': self.mr_rel[self.epoch],
                  'fmr_rel': self.fmr_rel[self.epoch],
                  'mrr_rel': self.mrr_rel[self.epoch],
                  'fmrr_rel': self.fmrr_rel[self.epoch],
                  'hits@1_rel': self.hit_rel[(self.epoch, 1)],
                  'filtered_hits@1_rel': self.fhit_rel[(self.epoch, 1)],
                  'hits@3_rel': self.hit_rel[(self.epoch, 3)],
                  'filtered_hits@3_rel': self.fhit_rel[(self.epoch, 3)],
                  'hits@5_rel': self.hit_rel[(self.epoch, 5)],
                  'filtered_hits@5_rel': self.fhit_rel[(self.epoch, 5)],
                  'hits@10_rel': self.hit_rel[(self.epoch, 10)],
                  'filtered_hits@10_rel': self.fhit_rel[(self.epoch, 10)]}
        return scores


class Evaluator:
    logger = logging.getLogger(__name__)

    def __init__(self, ins_model, type_model, pair_model, data, args):
        self.args = args
        self.ins_model = ins_model
        self.type_model = type_model
        self.pair_model = pair_model
        self.valid_data = data.triplets_withtype["valid"]
        self.test_data = data.triplets_withtype["test"]
        self.valid_type = data.triplets_type["valid"]
        self.test_type = data.triplets_type["test"]
        self.valid_pair = data.pairs_all["valid"]
        self.test_pair = data.pairs_all["test"]

        self.metric = Metric(data, args)
        self.device = args.device
        self.test_num = args.test_num
        self.ent_num = data.geoent_num
        self.rel_num = data.georel_num
        self.type_ent_num = data.geotype_ent_num
        self.type_rel_num = data.geotype_rel_num
        self.save_path = Path(args.model_save_path) / args.method / args.save_times

    def test_rel_rank(self, h, t, topk=-1):
        h_batch = torch.LongTensor([h]).repeat([self.rel_num]).to(self.device)
        relation_array = torch.LongTensor(list(range(self.rel_num))).to(self.device)
        t_batch = torch.LongTensor([t]).repeat([self.rel_num]).to(self.device)
        preds = self.ins_model.forward(h_batch, relation_array, t_batch, BatchType.SINGLE)
        _, rank = torch.topk(preds.view(-1), k=topk, largest=False)
        return rank

    def test_kg_ins(self, h, t):
        rrank = self.test_rel_rank(h, t, self.rel_num)
        return rrank

    def test(self, data_ins, data_ins_num, epoch=None):
        self.metric.reset()
        res_ins = []
        for i in tqdm(range(data_ins_num)):
            h, r, t = data_ins[i].h, data_ins[i].r, data_ins[i].t
            rrank = self.test_kg_ins(h, t)
            result_ins = [rrank.detach().cpu().numpy(), h, r, t, epoch]
            res_ins.append(result_ins)

        res_ins = [list(x) for x in zip(*res_ins)]
        self.metric.append_result(res_ins)
        self.metric.settle()
        self.metric.display_summary()
        if epoch >= self.args.epochs - 1:
            self.metric.save_test_summary(self.save_path)
        return self.metric.get_curr_scores()

    def mini_test(self, epoch=None):
        valid_num = min(self.test_num, len(self.valid_data))
        log.info("Mini-Testing on [%d/%d] triplets in the valid set." % (valid_num, len(self.valid_data)))
        return self.test(self.valid_data, valid_num, epoch=epoch)

    def full_test(self, epoch=None):
        test_num = len(self.test_data)
        log.info("Full-Testing on [%d/%d] triplets in the test set." % (test_num, len(self.test_data)))
        return self.test(self.test_data, test_num, epoch=epoch)