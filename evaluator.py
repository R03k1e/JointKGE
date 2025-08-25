"""
Evaluation pipeline for knowledge-graph embedding models.

Provides:
1. Metric tracking (MR, MRR, Hits@k) for relations, types, and entity–type pairs.
2. Rank computation with raw and filtered settings.
3. Early/minimal testing on validation set and full testing on test set.
"""

from tqdm import tqdm
import numpy as np
import timeit
import torch
import logging
import os
from pathlib import Path
from data_helper import BatchType

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metric container
# ------------------------------------------------------------------
class Metric:
    """
    Accumulates and reports ranking metrics for triples, types, and pairs.
    """

    def __init__(self, data, args):
        # Raw & filtered metrics
        self.mr = {}
        self.fmr = {}
        self.mrr = {}
        self.fmrr = {}

        # Relation-level
        self.mr_rel = {}
        self.fmr_rel = {}
        self.mrr_rel = {}
        self.fmrr_rel = {}

        # Type-level
        self.mr_type = {}
        self.mrr_type = {}
        self.fmr_type = {}
        self.fmrr_type = {}

        # Pair-level
        self.mr_pair = {}
        self.mrr_pair = {}
        self.fmr_pair = {}
        self.fmrr_pair = {}

        # Hit rates @k
        self.hit = {}
        self.fhit = {}
        self.hit_rel = {}
        self.fhit_rel = {}
        self.type_hit = {}
        self.type_fhit = {}
        self.pair_hit = {}
        self.pair_fhit = {}

        # Data & config references
        self.data = data
        self.args = args
        self.epoch = None
        self.reset()

        # Quick access to KG indices
        self.hr_t_idx = data.hr_t_idx
        self.tr_h_idx = data.tr_h_idx
        self.ht_r_idx = data.ht_r_idx
        self.hr_t_type_idx = data.hr_t_type_idx
        self.tr_h_type_idx = data.tr_h_type_idx
        self.ent_type_pairs_train = data.ent_type_pairs_train

    # ----------------------------------------------------------
    # House-keeping
    # ----------------------------------------------------------
    def reset(self):
        """Clear all rank lists and timers."""
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

    # ----------------------------------------------------------
    # Rank computation helpers
    # ----------------------------------------------------------
    def _compute_rank(self, candidates, true_ids, filter_set_fn):
        """
        Generic rank calculator.

        candidates : list[list[int]]
            Sorted candidate scores for each example.
        true_ids : list[int]
            Ground-truth id for each example.
        filter_set_fn : callable(idx) -> set[int]
            Returns the set of ids to filter out.

        Returns raw_rank, filtered_rank lists.
        """
        raw_ranks = []
        filt_ranks = []
        for idx, cand in enumerate(candidates):
            true_id = true_ids[idx]
            raw_rank = 0
            filt_rank = 0
            for j, c in enumerate(cand, start=1):
                if c != true_id:
                    raw_rank += 1
                    filt_rank += 1
                    if c in filter_set_fn(idx):
                        filt_rank -= 1
                else:
                    break
            raw_ranks.append(raw_rank)
            filt_ranks.append(filt_rank)
        return raw_ranks, filt_ranks

    # Wrappers for specific prediction targets
    def get_head_rank(self, head_candidate, h, r, t):
        return self._compute_rank(
            head_candidate, h,
            filter_set_fn=lambda i: self.tr_h_idx[(t[i], r[i])]
        )

    def get_tail_rank(self, tail_candidate, h, r, t):
        return self._compute_rank(
            tail_candidate, t,
            filter_set_fn=lambda i: self.hr_t_idx[(h[i], r[i])]
        )

    def get_rel_rank(self, rel_candidate, h, r, t):
        return self._compute_rank(
            rel_candidate, r,
            filter_set_fn=lambda i: self.ht_r_idx[(h[i], t[i])]
        )

    def get_headtype_rank(self, head_candidate, h, r, t):
        return self._compute_rank(
            head_candidate, h,
            filter_set_fn=lambda i: self.tr_h_type_idx[(t[i], r[i])]
        )

    def get_tailtype_rank(self, tail_candidate, h, r, t):
        return self._compute_rank(
            tail_candidate, t,
            filter_set_fn=lambda i: self.hr_t_type_idx[(h[i], r[i])]
        )

    def get_type_rank(self, type_candidate, ent, ent_type):
        def filter_fn(i):
            return self.ent_type_pairs_train.get(ent[i], set())
        return self._compute_rank(type_candidate, ent_type, filter_fn)

    # ----------------------------------------------------------
    # Result aggregation
    # ----------------------------------------------------------
    def append_result(self, res_ins):
        """Store computed relation ranks from a batch."""
        pred_rel = res_ins[0]
        h, r, t = res_ins[1], res_ins[2], res_ins[3]
        self.epoch = res_ins[4][0]

        r_rank, f_r_rank = self.get_rel_rank(pred_rel, h, r, t)
        self.rank_rel.append(r_rank)
        self.f_rank_rel.append(f_r_rank)

    def settle(self):
        """Compute final metrics from accumulated ranks."""
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

    # ----------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------
    def display_summary(self):
        """Log concise test results."""
        stop_time = timeit.default_timer()
        lines = [
            "",
            f"------Test Results : Epoch: {self.epoch} --- time: {stop_time - self.start_time:.2f}------------",
            "------Test Results for ins_rel------------",
            f'--# of entities, # of relations: {self.data.geoent_num}, {self.data.georel_num}',
            f'--mr,  filtered mr             : {self.mr_rel[self.epoch]:.4f}, {self.fmr_rel[self.epoch]:.4f}',
            f'--mrr, filtered mrr            : {self.mrr_rel[self.epoch]:.4f}, {self.fmrr_rel[self.epoch]:.4f}',
        ]
        for hit in self.args.hits:
            lines.append(f'--hits_rel{hit}                        : {self.hit_rel[(self.epoch, hit)]:.4f}')
            lines.append(f'--filtered hits_rel{hit}               : {self.fhit_rel[(self.epoch, hit)]:.4f}')
        lines.append("---------------------------------------------------------")
        lines.append("")
        log.info("\n".join(lines))

    def save_test_summary(self, path):
        """Persist configuration and final metrics to disk."""
        os.makedirs(str(path), exist_ok=True)
        existing = [f for f in os.listdir(str(path)) if self.args.method in f and 'Testing' in f]
        filename = str(Path(path) / f"{self.args.method}_summary_{len(existing)}.txt")
        with open(filename, 'w') as fh:
            fh.write('----------------summary----------------\n')
            for k, v in self.args.__dict__.items():
                if 'gpu' in k or 'knowledge_graph' in k:
                    continue
                if isinstance(v, list):
                    v = '[' + ','.join(map(str, v)) + ']'
                else:
                    v = str(v)
                fh.write(f"{k}:{v}\n")
            fh.write('-----------------------------------------\n')
            fh.write(f"Total Training triplets   :{len(self.data.triplets_withtype['train'])}\n")
            fh.write(f"Total validation triplets :{len(self.data.triplets_withtype['valid'])}\n")
            fh.write(f"Total Testing triplets    :{len(self.data.triplets_withtype['test'])}\n")
            fh.write(f"Total Entities            :{self.data.geoent_num}\n")
            fh.write(f"Total Relations           :{self.data.georel_num}\n")
            fh.write("---------------------------------------------\n")

    def get_curr_scores(self):
        """Return a dict of current metrics for external use (e.g., early stopping)."""
        return {
            'mr_rel': self.mr_rel[self.epoch],
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
            'filtered_hits@10_rel': self.fhit_rel[(self.epoch, 10)],
        }


# ------------------------------------------------------------------
# High-level evaluator
# ------------------------------------------------------------------
class Evaluator:
    """
    Coordinates model evaluation on validation and test sets.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, ins_model, type_model, pair_model, data, args):
        self.args = args
        self.ins_model = ins_model
        self.type_model = type_model
        self.pair_model = pair_model

        # Data splits
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

    # ----------------------------------------------------------
    # Relation prediction
    # ----------------------------------------------------------
    def test_rel_rank(self, h, t, topk=-1):
        """Score all relations for (h,t) and return top-k indices."""
        h_batch = torch.LongTensor([h]).repeat(self.rel_num).to(self.device)
        relation_array = torch.arange(self.rel_num, device=self.device)
        t_batch = torch.LongTensor([t]).repeat(self.rel_num).to(self.device)
        scores = self.ins_model(h_batch, relation_array, t_batch, BatchType.SINGLE)
        _, rank = torch.topk(scores.view(-1), k=topk, largest=False)
        return rank

    def test_kg_ins(self, h, t):
        """Compute relation ranks for a single (h,t) pair."""
        return self.test_rel_rank(h, t, self.rel_num)

    # ----------------------------------------------------------
    # Main test loops
    # ----------------------------------------------------------
    def test(self, data_ins, data_ins_num, epoch=None):
        """Run full evaluation on provided triple set."""
        self.metric.reset()
        results = []
        for i in tqdm(range(data_ins_num), desc="Eval"):
            h, r, t = data_ins[i].h, data_ins[i].r, data_ins[i].t
            rrank = self.test_kg_ins(h, t)
            results.append([rrank.detach().cpu().numpy(), h, r, t, epoch])

        # Transpose to list-of-lists
        results = [list(x) for x in zip(*results)]
        self.metric.append_result(results)
        self.metric.settle()
        self.metric.display_summary()
        if epoch is not None and epoch >= self.args.epochs - 1:
            self.metric.save_test_summary(self.save_path)
        return self.metric.get_curr_scores()

    def mini_test(self, epoch=None):
        """Quick evaluation on a subset of the validation set."""
        valid_num = min(self.test_num, len(self.valid_data))
        log.info("Mini-Testing on [%d/%d] triplets in the valid set.", valid_num, len(self.valid_data))
        return self.test(self.valid_data, valid_num, epoch=epoch)

    def full_test(self, epoch=None):
        """Evaluate on the entire test set."""
        test_num = len(self.test_data)
        log.info("Full-Testing on [%d/%d] triplets in the test set.", test_num, len(self.test_data))
        return self.test(self.test_data, test_num, epoch=epoch)
