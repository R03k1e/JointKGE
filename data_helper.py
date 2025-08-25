"""
Core data structures and data-loading utilities for a Geographic Knowledge Graph (GeoKG).
The module supports:
1. Triple-only facts (h, r, t)
2. Triples augmented with entity types
3. Entity-type pairs
4. Negative sampling for knowledge-graph embedding training
"""

import logging
import copy
from collections import defaultdict
import numpy as np
import torch
from enum import Enum

log = logging.getLogger(__name__)


"""
Enumeration of supported batch types used during negative sampling.
"""
class BatchType(Enum):
    HEAD_BATCH = 0  # Corrupt the head entity
    TAIL_BATCH = 1  # Corrupt the tail entity
    SINGLE = 2      # Placeholder (unused)


"""
Minimal container for a knowledge-graph triple.
No type information is stored here.
"""
class Triple_only:
    def __init__(self, h, r, t):
        self.h = h  # head entity id
        self.r = r  # relation id
        self.t = t  # tail entity id

    def set_ids(self, h, r, t):
        """Bulk-set entity/relation ids."""
        self.h = h
        self.r = r
        self.t = t

    def set_value(self, type, value):
        """Update either head ('h') or tail ('t')."""
        if type == 'h':
            self.h = value
        if type == 't':
            self.t = value


"""
Container for an entity and its associated list of types.
"""
class Pair:
    def __init__(self, ent, type):
        self.ent = ent       # entity id
        self.type = type     # list[str] of type labels

    def set_ids(self, ent, type):
        self.ent = ent
        self.type = type

    def set_value(self, mark, value):
        """Update the type list."""
        if mark == 'type':
            self.type = value


"""
Container for a triple plus the type labels of its head and tail entities.
"""
class Triple_withtype:
    def __init__(self, h, r, t, h_type, t_type):
        self.h = h          # head entity id (string)
        self.r = r          # relation id (string)
        self.t = t          # tail entity id (string)
        self.h_type = h_type
        self.t_type = t_type

    def set_ids(self, h, r, t, h_type, t_type):
        """Bulk-set all fields."""
        self.h = h
        self.r = r
        self.t = t
        self.h_type = h_type
        self.t_type = t_type

    def set_value(self, type, value):
        """Update any single field."""
        if type == 'h':
            self.h = value
        if type == 't':
            self.t = value
        if type == 'h_type':
            self.h_type = value
        if type == 't_type':
            self.t_type = value


"""
Main class that holds all KG data, builds indices, and provides negative-sampling helpers.
"""
class GeoKG():
    def __init__(self):
        """
        Vocabulary & frequency statistics
        """
        self.geoent_tokens = {}             # id -> entity string
        self.geoent_index = {}              # entity string -> id
        self.geoent_num = 0                 # number of entities
        self.geoent_forupdate = set([])     # set of entity strings seen in triples
        self.geoent_head_count = defaultdict(int)
        self.geoent_tail_count = defaultdict(int)
        self.geoent_count = defaultdict(int)  # head+tail occurrences

        self.georel_tokens = {}
        self.georel_index = {}
        self.georel_num = 0
        self.georel_forupdate = set([])
        self.georel_count = defaultdict(int)

        # Type-level KG
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

        """
        Raw data containers
        """
        self.triplets_withtype = {'train': [], 'test': [], 'valid': []}   # Triple_only objects
        self.triplets_record_withtype = set([])                           # for deduplication
        self.triplets_type = {'train': [], 'test': [], 'valid': []}       # type-level triples
        self.triplets_record_type = set([])
        self.triplets_count = defaultdict(int)        # raw triple frequency
        self.triplets_type_count = defaultdict(int)   # type-level triple frequency

        self.pairs = {'train': [], 'test': [], 'valid': []}   # Pair objects
        self.pairs_all = {'train': [], 'test': [], 'valid': []}
        self.pairs_record = set([])
        self.pair_count = defaultdict(int)

        """
        Connection indices
        """
        self.ent_relation = defaultdict(set)
        self.relation_headentity = defaultdict(set)
        self.relation_tailentity = defaultdict(set)

        # Fast lookup indices used for negative sampling
        self.hr_t_idx = defaultdict(set)
        self.tr_h_idx = defaultdict(set)
        self.ht_r_idx = defaultdict(set)
        self.hr_t_type_idx = defaultdict(set)
        self.tr_h_type_idx = defaultdict(set)

        # Frequency tables for subsampling
        self.hr_t_freq = {}
        self.tr_h_freq = {}
        self.hr_t_type_freq = {}
        self.tr_h_type_freq = {}

        # Entity-to-type mappings
        self.ent_type_pairs = {}            # entity id -> set(type ids)
        self.triple2type = {'train': [], 'test': [], 'valid': []}
        self.triple2type_type = {'train': [], 'test': [], 'valid': []}
        self.ent_type_pairs_train = defaultdict(set)

        # Type embeddings bookkeeping
        self.type = {}
        self.type_embed = {}
        self.type_embed_padded = np.array([0])
        self.type_length = 100
        self.type_index = np.array([0])

    # ---------------------------------------------------------------------
    # Public high-level entry point
    # ---------------------------------------------------------------------

    def prepare_data(self, paths_ins, paths_type, paths_pair):
        """
        Load all splits for:
          - raw triples (paths_ins)
          - type-level triples (paths_type)
          - entity-type pairs (paths_pair)

        After loading, indices are built and raw containers are deleted
        to save memory.
        """
        for key, path in paths_type.items():
            self.load_triplets_onlytype(key, path)
        for key, path in paths_ins.items():
            self.load_triplets(key, path)
        for key, path in paths_pair.items():
            self.load_pairs(key, path)

        self.load_ent_rel_type_idx()  # build vocabularies

        for key, path in paths_ins.items():
            self.load_triplets_idx(key)
        for key, path in paths_type.items():
            self.load_typetriplets_idx(key)
        for key, path in paths_pair.items():
            self.load_pairs_idx(key)

        del self.pairs  # free memory
        print("loaded all sample data!")

    # ---------------------------------------------------------------------
    # Vocabulary construction
    # ---------------------------------------------------------------------

    def load_ent_rel_type_idx(self):
        """
        Create sorted integer ids for all entities, relations, and types seen so far.
        """
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

    # ---------------------------------------------------------------------
    # Index builders
    # ---------------------------------------------------------------------

    def load_triplets_idx(self, key):
        """
        Convert string identifiers to integer ids for raw triples.
        Also builds hr_t, tr_h indices and frequency tables for training set.
        """
        init_cnt = 3  # small smoothing constant
        for t in self.triplets_withtype[key]:
            head = self.geoent_index[t.h]
            rel = self.georel_index[t.r]
            tail = self.geoent_index[t.t]

            if key == 'train':
                if (head, rel) not in self.hr_t_freq:
                    self.hr_t_freq[(head, rel)] = init_cnt
                if (tail, rel) not in self.tr_h_freq:
                    self.tr_h_freq[(tail, rel)] = init_cnt

                self.hr_t_idx[(head, rel)].add(tail)
                self.tr_h_idx[(tail, rel)].add(head)

                self.hr_t_freq[(head, rel)] += 1
                self.tr_h_freq[(tail, rel)] += 1

            t.set_ids(head, rel, tail)

    def load_typetriplets_idx(self, key):
        """
        Same as above but for type-level triples.
        """
        init_cnt = 3
        for t in self.triplets_type[key]:
            head = self.geotype_ent_index[t.h]
            rel = self.geotype_rel_index[t.r]
            tail = self.geotype_ent_index[t.t]

            if key == 'train':
                if (head, rel) not in self.hr_t_type_freq:
                    self.hr_t_type_freq[(head, rel)] = init_cnt
                if (tail, rel) not in self.tr_h_type_freq:
                    self.tr_h_type_freq[(tail, rel)] = init_cnt

                self.hr_t_type_idx[(head, rel)].add(tail)
                self.tr_h_type_idx[(tail, rel)].add(head)

                self.hr_t_type_freq[(head, rel)] += 1
                self.tr_h_type_freq[(tail, rel)] += 1

            t.set_ids(head, rel, tail)

    def load_pairs_idx(self, key):
        """
        Convert entity-type pairs to integer ids and populate entity->type indices.
        """
        # Clean malformed pairs
        for t in self.pairs[key]:
            if t.type == '[]':
                self.pairs[key].remove(t)
                continue
            if 'establishment' == t.type[0]:
                t.type.reverse()
                if 'natural_feature' in t.type and len(t.type) > 2:
                    del (t.type[0])
            while '' in t.type:
                t.type.remove('')

        # Build type strings for each triple
        triple2type_temp = []
        for tr in self.triple2type[key]:
            m = copy.deepcopy(tr)
            m1 = m[0].type[0] + '-' + m[1].type[0]
            triple2type_temp.append(m1)
        self.triple2type_type[key] = triple2type_temp

        # Populate entity->type mappings
        for t in self.pairs[key]:
            s = copy.deepcopy(t)
            ent_id = self.geoent_index[s.ent]
            type_id = self.geotype_ent_index[s.type[0]]

            if ent_id in self.ent_type_pairs:
                self.ent_type_pairs[ent_id].update([x for x in s.type])
            else:
                self.ent_type_pairs[ent_id] = set([x for x in s.type])

            if key == 'train':
                self.ent_type_pairs_train[ent_id].add(type_id)

            t.set_ids(ent_id, type_id)
            self.pairs_all[key].append(t)

        log.info('Loaded alltype pairs from %s : %s pairs', key, len(self.pairs_all[key]))

    # ---------------------------------------------------------------------
    # Raw data loaders
    # ---------------------------------------------------------------------

    def load_triplets_onlytype(self, key, path):
        """
        Load type-level triples (format: h r t).
        """
        triplets_type = []
        entities = set()
        relations = set()
        with open(path, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) != 3:
                    raise ValueError(f"Invalid triplet format at line {ln}: {line}")

                h, r, t = parts
                self.triplets_type_count[(h, r, t)] += 1
                triplet_obj = Triple_only(h, r, t)
                triplets_type.append(triplet_obj)
                self.triplets_record_type.add(triplet_obj)

                entities.update([h, t])
                relations.add(r)

                self.geotype_head_count[h] += 1
                self.geotype_tail_count[t] += 1
                self.geotype_ent_count[h] += 1
                self.geotype_ent_count[t] += 1
                self.geotype_rel_count[r] += 1

        self.geoent_forupdate_type.update(entities)
        self.georel_forupdate_type.update(relations)
        self.triplets_type[key] = triplets_type
        log.info('Loaded type triplets from %s : %s triplets, %s ents, %s rels',
                 key, len(triplets_type), len(entities), len(relations))

    def load_triplets(self, key, path):
        """
        Load normal triples (format: h r t h_type t_type).
        """
        triplets_withtype = []
        entities = set()
        relations = set()

        with open(path, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) < 5:
                    raise ValueError(f"Invalid triplet format at line {ln}: {line}")

                h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
                triplet_obj = Triple_only(h, r, t)
                self.triplets_count[(h, r, t)] += 1
                self.geoent_head_count[h] += 1
                self.geoent_tail_count[t] += 1
                self.geoent_count[h] += 1
                self.geoent_count[t] += 1
                self.georel_count[r] += 1

                triplets_withtype.append(triplet_obj)
                self.triplets_record_withtype.add(triplet_obj)

                entities.update([h, t])
                relations.add(r)

                # Reverse indices for head/tail type extraction
                self.ent_relation[h].add((r, parts[3].strip(), 'head'))
                self.ent_relation[t].add((r, parts[4].strip(), 'tail'))
                self.relation_headentity[r].add((h, parts[3].strip()))
                self.relation_tailentity[r].add((t, parts[4].strip()))

        self.geoent_forupdate.update(entities)
        self.georel_forupdate.update(relations)
        self.triplets_withtype[key] = triplets_withtype
        log.info('Loaded triplets from %s : %s triplets, %s ents, %s rels',
                 key, len(triplets_withtype), len(entities), len(relations))

    def load_pairs(self, key, path):
        """
        Load entity-type pairs (format: h r t h_type t_type).
        The actual triple structure is ignored; only entity-type pairs are kept.
        """
        geo_type = set()
        geo_ent = set()
        pairs = []
        triplets_type = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split(' ')
                h_type = parts[3].strip()
                t_type = parts[4].strip()

                self.pair_count[(parts[0].strip(), h_type)] += 1
                self.pair_count[(parts[2].strip(), t_type)] += 1

                # Parse multi-type strings
                def parse_type(t_str):
                    if len(t_str) > 2:
                        return eval(
                            (t_str[:t_str.find(']') + 1] +
                             t_str[t_str.find(']') + 1:].replace(",", "', '") +
                             "'").replace("']", '') + ']'
                        )
                    else:
                        return t_str

                h_types = parse_type(h_type)
                t_types = parse_type(t_type)

                pair_h = Pair(parts[0].strip(), h_types)
                pair_t = Pair(parts[2].strip(), t_types)
                pairs.extend([pair_h, pair_t])
                triplets_type.append((pair_h, pair_t))
                self.pairs_record.update([pair_h, pair_t])
                geo_ent.update([parts[0].strip(), parts[2].strip()])
                geo_type.update(h_types)
                geo_type.update(t_types)

        self.pairs[key] = pairs
        self.triple2type[key] = triplets_type
        log.info('Loaded pairs from %s : %s pairs, %s ents, %s types',
                 key, len(pairs), len(geo_ent), len(geo_type))

    # ---------------------------------------------------------------------
    # Negative sampling utilities
    # ---------------------------------------------------------------------

    def corrupt_batch_triplets_ins(self, neg_rate, data, batch_type):
        """
        Generate negative samples for instance-level triples.
        Returns lists of torch tensors (one per positive triple) and subsampling weights.
        """
        positive_triplets = {(t.h, t.r, t.t): 1 for t in data}
        neg_batch_tr = []
        subsampling_weights = []

        for t in data:
            neg_triplets = []
            head, rel, tail = t.h, t.r, t.t

            subsampling_weight = self.hr_t_freq[(head, rel)] + self.tr_h_freq[(tail, rel)]
            subsampling_weight = torch.sqrt(1 / torch.tensor([subsampling_weight], dtype=torch.float))

            for _ in range(neg_rate):
                if batch_type == BatchType.TAIL_BATCH:
                    idx_replace = np.random.randint(self.geoent_num)
                    while (t.h, t.r, idx_replace) in positive_triplets:
                        idx_replace = np.random.randint(self.geoent_num)
                else:  # HEAD_BATCH
                    idx_replace = np.random.randint(self.geoent_num)
                    while (idx_replace, t.r, t.t) in positive_triplets:
                        idx_replace = np.random.randint(self.geoent_num)
                neg_triplets.append(idx_replace)

            subsampling_weights.append(subsampling_weight)
            neg_batch_tr.append(torch.from_numpy(np.array(neg_triplets, dtype=np.int64)))

        return neg_batch_tr, subsampling_weights

    def corrupt_batch_triplets_type(self, neg_rate, data, batch_type):
        """
        Negative sampling for type-level triples.
        Uses pre-built indices to avoid known positives.
        """
        neg_batch_triplets = []
        subsampling_weights = []

        for t in data:
            head, rel, tail = t.h, t.r, t.t
            subsampling_weight = self.hr_t_type_freq[(head, rel)] + self.tr_h_type_freq[(tail, rel)]
            subsampling_weight = torch.sqrt(1 / torch.tensor([subsampling_weight], dtype=torch.float))

            neg_triplets = []
            needed = neg_rate
            while needed > 0:
                candidates = np.random.randint(self.geotype_ent_num, size=needed * 2)
                if batch_type == BatchType.HEAD_BATCH:
                    mask = np.in1d(candidates, self.tr_h_type_idx[(tail, rel)], assume_unique=True, invert=True)
                elif batch_type == BatchType.TAIL_BATCH:
                    mask = np.in1d(candidates, self.hr_t_type_idx[(head, rel)], assume_unique=True, invert=True)
                else:
                    raise ValueError(f'Invalid BatchType: {batch_type}')
                candidates = candidates[mask]
                neg_triplets.append(candidates)
                needed -= candidates.size
            neg_triplets = np.concatenate(neg_triplets)[:neg_rate]

            neg_batch_triplets.append(torch.from_numpy(neg_triplets.astype(np.int64)))
            subsampling_weights.append(subsampling_weight)

        return neg_batch_triplets, subsampling_weights

    def corrupt_batch_pairs(self, neg_rate, data):
        """
        Negative sampling for entity-type pairs.
        Samples type ids that the entity does NOT have in training.
        """
        neg_batch_pairs = []
        for t in data:
            ent, type = t.ent, t.type
            needed = neg_rate
            neg_triplets = []
            while needed > 0:
                candidates = np.random.randint(self.geotype_ent_num, size=needed * 2)
                mask = np.in1d(candidates, self.ent_type_pairs_train[ent], assume_unique=True, invert=True)
                candidates = candidates[mask]
                neg_triplets.append(candidates)
                needed -= candidates.size
            neg_triplets = np.concatenate(neg_triplets)[:neg_rate]
            neg_batch_pairs.append(torch.from_numpy(neg_triplets.astype(np.int64)))
        return neg_batch_pairs
