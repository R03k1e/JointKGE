from enum import Enum


class Monitor(Enum):
    MEAN_RANK_REL = "mr_rel"
    FILTERED_MEAN_RANK_REL = "fmr_rel"
    MEAN_RECIPROCAL_RANK_REL = "mrr_rel"
    FILTERED_MEAN_RECIPROCAL_RANK_REL = "fmrr_rel"
