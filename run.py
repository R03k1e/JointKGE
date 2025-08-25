"""
Main training script for JT-KGE.

Parses hyper-parameters, loads data, builds the model and starts training.
"""

import argparse
from data_helper import GeoKG
from train import Join_trainer
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Restrict GPU visibility (single-GPU training)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def GeoKGArgparse(args=None):
    """
    Define and parse command-line / programmatic arguments for JT-KGE.

    Returns
    -------
    argparse.Namespace
        Parsed arguments ready for downstream use.
    """

    parser = argparse.ArgumentParser(description='JT-KGE')

    # ------------------ model & training ------------------
    parser.add_argument('--method', type=str, default='DistMult',
                        help='Embedding method to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total number of training epochs')
    parser.add_argument('--test_step', type=int, default=20,
                        help='Evaluate every N epochs')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Mini-test sample size (validation triplets)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda / cpu')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early-stopping patience (epochs without improvement)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Checkpoint path to resume training (optional)')
    parser.add_argument('--model_save_path', type=str, default='./model',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_times', type=str, default='savefold_name',
                        help='Sub-folder suffix for this run')

    # ------------------ optimization ------------------
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--dim_ins', type=int, default=200,
                        help='Instance-level embedding dimension')
    parser.add_argument('--dim_type', type=int, default=200,
                        help='Type-level embedding dimension')
    parser.add_argument('--batch_size_type', type=int, default=256,
                        help='Batch size for type-level triples')
    parser.add_argument('--batch_size_ins', type=int, default=256,
                        help='Batch size for instance triples')
    parser.add_argument('--batch_size_pair', type=int, default=256,
                        help='Batch size for entity-type pairs')

    # ------------------ loss hyper-parameters ------------------
    parser.add_argument('--margin_ins', type=float, default=12,
                        help='Margin for instance triples')
    parser.add_argument('--margin_type', type=float, default=12,
                        help='Margin for type triples')
    parser.add_argument('--margin_pair', type=float, default=20,
                        help='Margin for entity-type pairs')
    parser.add_argument('--weight_G', type=float, default=1,
                        help='Weight for geometric loss')
    parser.add_argument('--weight_J', type=float, default=1,
                        help='Weight for joint loss')

    # ------------------ data folds ------------------
    parser.add_argument('--fold_ins', type=int, default=1)
    parser.add_argument('--fold_type', type=int, default=5)
    parser.add_argument('--fold_pair', type=int, default=1)

    # ------------------ misc ------------------
    parser.add_argument('--l1', type=int, default=2,
                        help='Norm order for distance (1 or 2)')
    parser.add_argument('--hits', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='Hit@k evaluation metrics')
    parser.add_argument('--neg_rate', type=float, default=5,
                        help='Negative samples per positive')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Self-adversarial temperature')
    parser.add_argument('--training_order', type=str, default='0,1,2',
                        help='Training order of sub-tasks (comma-separated indices)')

    return parser.parse_args(args)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    pid = os.getpid()
    print('pid:', pid)

    prefix = './sample data'

    # ------------------ instance triples ------------------
    pathcollect_ins = {
        'train': f'{prefix}/train.txt',
        'valid': f'{prefix}/valid.txt',
        'test':  f'{prefix}/test.txt'
    }

    # ------------------ type triples ------------------
    pathcollect_type = {
        'train': f'{prefix}/train_type.txt',
        'valid': f'{prefix}/valid_type.txt',
        'test':  f'{prefix}/test_type.txt'
    }

    # ------------------ entity-type pairs ------------------
    pathcollect_pair = {
        'train': f'{prefix}/train.txt',
        'valid': f'{prefix}/valid.txt',
        'test':  f'{prefix}/test.txt'
    }

    # ------------------ load data ------------------
    dh = GeoKG()
    dh.prepare_data(pathcollect_ins, pathcollect_type, pathcollect_pair)

    # ------------------ run training ------------------
    args = GeoKGArgparse(['--method', 'CompoundE', '--device', 'cuda'])
    trainer = Join_trainer(args=args)
    trainer.build(data=dh)
    trainer.join_train()
