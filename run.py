import argparse
from data_helper import GeoKG
from train import Join_trainer
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def GeoKGArgparse(args=None):
    # parameter parsing
    parser = argparse.ArgumentParser(description='JT-KGE')
    parser.add_argument('--method', type=str, default='DistMult', help='embedding method')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--test_step', type=int, default=20, help='number of epoch for test')
    parser.add_argument('--test_num', type=int, default=1000, help='number of triplets for mini-test')
    parser.add_argument('--device', type=str, default='cuda', help='GPU Usage')
    parser.add_argument('--patience', type=int, default=3, help='times for stop the train')
    parser.add_argument('--load_model', type=str, default=None, help='path to load model')
    parser.add_argument('--model_save_path', type=str, default='./model', help='save model path')
    parser.add_argument('--save_times', type=str, default='savefold_name', help='times to save')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dim_ins', type=int, default=200, help='G-ins dimension')
    parser.add_argument('--dim_type', type=int, default=200, help='G-type dimension')
    parser.add_argument('--batch_size_type', type=int, default=256, help='number of batch for G-type')
    parser.add_argument('--batch_size_ins', type=int, default=256, help='number of batch for G-ins')
    parser.add_argument('--batch_size_pair', type=int, default=256, help='number of batch for p-type')
    parser.add_argument('--margin_ins', type=float, default=12, help='instance training margin')
    parser.add_argument('--margin_type', type=float, default=12, help='type training margin')
    parser.add_argument('--margin_pair', type=float, default=20, help='pair training margin')
    parser.add_argument('--weight_G', type=float, default=1)
    parser.add_argument('--weight_J', type=float, default=1)
    parser.add_argument('--fold_ins', type=int, default=1)
    parser.add_argument('--fold_type', type=int, default=5)
    parser.add_argument('--fold_pair', type=int, default=1)
    parser.add_argument('--l1', type=int, default=2)
    parser.add_argument('--hits', type=int, default=[1, 3, 5, 10], nargs='+', help='metrics')
    parser.add_argument('--neg_rate', type=float, default=5, help='The number of negative samples generated per positve one.')
    parser.add_argument('--alpha', type=float, default=1.0, help='The alpha used in self-adversarial negative sampling.')
    parser.add_argument('--training_order', type=str, default='0,1,2')
    return parser.parse_args(args)

if __name__ == '__main__':
    pid = os.getpid()
    print('pid: ', pid)
    prefix = './sample data'
    #G-ins, sample data
    train_triple_file = 'train.txt'
    dev_triple_file = 'valid.txt'
    test_triple_file = 'test.txt'
    pathcollect_ins = {'train': prefix + "/" + train_triple_file,
                   'valid': prefix + "/" + dev_triple_file,
                   'test': prefix + "/" + test_triple_file}

    #G-type
    train_triple_file_type = 'train_type.txt'
    dev_triple_file_type = 'valid_type.txt'
    test_triple_file_type = 'test_type.txt'
    pathcollect_type = {'train': prefix + "/" + train_triple_file_type,
                        'valid': prefix + "/" + dev_triple_file_type,
                        'test': prefix + "/" + test_triple_file_type}
    #P-type, sample data
    train_triple_file_pair = 'train.txt'
    dev_triple_file_pair = 'valid.txt'
    test_triple_file_pair = 'test.txt'
    pathcollect_pair = {'train': prefix + "/" + train_triple_file_pair,
                        'valid': prefix + "/" + dev_triple_file_pair,
                        'test': prefix + "/" + test_triple_file_pair}

    dh = GeoKG()
    dh.prepare_data(pathcollect_ins, pathcollect_type, pathcollect_pair)
    args = GeoKGArgparse(['--method', 'CompoundE', '--device', 'cuda'])

    kg_train = Join_trainer(args=args)
    kg_train.build(data=dh)
    kg_train.join_train()