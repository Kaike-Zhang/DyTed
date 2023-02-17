import argparse
import torch
import os

parser = argparse.ArgumentParser(description='DyTed')

parser.add_argument('--model', type=str, default='EvolveGCN', help='model name')
# 1.dataset
parser.add_argument('--dataset', type=str, default='uci', help='dataset')

parser.add_argument('--node_num', type=int, default=1809, help='dim of input feature')
parser.add_argument('--time_steps', type=int, nargs='?', default=13, help="total time steps used for train, eval and test")

parser.add_argument('--pre_defined_feature', default=None, help='pre-defined node feature')
parser.add_argument('--use_trainable_feature', default=True, help='pre-defined node feature')
parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
parser.add_argument('--out_feat', type=int, default=64, help='dim of output feature')
parser.add_argument('--log_path', type=str, default='./log', help='log path')

# 2.model
parser.add_argument('--heads', type=int, default=4, help='attention heads.')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate (1 - keep probability).')

parser.add_argument('--dis_in', type=int, default=128, help='discriminator in feat.')
parser.add_argument('--dis_hid', type=int, default=64, help='discriminator hidden feat.')

# 3.experiment
parser.add_argument('--use_gpu', type=bool, default=True, help='use gru or not')
parser.add_argument('--device', type=str, default='cpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')

parser.add_argument('--min_epoch', type=int, default=50, help='min epoch')
parser.add_argument('--max_epoch', type=int, default=300, help='number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size (# nodes)')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')

parser.add_argument('--log_interval', type=int, default=20, help='log interval, default: 20,[20,40,...]')
parser.add_argument('--dis_weight', type=float, default=0.6, help='discriminator weight')
parser.add_argument('--ti_weight', type=float, default=0.3, help='temporal invariant weight')

parser.add_argument('--dis_start', type=int, default=10, help='start training discriminator')
parser.add_argument('--dis_epoch', type=int, default=5, help='training discriminator each epoch')
parser.add_argument('--dis_sample_num', type=int, default=500, help='number of positive samples')

parser.add_argument('--t', type=float, default=0.1, help='hot InfoNce')

# LSTM
parser.add_argument('--cov_num', type=int, default=1, help='layers of  gcn cov.')
parser.add_argument('--in_feature_list', type=list, default=[143], help='in feature of each layer.')
parser.add_argument('--gcn_drop', type=float, default=0.2, help='dropout of gcn.')

# EvolveGCN
parser.add_argument('--nhid', type=int, default=64, help='dim of hidden embedding')
parser.add_argument('--nout', type=int, default=16, help='dim of output embedding')
parser.add_argument('--testlength', type=int, default=3, help='length for test, default:3')
parser.add_argument('--egcn_type', type=str, default='EGCNH', help='Type of EGCN: EGCNH or EGCNO')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

# set the running device
if torch.cuda.is_available() and args.use_gpu:
    print('using gpu:{} to train the model'.format(args.device_id))
    args.device_id = list(range(torch.cuda.device_count()))
    args.device = torch.device("cuda:{}".format(0))

else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

if args.use_trainable_feature:
    print('using trainable feature')


if args.model == "EvolveGCN":
    args.nout = args.out_feat

elif args.model == "LSTMGCN":
    args.nout = args.out_feat
    args.in_feature_list[0] = args.nfeat


