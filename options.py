import argparse

parser = argparse.ArgumentParser(description='D2-Net')
parser.add_argument('--lr', type=float, default=0.0001,help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=20, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model-name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num-class', type=int, default=20, help='number of classes (default: )')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on')
parser.add_argument('--max-seqlen', type=int, default=750, help='maximum sequence length during training (default: 750)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=20000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--summary', default='no summary', help='Summary of expt')
parser.add_argument('--activity-net', action='store_true', default=False, help='ActivityNet v1.2 dataset')
parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda')
parser.add_argument('--grouping-wt', type=float, default=0.01, help='Compactness grouping weight')
parser.add_argument('--lds-wt', type=float, default=0.1, help='LDS weight')
parser.add_argument('--ldv-wt', type=float, default=0.1, help='LDV weight')
parser.add_argument('--pdmi-iter', type=int, default=1000, help='Start of pdmi itr')
parser.add_argument('--test-iter', type=int, default=500, help='Test itr interval')

