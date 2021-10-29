from __future__ import print_function
import argparse
import os
import torch
from model import Model
from video_dataset import Dataset
from test import test
from train import train
import utils
from tensorboard_logger import Logger
import options
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.optim as optim

if __name__ == '__main__':
    print('Started')
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda") if args.cuda else torch.device("cpu")

    dataset = Dataset(args)
    
    os.system('mkdir -p ./ckpt/')
    os.system('mkdir -p ./logs/' + args.model_name)
    logger = Logger('./logs/' + args.model_name)
    
    
    model = Model(dataset.feature_size, dataset.num_class, args, dataset.labels101to20).to(device)
    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))
    print(model)

    best_acc = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)


    for itr in range(args.max_iter):
        
        train(itr, dataset, args, model, optimizer, logger, device)
          
        if itr % args.test_iter == 0 and itr>0:
            if itr == args.test_iter:
                utils.write_summary(args.dataset_name + args.model_name, args.summary)
            acc = test(itr, dataset, args, model, logger, device)
            print(args.summary)
            if acc > best_acc:
              torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
              best_acc = acc


    print('Done')
