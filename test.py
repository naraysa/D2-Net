import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Model
from video_dataset import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
from detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
import time
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def test(itr, dataset, args, model, logger, device):
    
    done = False
    tcam_stack = []
    labels_stack = []
    cos = torch.nn.CosineSimilarity(dim=1).to(device)

    while not done:

        if dataset.currenttestidx % 100 == 0:
            print('Testing test data point %d of %d' %(dataset.currenttestidx, len(dataset.testidx)))        
        features, labels, done = dataset.load_data(is_training=False)
        seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
        features = torch.from_numpy(features).float().to(device)
        
        with torch.no_grad():
            feat_f, logits_f, feat_r, logits_r = model(Variable(features), device, is_training=False)
            logits_f, logits_r = logits_f[0], logits_r[0]
            feat_r, feat_f = feat_r[0], feat_f[0]

        tcam = (torch.sigmoid(logits_f) + torch.sigmoid(logits_r))/2
        topk = int(np.ceil(len(features[0])/8))
        labels_predcam = torch.mean(torch.topk(tcam, k=topk, dim=0)[0], dim=0).cpu().data
        _, lidx = torch.topk(labels_predcam,k=2)
        if args.activity_net:
            _, negidx = torch.topk(-labels_predcam,k=args.num_class-2)
            tcam[:,negidx].fill_(0)

        # reweight pred
        atn = torch.max(tcam,dim=1)[0] 
        _, bgid = torch.topk(1-atn, k=int(1),dim=0)
        feat = (feat_f+feat_r)/2
        bgfeat = model.running_bg.data.clone() 
        atn_score = (1-cos(feat,bgfeat))/2        
        atn_score = atn_score*2/atn_score.max() if args.activity_net else atn_score
        tcam = tcam*atn_score.unsqueeze(1)
        tcam = tcam.cpu().data.numpy() 
        
        tcam_stack.append(tcam)
        labels_stack.append(labels)

    labels_stack = np.array(labels_stack)
    
    dmap, iou = dmAP(tcam_stack, dataset.path_to_annotations, args.activity_net, valid_id=dataset.lst_valid)
    for k in range(len(dmap)):
        print('Detection map @ %f = %f' %(iou[k], dmap[k]))  
    print('Mean Detection map = %f' %(np.mean(dmap)))

    dmap += [np.mean(dmap)]    
    for item in list(zip(dmap,iou)):
        logger.log_value('Test Detection mAP @ IoU = ' + str(item[1]), item[0], itr)

    utils.write_to_file(args.dataset_name + args.model_name, dmap, itr)
    
    if args.activity_net:
        return dmap.pop()
    else:
        return dmap[4]




