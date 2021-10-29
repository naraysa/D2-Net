import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Model
from video_dataset import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
from pdmi import pDMI
import time
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.autograd.set_detect_anomaly(True)
import random



def DISLOSS(element_logits, features, seq_len, batch_size, labels, device, bg_mean=0, pdmi=False, gamma=None): 

    bceloss = torch.zeros(1).to(device)
    dmiloss = torch.zeros(1).to(device)
    k = np.ceil(seq_len/4).astype('int32') if element_logits.size(2) == 100 else np.ceil(seq_len/8).astype('int32') 

    empty = torch.zeros(0).to(device)
    feat_fg, feat_bg, feat_bg2 = empty.clone(), empty.clone(), empty.clone()
    lab, instance_logits = empty.clone(), empty.clone()
    cosfocusfgwt, maskwt_fg, maskwt_bg = empty.clone(), empty.clone(), empty.clone()
    onetensor = torch.ones(1).to(device)
    identity_wt = torch.ones(1,2).to(device)
    emb_lab = torch.ones(2).to(device)
    emb_lab[1] = -1
    emb_crit = torch.nn.CosineEmbeddingLoss(margin=0,reduction='none').to(device)

    for i in range(batch_size):
        if seq_len[i] < 5 or labels[i].sum() == 0:
            continue

        labi = (torch.arange(labels.size(1))[labels[i]>0])
        tcam = element_logits[i][:seq_len[i]]
        atn_fg = torch.max(tcam,dim=1)[0]
        atn_bg = 1 - atn_fg

        tmp, topki = torch.topk(tcam, k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
        lab = torch.cat([lab, labels[[i]]], dim=0)
        
        _, bgid_top = torch.topk(atn_bg, k=1, dim=0)
        atn_fg[atn_fg<0.5] = 0
        if lab.size(1) == 100: 
            atn_bg[atn_bg<0.8] = 0  ### Same as atn_bg[atn_fg>0.2] = 0
        else:
            atn_bg[atn_bg<0.5] = 0
        
        # FG and BG embeddings
        if atn_fg.sum() > 0:
            maskwt_fg = torch.cat([maskwt_fg, onetensor],dim=0)
            feati = torch.sum(features[i][:seq_len[i]]*atn_fg.unsqueeze(1),dim=0,keepdim=True)/atn_fg.sum()
            feat_fg = torch.cat([feat_fg, feati], dim=0)
        else:
            maskwt_fg = torch.cat([maskwt_fg, 0*onetensor],dim=0)
        if atn_bg.sum() > 0:
            maskwt_bg = torch.cat([maskwt_bg, onetensor],dim=0)
            feati = torch.sum(features[i][:seq_len[i]]*atn_bg.unsqueeze(1),dim=0,keepdim=True)/atn_bg.sum()
            feat_bg = torch.cat([feat_bg, feati], dim=0)
            # for running mean of bg    
            feati = torch.sum(features[i][:seq_len[i]][bgid_top],dim=0,keepdim=True)
            feat_bg2 = torch.cat([feat_bg2, feati], dim=0)
        else:
            maskwt_bg = torch.cat([maskwt_bg, 0*onetensor],dim=0)


    num_fg, num_bg = feat_fg.size(0), feat_bg.size(0)
    num_fgbg = min(num_fg, num_bg)
    wt = torch.zeros(maskwt_fg.numel(),3)

    # grouping and separation wts
    if num_fg > 0:
        fgidx = torch.arange(maskwt_fg.numel())[maskwt_fg==1]
        randidx = torch.randperm(num_fg)
        fgwt = emb_crit(feat_fg, feat_fg[randidx], torch.ones(num_fg).to(device))
        wt[fgidx,0] = fgwt
    if num_bg > 0:
        bgidx = torch.arange(maskwt_bg.numel())[maskwt_bg==1]
        randidx = torch.randperm(num_bg)
        bgwt = emb_crit(feat_bg, feat_bg[randidx], torch.ones(num_bg).to(device))
        wt[bgidx,1] = bgwt
    if num_fgbg > 0:
        fg_idx = torch.randperm(num_fg)[:num_fgbg]
        bg_idx = torch.randperm(num_bg)[:num_fgbg]
        bgfgwt = emb_crit(feat_fg[fg_idx], feat_bg[bg_idx], -1.0*torch.ones(num_fgbg).to(device))
        wt[fg_idx,2] = bgfgwt

    # Running mean of BG embedding
    if feat_bg2.size(0) > 0:
        batch_bg = torch.mean(feat_bg2.data,dim=0,keepdim=True)
        bg_mean = 0.9*bg_mean + 0.1*batch_bg
    
    if lab.numel() > 0:
        instance_logits = torch.clamp(instance_logits, min=1e-3, max=1-(1e-3))
        # Focus with cosine distance wt
        if gamma is None:
            gamma = 1.0/instance_logits.size(1)

        ### Add grouping/separation wts into focal penalty
        bceloss =  -1.0*((1-instance_logits + 1*wt[:,[2]] + gamma*wt[:,[0]]).pow(2)*lab*torch.log(instance_logits) + (instance_logits + 1*wt[:,[2]] + gamma*wt[:,[1]]).pow(2)*(1-lab)*torch.log(1-instance_logits))        
        # fg-bg balance
        pos_wt = torch.ones(lab.size(1)).fill_(element_logits.size(2))
        pos_wt_new = pos_wt.unsqueeze(1).permute([1,0]) * torch.ones(lab.size()) * lab
        pos_wt_new[pos_wt_new<1] = 1
        bceloss = (bceloss * pos_wt_new).mean()

        ### pDMI for video
        if pdmi:
            instance_logits2 = empty.clone()
            for bs in range(instance_logits.size(0)):
                tmpmask = instance_logits[bs] > 0.2*instance_logits[bs].max()
                tmplogits = instance_logits[bs] * tmpmask.float()
                tmplogits = tmplogits.unsqueeze(0)/tmplogits.sum()
                instance_logits2 = torch.cat([instance_logits2, tmplogits], dim=0)
            dmiloss = svddmi(instance_logits2, lab.clone(), device)
            if torch.isnan(dmiloss).sum() or torch.isinf(dmiloss).sum(): 
                dmiloss = torch.zeros(1).to(device)
            
    return bceloss, dmiloss, bg_mean





def pDMILOSS(element_logits, features, seq_len, batch_size, labels, device, bg_mean=0):

    pdmiloss = torch.zeros(1).to(device)
    if bg_mean.abs().sum() == 0:
        return dmiloss
    num = 0
    k = np.ceil(seq_len/8).astype('int32') 
    
    empty = torch.zeros(0).to(device)
    lab, instance_logits = empty.clone, empty.clone()

    y_lab, x_feat = empty.clone(), empty.clone()
    
    for i in range(batch_size):
        if seq_len[i] < 5 or labels[i].sum() == 0:
            continue
        labi = (torch.arange(labels.size(1))[labels[i]>0])
        atn = element_logits[i][:seq_len[i]]

        # Bottom-up attention for computing FG and BG indices
        atn_score = ((1-cosine_sim(features[i][:seq_len[i]],bg_mean.detach()))/2)

        # top-down attention for fg-bg confidences
        atnk, atnk_id = atn[:,labi].max(1)
        atnk_all = atn.max(1)[0]

        # fg and bg indices using bottom-up attention
        fgid = torch.arange(atn_score.numel())[atn_score>0.5]
        if atn.size(1) == 100:
            bgid = torch.arange(atn_score.numel())[atn_score<0.3]
        else:
            bgid = torch.arange(atn_score.numel())[atn_score<0.5]
            
        if fgid.numel() == 0 or bgid.numel()==0:
            continue
        
        y_lab = torch.cat([torch.zeros(fgid.numel()), torch.ones(bgid.numel())], dim=0)
        x_feat = torch.cat([atnk[fgid], atnk_all[bgid]], dim=0)
        if y_lab.sum() > 0 and y_lab.sum() < y_lab.numel():
            ### Using pDMI
            x_feat = x_feat.unsqueeze(1)
            x_feat = torch.cat([x_feat,1-x_feat], dim=1)
            pdmilossi = pDMI(x_feat, y_lab) 
            pdmiloss += pdmilossi
        
        num += 1
        
    if torch.isnan(pdmiloss).sum() or torch.isinf(pdmiloss).sum(): 
        pdmiloss = torch.zeros(1).to(device)

    pdmiloss /= max(1,num)

    return pdmiloss


def svddmi(x,y,device):
    if len(y.size()) == 2:
        Y_all = y
    else:
        Y_all = torch.zeros(x.size()).to(device).scatter_(1, y.unsqueeze(1).long(), 1)
    Y_all = Y_all.transpose(0,1)
    Y_all /= Y_all.sum(1).unsqueeze(1).clamp(min=1)
    joint = Y_all @ x
    _, s1, _ = joint.svd()
    s2 = s1[s1>1e-5]
    dmiloss_c = (s2.max()/s2.min()).log()
    return dmiloss_c


sigmoid = torch.nn.Sigmoid().cuda()
softmax = torch.nn.Softmax(dim=1).cuda()
cosine_sim = torch.nn.CosineSimilarity(dim=1).cuda()


def train(itr, dataset, args, model, optimizer, logger, device):
    
    total_loss, LDS = Variable(torch.zeros(1).to(device)), Variable(torch.zeros(1).to(device))
    LDV = Variable(torch.zeros(1).to(device))

    features, labels = dataset.load_data()
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    
    feat_f, logits_f, feat_r, logits_r = model(Variable(features), device)
    tcam = (sigmoid(logits_f) + sigmoid(logits_r))/2
    tfeat = (feat_f+feat_r)/2
    
    pdmi_cond = itr > args.pdmi_iter and itr % 2 == 0

    LDis, LDV, bg_mean = DISLOSS(tcam, tfeat, seq_len, args.batch_size, labels, device, bg_mean=model.running_bg.data, pdmi=pdmi_cond, gamma=args.grouping_wt)
    if bg_mean is not None:
        model.running_bg.data = bg_mean.clone()

    if pdmi_cond:
        LDS = pDMILOSS(tcam, tfeat, seq_len, args.batch_size, labels, device, bg_mean=model.running_bg.data)
        
    total_loss =  LDis + args.lds_wt*LDS + args.ldv_wt*LDV

    logger.log_value('total_loss', total_loss, itr)
    try:
        print('Iteration: %d, Loss: %.3f, LDis: %.3f, LDS: %.3f, LDV: %.3f' %(itr, total_loss.data.cpu().numpy(), 
            LDis.data.cpu().numpy(), LDS.data.cpu().numpy(), LDV.data.cpu().numpy() ))
    except:
        print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))
    optimizer.zero_grad()
    if total_loss > 0 and not torch.isnan(total_loss):
        total_loss.backward()
    else:
        return
    if total_loss > 0:
        optimizer.step()




