import torch

def pDMI(outputs, target):
    """
    outputs is N x C size, C=number of classes and N=batch size
    target=N size with labels in {0..C-1}
    """
    num_classes = outputs.size(1)
    targets = target.reshape(target.size(0), 1)
    y_onehot = torch.FloatTensor(target.size(0), num_classes).zero_().cuda()
    y_onehot.scatter_(1, targets.long(), 1)
    y_onehot = y_onehot.transpose(0, 1)
    if (y_onehot.sum(1) == 0).sum() > 0:
        return torch.zeros(1).cuda()
    
    y_onehot /= y_onehot.sum(1).unsqueeze(1).clamp(min=1)
    mat = y_onehot @ outputs 
    mat = mat.clamp(min=1e-10) 
    _,s1,_=mat.svd()
    pdmiloss = (s1.max()/s1.min()).log() 

    return pdmiloss

    



