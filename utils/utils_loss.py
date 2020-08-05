import torch 
import torch.nn.functional as F
import numpy as np

def partial_loss(output1, target, true):
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * output
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

    new_target = revisedY


    return loss, new_target
