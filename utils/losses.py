import torch
from torch import nn

#ce loss
crossentropy_loss = nn.CrossEntropyLoss()
# class_weights = torch.Tensor([0.1150, 0.9106, 0.9887, 0.9857]).to(device)
# weight=class_weights

def multi_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = crossentropy_loss(d0, labels_v.type(dtype=torch.long))
    loss1 = crossentropy_loss(d1, labels_v.type(dtype=torch.long))
    loss2 = crossentropy_loss(d2, labels_v.type(dtype=torch.long))
    loss3 = crossentropy_loss(d3, labels_v.type(dtype=torch.long))
    loss4 = crossentropy_loss(d4, labels_v.type(dtype=torch.long))
    loss5 = crossentropy_loss(d5, labels_v.type(dtype=torch.long))
    loss6 = crossentropy_loss(d6, labels_v.type(dtype=torch.long))

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss


#hog loss
mse = nn.MSELoss()


def multi_hog_loss_fusion(pred_list, gt):
    loss = 0.0
    for pred in pred_list:
        loss += mse(pred, gt)
    return loss