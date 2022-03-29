

import torch
import torch.nn as nn


crossentropy_loss = nn.CrossEntropyLoss()

def muti_ce_loss(d0, d1, d2, d3, d4, d5, d6, labels_v):
	loss0 = crossentropy_loss(d0,labels_v.type(dtype=torch.long))
	loss1 = crossentropy_loss(d1,labels_v.type(dtype=torch.long))
	loss2 = crossentropy_loss(d2,labels_v.type(dtype=torch.long))
	loss3 = crossentropy_loss(d3,labels_v.type(dtype=torch.long))
	loss4 = crossentropy_loss(d4,labels_v.type(dtype=torch.long))
	loss5 = crossentropy_loss(d5,labels_v.type(dtype=torch.long))
	loss6 = crossentropy_loss(d6,labels_v.type(dtype=torch.long))

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

	return loss



mse = nn.MSELoss()

def multi_mse_loss(hog1, hog2, hog3, hog4, hog5, hog6, gt):
	loss0 = mse(hog1.float(),gt.float())
	loss1 = mse(hog2.float(),gt.float())
	loss2 = mse(hog3.float(),gt.float())
	loss3 = mse(hog4.float(),gt.float())
	loss4 = mse(hog5.float(),gt.float())
	loss5 = mse(hog6.float(),gt.float())
	return loss0 + loss1 + loss2 + loss3 + loss4 + loss5