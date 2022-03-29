
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *
from utils.loss import muti_ce_loss
from utils.metric import compute_mean_iou

def eval_net(opt, net, loader, device, current_step):

    net.eval()
    n_val = len(loader)  # the number of batch
    loss_ce = 0
    miou_list = []
    save_path = opt['path']['checkpoints']['val_results']
    print_freq = opt['train']['print_freq']
    dataset = opt['dataset']
    task = opt['task']


    for batch in tqdm(loader):
        imgs, true_masks, idx = batch['image'], batch['mask'], batch['idx']

        imgs = imgs.to(device=device)
        true_masks = true_masks.to(device=device)

        save_path_img = os.path.join(save_path, idx[0])
        mkdirs(save_path_img)

        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6, hx1d, hx2d, hx3d, hx4d, hx5d, hx6 = net(imgs)

        l_ce = muti_ce_loss(d0, d1, d2, d3, d4, d5, d6, true_masks)
        loss_ce += l_ce

        pred = torch.argmax(d0, 1)

        miou = compute_mean_iou(pred.squeeze().cpu().numpy().flatten().astype(np.uint8), true_masks.squeeze().cpu().numpy().flatten().astype(np.uint8))
        miou_list.append(miou)

        save_validation(pred, imgs, true_masks, idx, current_step, save_path_img, dataset, task)
        
    net.train()
    return loss_ce/n_val, np.mean(miou_list) 
