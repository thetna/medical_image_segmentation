import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from utils.metrics import IoU
from utils.utils import *
from utils.losses import *

def eval_net(cfg, u2net, eval_loader, current_step):

    device = cfg["device"]    
    res_save_dir = cfg["path"]["checkpoints"]["val_result_dir"]
    
    u2net.eval()
    

    
#     if cfg["with_HOG"]:
#         models['hog_reg'].eval()

    n_val = len(eval_loader)
    loss_ce = 0
#     loss_hog = 0

    metric_fold = IoU(4)
    metric_bg = IoU(2)
    metric_vessel = IoU(2)
    metric_tool = IoU(2)
    metric_fetus = IoU(2)

    for i, batch in enumerate(eval_loader):
        imgs, true_masks, idx, hog_true = batch['image'], batch['mask'], batch['idx'][0],batch['hog_f'].to(device)

        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)

        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6, hx1d, hx2d, hx3d, hx4d, hx5d, hx6 = u2net(imgs)
            
            l_ce = multi_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, true_masks)
            
#             if cfg["with_HOG"]:
#                 hog1, hog2, hog3, hog4, hog5, hog6 = models['hog_reg'](hx1d, hx2d, hx3d, hx4d, hx5d, hx6)

#                 l_hog = multi_hog_loss_fusion([hog1, hog2, hog3, hog4, hog5, hog6], hog_true)

#                 loss_hog += l_hog

        
        
        loss_ce += l_ce
        pred = torch.argmax(d0, 1)

        metric_fold.add(pred, true_masks)
        metric_bg.add((pred==0).float(), (true_masks==0).float())
        metric_vessel.add((pred==1).float(), (true_masks==1).float())
        metric_tool.add((pred==2).float(), (true_masks==2).float())
        metric_fetus.add((pred==3).float(), (true_masks==3).float())

        if cfg["path"]["checkpoints"]["save_val_results"]:
            
            img_path = os.path.join(res_save_dir, idx)        
            makedirs(img_path)
            pred_img = tensor2img(pred)

            temp_inp = imgs.squeeze().float().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255
            gt_mask = tensor2img(true_masks)
            border = np.ones((gt_mask.shape[0], 5, 3))*255
            imgs_comb = np.hstack((temp_inp, border.astype(np.uint8),gt_mask, border.astype(np.uint8), pred_img))
            cv2.imwrite(os.path.join(img_path, f'{idx}__{current_step}.png'), imgs_comb)

    _, fold_iou = metric_fold.value()
    bg_iou, bg_iou_mean = metric_bg.value()
    v_iou, v_iou_mean = metric_vessel.value()
    t_iou, t_iou_mean = metric_tool.value()
    f_iou, f_iou_mean = metric_fetus.value()

    u2net.train()
    
#     if cfg["with_HOG"]:
#         models['hog_reg'].train()
        
    return loss_ce/n_val, fold_iou, bg_iou, bg_iou_mean, v_iou, v_iou_mean, t_iou, t_iou_mean, f_iou, f_iou_mean
