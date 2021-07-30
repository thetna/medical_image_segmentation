import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image

# from jaccard_loss import JaccardLoss
# from dice_loss import dice_coeff

from metrics import IoU

from tqdm import tqdm


# class_weights = torch.Tensor([0.1150, 0.9106, 0.9887, 0.9857]).to(device)
# weight=class_weights
crossentropy_loss = nn.CrossEntropyLoss()

def muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
	loss0 = crossentropy_loss(d0,labels_v.type(dtype=torch.long))
	loss1 = crossentropy_loss(d1,labels_v.type(dtype=torch.long))
	loss2 = crossentropy_loss(d2,labels_v.type(dtype=torch.long))
	loss3 = crossentropy_loss(d3,labels_v.type(dtype=torch.long))
	loss4 = crossentropy_loss(d4,labels_v.type(dtype=torch.long))
	loss5 = crossentropy_loss(d5,labels_v.type(dtype=torch.long))
	loss6 = crossentropy_loss(d6,labels_v.type(dtype=torch.long))

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

	return loss0, loss


mse = nn.MSELoss()
def muti_hog_loss_fusion(hog1, hog2, hog3, hog4, hog5, hog6, gt):
	loss0 = mse(hog1.float(),gt.float())
	loss1 = mse(hog2.float(),gt.float())
	loss2 = mse(hog3.float(),gt.float())
	loss3 = mse(hog4.float(),gt.float())
	loss4 = mse(hog5.float(),gt.float())
	loss5 = mse(hog6.float(),gt.float())
	return loss0 + loss1 + loss2 + loss3 + loss4 + loss5

#metric for iou

metric_fold = IoU(4)

metric_bg = IoU(2)
metric_vessel = IoU(2)
metric_tool = IoU(2)
metric_fetus = IoU(2)




def get_colormap():
    """
    Returns FetReg colormap
    """
    colormap = np.asarray(
        [
            [0, 0, 0],   # 0 - background
            [0, 0, 255], # 1 - vessel
            [255, 0, 0], # 2 - tool
            [0, 255, 0], # 3 - fetus

        ]
        )
    return colormap

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):

    colormap = get_colormap()
    n_dim = tensor.dim()

    tensor = tensor.squeeze().float().cpu()

    mask_rgb = np.zeros(tensor.shape[:2] + (3,), dtype=np.uint8)
    for cnt in range(len(colormap)):
        mask_rgb[tensor == cnt] = colormap[cnt]

    return mask_rgb.astype(out_type)

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval_net(net, hog_decoder, loader, device, current_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    hog_decoder.eval()
#     mask_type = torch.float32 if net.n_classes == 1 else torch.long
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    loss_ce = 0
    loss_hog = 0
    metric_fold.reset()
    metric_bg.reset()
    metric_vessel.reset()
    metric_tool.reset()
    metric_fetus.reset()

    for batch in tqdm(loader):
        imgs, true_masks, idx, hog_true = batch['image'], batch['mask'], batch['idx'][0], batch['hog_f'].to(device)

        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

#         print('------------------',idx[0])



        img_path = os.path.join('val_results', idx)
        mkdirs(img_path)

        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6, hx1d, hx2d, hx3d, hx4d, hx5d, hx6 = net(imgs)
            hog1, hog2, hog3, hog4, hog5, hog6 = hog_decoder(hx1d, hx2d, hx3d, hx4d, hx5d, hx6)



#             print(d1.requires_grad)

        _, l_ce = muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, true_masks)
        l_hog = muti_hog_loss_fusion(hog1, hog2, hog3, hog4, hog5, hog6, hog_true)
        loss_ce += l_ce
        loss_hog += l_hog

        pred = torch.argmax(d1, 1)
        metric_fold.add(pred, true_masks)


        metric_bg.add((pred==0).float(), (true_masks==0).float())
        metric_vessel.add((pred==1).float(), (true_masks==1).float())
        metric_tool.add((pred==2).float(), (true_masks==2).float())
        metric_fetus.add((pred==3).float(), (true_masks==3).float())

        pred_img = tensor2img(pred)

        temp_inp = imgs.squeeze().float().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255
#         print(temp_inp.shape, type(temp_inp))
#         inp_img = Image.fromarray(temp_inp)

        gt_mask = tensor2img(true_masks)
        border = np.ones((gt_mask.shape[0], 5, 3))*255
        imgs_comb = np.hstack((temp_inp, border.astype(np.uint8),gt_mask, border.astype(np.uint8), pred_img))
#         gt_mask = tensor2img(true_masks)
#         imgs_comb = np.hstack((pred_img, gt_mask))
        cv2.imwrite(os.path.join(img_path, f'{idx}__{current_step}.png'), imgs_comb)

#         gt_mask = tensor2img(true_masks)
#         imgs_comb = np.hstack((pred_img, gt_mask))
#         cv2.imwrite(os.path.join(img_path, f'{idx}__{current_step}.png'), imgs_comb)

#         tot += dice_coeff((pred==1).float(), (true_masks==1).float()).item()

#     per_class_iou, mean_iou = metric_fold.value()

#     fold_iou = mean_iou.item()
#     bg_iou = per_class_iou[0].item()
#     v_iou = per_class_iou[1].item()
#     t_iou = per_class_iou[2].item()
#     f_iou = per_class_iou[3].item()
    _, fold_iou = metric_fold.value()
    bg_iou, bg_iou_mean = metric_bg.value()
    v_iou, v_iou_mean = metric_vessel.value()
    t_iou, t_iou_mean = metric_tool.value()
    f_iou, f_iou_mean = metric_fetus.value()

    net.train()
    hog_decoder.train()
    return loss_ce/n_val, loss_hog/n_val, fold_iou, bg_iou, bg_iou_mean, v_iou, v_iou_mean, t_iou, t_iou_mean, f_iou, f_iou_mean
