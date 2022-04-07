

import os
import sys
import numpy as np

from tqdm import tqdm
import PIL.Image as Image
from pathlib import Path

import torch
import torchvision.transforms.functional as TF


from models import U2Net

from utils.cadis_vis import *
from utils.utils import *


# config_path = sys.argv[1]
config_path = sys.argv[1]
opt = load_yaml(config_path)

device = opt['device']
dataset = opt['dataset']
task = opt['task']

inp_path = Path(opt['path']['inp_path'])
pred_save_path = Path(opt['path']['pred_save_path'])

gt_path = Path(opt['path']['gt_path'])
gt_save_path = Path(opt['path']['gt_save_path'])

model_path = opt['path']['weight_path']
in_nc = opt['network']['in_nc']
out_nc = opt['network']['out_nc']

image_ext = opt['path']['image_ext']

mkdirs(pred_save_path)
mkdirs(gt_save_path)

model = U2Net(in_nc, out_nc)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)
model.eval()

if gt_path is not None:
    from utils.loss import muti_ce_loss
    from utils.metric import compute_mean_iou

    label_ext = opt['path']['label_ext']
    label_suffix = opt['path']['label_suffix'] if opt['path']['label_suffix'] is not None else ''
    miou_list = []

if dataset == 'Cadis':
    colormap = get_colormap(task)


for img_path in tqdm(inp_path.glob(f'*.{image_ext}')):

    image = Image.open(img_path)

    img = TF.to_tensor(np.array(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)

    mask_pred = torch.argmax(pred[0], 1)

    if gt_path is not None:
        mask_gt = Image.open(f'{gt_path}/{img_path.stem}{label_suffix}.{label_ext}')
        
        if dataset == 'Cadis':
            mask_gt = get_mask(task, np.array(mask_gt))
            mask_gt = torch.from_numpy(mask_gt)
            mask_gt_img = Image.fromarray(mask_to_colormap(mask_gt.squeeze().float().cpu(), colormap))

        else:
            mask_gt = torch.from_numpy(np.array(mask_gt)).unsqueeze(0).to(device, dtype=torch.float32) 
            mask_gt_img = Image.fromarray(tensor2img(mask_gt))

        miou = compute_mean_iou(mask_pred.squeeze().cpu().numpy().flatten().astype(np.uint8), mask_gt.squeeze().cpu().numpy().flatten().astype(np.uint8))
        miou_list.append(miou) 

        mask_gt_img.save(f'{gt_save_path}/{img_path.stem}_GT.png')   
        
    if dataset == 'Cadis':
        mask_pred = Image.fromarray(mask_to_colormap(mask_pred.squeeze().float().cpu(), colormap))
    else:
        mask_pred = Image.fromarray(tensor2img(mask_pred))


    mask_pred.save(f'{pred_save_path}/{img_path.stem}_pred.png')
    
if gt_path is not None:
    print('mIOU: ', np.mean(miou_list))
