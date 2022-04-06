import os
import sys
import copy
import math
import wandb

import numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import SegDataset, create_loader
from utils.evaluation import eval_net
from utils.loss import muti_ce_loss, multi_mse_loss
from utils.utils import *


# config_path = sys.argv[1]
config_path = sys.argv[1]
opt = load_yaml(config_path)
device = opt['device']

setup_seed(opt['seed'])
train_logger, val_logger = get_logger(opt['path']['logs'])

for phase, dataset_opt in opt['datasets'].items():
    if phase == 'train':
        train_set = SegDataset(dataset_opt, opt['dataset'], opt['task'], dim=opt['hog_decoder']['out_dim'], is_train=True)
        train_size = int(
            math.ceil(len(train_set) / dataset_opt['batch_size']))
        train_logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
            len(train_set), train_size))
        total_iters = int(opt['train']['niters'])
        total_epochs = int(math.ceil(total_iters / train_size))
        train_logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
            total_epochs, total_iters))
        train_loader = create_loader(
            train_set, opt=dataset_opt, is_train=True)
    elif phase == 'valid':
        val_set = SegDataset(dataset_opt, opt['dataset'], opt['task'], is_train=False)
        val_loader = create_loader(
            val_set, opt=dataset_opt, is_train=False)
        train_logger.info('Number of validation images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                            len(val_set)))
    else:
        raise NotImplementedError(
            'Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None



wandb.login()
wandb.init(project=opt['wandb']['project'], entity=opt['wandb']['entity'], name=opt['wandb']['name'])


seg_net, hog_decoder = define_networks(opt)


optim_seg  = optim.Adam(seg_net.parameters(), lr=opt['train']['lr_seg_net'], betas=(opt['train']['b1_seg_net'], opt['train']['b2_seg_net']), eps=1e-08, weight_decay=1e-7)
sch_seg= optim.lr_scheduler.MultiStepLR(optim_seg, opt['train']['lr_steps'], opt['train']['lr_gamma'])

if hog_decoder is not None:
    optim_hog  = optim.Adam(hog_decoder.parameters(), lr=opt['train']['lr_hog_dec'], betas=(opt['train']['b1_hog_dec'], opt['train']['b2_hog_dec']), eps=1e-08, weight_decay=1e-7)
    sch_hog = optim.lr_scheduler.MultiStepLR(optim_hog, opt['train']['lr_steps'], opt['train']['lr_gamma'])

start_epoch = opt['train']['start_epoch']
ite_num = train_size * start_epoch

best_miou = 0
best_ite = 0

best_seg_net = None
best_hog_decoder = None

train_logger.info('Start training from epoch: {:d}, iter: {:d}'.format(
        start_epoch, ite_num))

log_dict = OrderedDict()

ckpt_dir = opt['path']['checkpoints']['models']
mkdirs(ckpt_dir)

for epoch in range(start_epoch, total_epochs):
    for _, train_data in enumerate(train_loader):

        ite_num = ite_num + 1

        inputs, labels, hog_true = train_data['image'].to(device), train_data['mask'].to(device), train_data['hog_f'].to(device)
        optim_seg.zero_grad()
        optim_hog.zero_grad()

        d0, d1, d2, d3, d4, d5, d6, hx1d, hx2d, hx3d, hx4d, hx5d, hx6 = seg_net(inputs)

        loss_ce = muti_ce_loss(d0, d1, d2, d3, d4, d5, d6, labels)

        loss_total = loss_ce

        if hog_decoder is not None:

            hog1, hog2, hog3, hog4, hog5, hog6 = hog_decoder(hx1d, hx2d, hx3d, hx4d, hx5d, hx6)
            loss_mse = multi_mse_loss(hog1, hog2, hog3, hog4, hog5, hog6, hog_true)

            loss_total = loss_ce + loss_mse
        

        loss_total.backward()

        optim_seg.step()
        sch_seg.step()

        if hog_decoder is not None:
            optim_hog.step()
            sch_hog.step()

        log_dict['CE_loss'] = loss_ce.item()
        log_dict['MSE_loss'] = loss_mse.item()
        log_dict['LR_seg_net'] = sch_seg.get_last_lr()[0]
        log_dict['LR_hog_net'] = sch_hog.get_last_lr()[0]
        
        wandb.log({"Train/CE_loss":loss_ce.item()}, step=ite_num)
        wandb.log({"Train/MSE_loss":loss_mse.item()}, step=ite_num)
        wandb.log({"Train/LR_seg_net":sch_seg.get_last_lr()[0]}, step=ite_num)
        wandb.log({"Train/LR_hog_net":sch_hog.get_last_lr()[0]}, step=ite_num)

        if ite_num % opt['train']['print_freq'] == 0:
            if hog_decoder is not None:
                message = '<epoch:{:3d}, iter:{:8,d}, lr_seg:{:.3e}, lr_hog:{:.3e}> '.format(
                epoch, ite_num, sch_seg.get_last_lr()[0], sch_hog.get_last_lr()[0])
            else:
                message = '<epoch:{:3d}, iter:{:8,d}, lr_seg:{:.3e}> '.format(
                epoch, ite_num, sch_seg.get_last_lr()[0])
            for k, v in log_dict.items():
                message += '{:s}: {:.4e} '.format(k, v)
            train_logger.info(message)

        if ite_num % opt['train']['save_step'] == 0:
            torch.save(seg_net.state_dict(), os.path.join(ckpt_dir, f'seg_net_{ite_num}.pth'))
            if hog_decoder is not None:
                torch.save(hog_decoder.state_dict(), os.path.join(ckpt_dir, f'hog_dec_{ite_num}.pth'))

            train_logger.info('Saved Checkpoints.')


        if ite_num % opt['train']['val_freq'] == 0:

            ce_val_loss, miou = eval_net(opt, seg_net, val_loader, device, ite_num)

            if miou.item() > best_miou:
                best_miou = miou.item()
                best_ite = ite_num
                best_seg = copy.deepcopy(seg_net)
                best_hog = copy.deepcopy(hog_decoder)

            wandb.log({"Valid/CE_loss": ce_val_loss, 
                        "Metric/mIOU": miou.item()}, 
                        step=ite_num)
            val_logger.info('<epoch:{:3d}, iter:{:8,d}> mIOU: {:.4e} Val_loss_ce: {:.4e}'.format(
                    epoch, ite_num, miou.item(), ce_val_loss))
            train_logger.info(f'Iteration: {ite_num}\t mIOU: {miou.item()}\t Val_loss_ce: {ce_val_loss}')

        if total_iters <= ite_num:
            torch.save(seg_net.state_dict(), os.path.join(ckpt_dir, f'best_seg_net_{ite_num}.pth'))
            if hog_decoder is not None:
                torch.save(hog_decoder.state_dict(), os.path.join(ckpt_dir, f'best_hog_dec_{ite_num}.pth'))
            break
  
    if total_iters <= ite_num:
        break
