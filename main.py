import sys
import logging
import numpy as np

import wandb
import torch.optim as optim

from models.u2net import U2NET
from utils.dataset import get_dataloader
from utils.evaluate import eval_net
from utils.losses import multi_ce_loss_fusion
from utils.utils import *



wandb.login()

set_random_seed(0)

#config file
config_path = sys.argv[1]
cfg = load_yaml(config_path)


max_miou = 0
max_mbg = 0
max_mve = 0
max_mto = 0
max_mfe = 0
max_ite = 0


wandb_cfg = cfg['wandb']

wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], name=wandb_cfg['name'])

makedirs(cfg['path']['log'])

setup_logger(None, cfg['path']['log'], 'train', level=logging.INFO, screen=True)
setup_logger('val', cfg['path']['log'], 'val', level=logging.INFO)

logger = logging.getLogger('base')
logger_val = logging.getLogger('val')

logger_val.info('Name: {}\n'.format(cfg["wandb"]["name"]))
logger_val.info('Fold: {} \t lr_u2net: {} \t batch_size: {}\n'.format(cfg["datasets"]["fold"], cfg["train"]["lr_u2net"], cfg["datasets"]["train"]["batch_size"]))
#dataset

train_loader, val_loader= get_dataloader(cfg['datasets'])

u2net = U2NET(cfg['U2Net']['in_ch'], cfg['U2Net']['out_ch'])
u2net.to(cfg['device'])
u2net.train()

opt_u2net  = optim.Adam(u2net.parameters(), lr=cfg['train']['lr_u2net'], betas=(cfg['train']['b1'], cfg['train']['b2']), eps=float(cfg['train']['eps']), weight_decay=float(cfg['train']['weight_decay']))

scheduler_u2net = optim.lr_scheduler.MultiStepLR(opt_u2net, cfg['train']['lr_steps'], cfg['train']['lr_gamma'])


if cfg['with_HOG']:
    from models.hog_reg import HOG_Regressor
    from utils.losses import multi_hog_loss_fusion
    
    hog_reg = HOG_Regressor(cfg['Hog_Regressor']['out_dim'])
    hog_reg.to(cfg['device'])
    hog_reg.train()

    logger_val('lr_hog_reg: {} \t hog_dim: {}\t hog_bins: {}\t ppc: {}\n'.format(cfg["train"]["lr_hog"], cfg["Hog_Regressor"]["out_dim"], cfg["datasets"]["train"]["hog_bins"], cfg["datasets"]["train"]["pix_per_cell"]))
    
    
    opt_hog_reg  = optim.Adam(hog_reg.parameters(), lr=cfg['train']['lr_hog'], betas=(cfg['train']['b1'], cfg['train']['b2']), eps=float(cfg['train']['eps']), weight_decay=float(cfg['train']['weight_decay']))

    scheduler_hog_reg = optim.lr_scheduler.MultiStepLR(opt_hog_reg, cfg['train']['lr_steps'], cfg['train']['lr_gamma'])

current_step = 0

while True:

    for i, data in enumerate(train_loader):

        #validation at 0 iter
        
        if current_step % cfg['train']['val_step'] == 0:            
            ce_loss_val, fold_iou, bg_iou, bg_iou_mean, v_iou, v_iou_mean, t_iou, t_iou_mean, f_iou, f_iou_mean = eval_net(cfg, u2net, val_loader, current_step)

            if fold_iou.item() > max_miou:
                max_miou = fold_iou.item()
                max_mbg = bg_iou_mean.item()
                max_mve = v_iou_mean.item()
                max_mto = t_iou_mean.item()
                max_mfe = f_iou_mean.item()
                max_ite = current_step

                best_model = {'u2net': u2net}
                if cfg["with_HOG"]:
                    best_model['hog_reg'] = hog_reg


            wandb.log(
                {
                "Valid/CE_loss": ce_loss_val, 
                "Metric/mIOU": fold_iou.item(), 
                "Metric/IOU_BG": bg_iou_mean.item(), 
                "Metric/IOU_Vessel": v_iou_mean.item(), 
                "Metric/IOU_Tool": t_iou_mean.item(), 
                "Metric/IOU_Fetus": f_iou_mean.item(), 
                "Metric/IOU_BG0": bg_iou[0].item(), 
                "Metric/IOU_BG1": bg_iou[1].item(), 
                "Metric/IOU_Vessel0": v_iou[0].item(),
                "Metric/IOU_Vessel1": v_iou[1].item(), 
                "Metric/IOU_Tool0": t_iou[0].item(), 
                "Metric/IOU_Tool1": t_iou[1].item(), 
                "Metric/IOU_Fetus0": f_iou[0].item(), 
                "Metric/IOU_Fetus1": f_iou[1].item()
            }, 
                step=current_step
            )

            logger_val.info('Iter:{:10d}\tmIOU: {:.4f}\tIOU_BG: {:.4f}\tIOU_Vessel: {:.4f}\tIOU_Tool: {:.4f}\tIOU_Fetus: {:.4f}'.format(current_step, fold_iou.item(), bg_iou_mean.item(), v_iou_mean.item(), t_iou_mean.item(), f_iou_mean.item()))


        images, masks, hog_features = data['image'].to(cfg['device']), data['mask'].to(cfg['device']), data['hog_f'].to(cfg['device'])
        opt_u2net.zero_grad()

        d0, d1, d2, d3, d4, d5, d6, hx1d, hx2d, hx3d, hx4d, hx5d, hx6 = u2net(images)
        
        loss_ce = multi_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, masks)
        
        loss = loss_ce
        loss_hog = torch.Tensor([0])
        
        if cfg['with_HOG']:
            
            opt_hog_reg.zero_grad()
            
                        
            hog1, hog2, hog3, hog4, hog5, hog6 = hog_reg(hx1d, hx2d, hx3d, hx4d, hx5d, hx6)

            loss_hog = multi_hog_loss_fusion([hog1, hog2, hog3, hog4, hog5, hog6], hog_features)

            loss = cfg['train']['wt_ce'] * loss_ce + cfg['train']['wt_hog'] * loss_hog
                

#         print(f'Taining: {u2net.training}')
        loss.backward()
        
        opt_u2net.step()
        scheduler_u2net.step()
        
        wandb.log({"Train/Total_loss":loss.item()}, step=current_step)
        wandb.log({"Train/CE_loss":loss_ce.item()}, step=current_step)
        
        if cfg['with_HOG']:
            opt_hog_reg.step()  
            scheduler_hog_reg.step()         
            wandb.log({"Train/Hog_loss":loss_hog.item()}, step=current_step)
        
        
        if current_step % cfg['train']['print_freq'] == 0:
            logger.info('Iter:{:7d}\tCE_loss: {:.4f}\tHog_loss: {:.4f}\tTotal_loss: {:.4f}'.format(current_step, loss_ce.item(), loss_hog.item(), loss.item()))


        if current_step % cfg['train']['save_step'] == 0:
            model_dict = {'u2net': u2net}
            if cfg['with_HOG']:
                model_dict['hog_reg'] = hog_reg
            
            save_model(cfg, model_dict, current_step)
            logger.info('Saved checkpoints.')
                    
            
        current_step = current_step + 1    

        if cfg["train"]["niters"] < current_step:
            np.array([max_ite, max_mbg, max_mve, max_mto, max_mfe, max_miou]).tofile(results, sep="\t")
            results.close()
            logger_val.info('\nBest Result\n')
            logger_val.info('Iter:{:10d}\tmIOU: {:.4f}\tIOU_BG: {:.4f}\tIOU_Vessel: {:.4f}\tIOU_Tool: {:.4f}\tIOU_Fetus: {:.4f}'.format(max_ite, max, max_mbg, max_mve, max_mto, max_mfe))
            save_model(cfg, best_model, max_ite)
            break


    if cfg["train"]["niters"] < current_step:
        break
wandb.finish()