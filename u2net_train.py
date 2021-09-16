import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import os


from model import U2NET, HOG_Decoder
from dataset import SegDataset
from eval import eval_net
from utils import *

#change here
device=0
set_random_seed(0)
model_dir = 'saved_models/'
batch_size_train = 16
batch_size_val = 1
train_num = 0
val_num = 0
eval_frq = 1000

save_frq = 20000 # save the model every 50000 iterations
total_iters = 200000

fold = 1
fold_name = f'f{fold}_hog_decall_bin9_8_11_git'


data_root = '/raid/binod/rr/segmentation/dataset'
train_dir_list, val_dir_list = get_dir_lists(fold=fold)



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

import wandb
wandb.login()
# ------- 1. define loss function --------

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

# hog loss
mse = nn.MSELoss()
def muti_hog_loss_fusion(hog1, hog2, hog3, hog4, hog5, hog6, gt):
	loss0 = mse(hog1.float(),gt.float())
	loss1 = mse(hog2.float(),gt.float())
	loss2 = mse(hog3.float(),gt.float())
	loss3 = mse(hog4.float(),gt.float())
	loss4 = mse(hog5.float(),gt.float())
	loss5 = mse(hog6.float(),gt.float())
	return loss0 + loss1 + loss2 + loss3 + loss4 + loss5



# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'




mkdir(model_dir)


#dataset

train_dataset = SegDataset(data_root, train_dir_list, True)
val_dataset = SegDataset(data_root, val_dir_list, False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
# eval_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
eval_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=1, pin_memory=False, drop_last=True)


net = U2NET(3, 4)

hog_decoder = HOG_Decoder()

if torch.cuda.is_available():
    net.to(device)
    hog_decoder.to(device)

    net.train()
    hog_decoder.train()


# load from checkpoints

#wandb

wandb.init(project='miccai', entity='naamii', name=fold_name)
wandb.watch([net], log='all')

# ------- 4. define optimizer --------
print("---define optimizer...")

optimizer  = optim.Adam(list(net.parameters())+ list(hog_decoder.parameters()), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [75000, 150000], 0.5)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0


results = open('results.txt', 'w')
print("Writing to a file")
results.write('Iter\t BG\t VES\t TOOL\t FETUS\t MEAN\n')
max_miou = 0
max_mbg = 0
max_mve = 0
max_mto = 0
max_mfe = 0
max_ite = 0

best_u2net = None
best_hog = None


while True:

    for i, data in enumerate(train_dataloader):
        ite_num = ite_num + 1

        inputs, labels, hog_true = data['image'].to(device), data['mask'].to(device), data['hog_f'].to(device)

        # forward + backward + optimize encoder and decoder

#         for p in regressor.parameters():
#             p.requires_grad = False

        optimizer.zero_grad()

        d0, d1, d2, d3, d4, d5, d6, hx1d, hx2d, hx3d, hx4d, hx5d, hx6 = net(inputs)

        hog1, hog2, hog3, hog4, hog5, hog6 = hog_decoder(hx1d, hx2d, hx3d, hx4d, hx5d, hx6)

        loss_hog = muti_hog_loss_fusion(hog1, hog2, hog3, hog4, hog5, hog6, hog_true)


        loss2_ce, loss_ce = muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
        loss = loss_ce
        loss = loss_ce + loss_hog

        loss.backward()
        optimizer.step()
        scheduler.step()

#         print(f'Iter: {ite_num}\t Loss CE: {loss.item()}')

        wandb.log({"Train/Total_loss":loss.item()}, step=ite_num)
        wandb.log({"Train/CE_loss":loss_ce.item()}, step=ite_num)
        wandb.log({"Train/Hog_loss":loss_hog.item()}, step=ite_num)

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, hx6, loss2_ce, loss_ce, loss_hog, loss, hog1, hog2, hog3, hog4, hog5, hog6

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name+"net%d.pth" % (ite_num))
            torch.save(hog_decoder.state_dict(), model_dir + model_name+"hog_decoder%d.pth" % (ite_num))
            print('saving_model')


        if ite_num % eval_frq == 0:
            print('Validating')
             #Validation at the end of each 5000 iters
            ce_loss_val, hog_loss_val, fold_iou, bg_iou, bg_iou_mean, v_iou, v_iou_mean, t_iou, t_iou_mean, f_iou, f_iou_mean = eval_net(net, hog_decoder, eval_dataloader, device, ite_num)
        #     scheduler.step(val_loss)

            if fold_iou.item() > max_miou:
                max_miou = fold_iou.item()
                max_mbg = bg_iou_mean.item()
                max_mve = v_iou_mean.item()
                max_mto = t_iou_mean.item()
                max_mfe = f_iou_mean.item()
                max_ite = ite_num
                best_u2net = net
                best_hog = hog_decoder


            wandb.log({"Valid/CE_loss": ce_loss_val,
                        "Valid/Hog_loss": hog_loss_val, 
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
                        "Metric/IOU_Fetus1": f_iou[1].item()}, step=ite_num)

            print(f'Ite: {ite_num}\t Val_loss_ce: {ce_loss_val}\t MeanIOU: {fold_iou.item()}')
            del ce_loss_val, hog_loss_val, fold_iou, bg_iou, v_iou, t_iou, f_iou

        if total_iters <= ite_num:
            np.array([max_ite, max_mbg, max_mve, max_mto, max_mfe, max_miou]).tofile(results, sep="\t")
            results.write('\n')
            results.close()
            torch.save(best_u2net.state_dict(), model_dir + model_name+"bestnet%d.pth" % (ite_num))
            torch.save(best_hog.state_dict(), model_dir + model_name+"besthog%d.pth" % (ite_num))
           
            break
  
    if total_iters <= ite_num:
        break
