from os.path import splitext
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader


from utils.utils import *


def get_dataloader(config):
    
    data_root = config["data_root"]
    fold = int(config["fold"])
    
    train_dir_list, val_dir_list = get_dir_lists(fold)
    
    train_dataset = SegDataset(config, data_root, train_dir_list, True, True)
    val_dataset = SegDataset(config, data_root, val_dir_list, False, False)
    
    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["valid"]["batch_size"], shuffle=False, num_workers=1, pin_memory=True)
    
    return train_loader, val_loader 
    
    


class SegDataset(Dataset):
    def __init__(self, config, root_dir, dir_list, is_train, augment, mask_suffix=''):
        
        self.config = config
        self.is_train = is_train   
        self.augment = augment
        self.imgs_dir, self.masks_dir = get_image_paths(dir_list, root_dir)
        self.mask_suffix = mask_suffix


    def __len__(self):
        return len(self.imgs_dir)


    def __getitem__(self, idx):
        mask_file = self.masks_dir[idx]   
        img_file = self.imgs_dir[idx]
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        
        if self.is_train:
            img, mask   = transform(self.config, [img, mask], self.augment)
        img_copy = img.copy()
        img = TF.to_tensor(np.array(img))
        mask = torch.from_numpy(np.array(mask))
        hog_f = get_hog_f(self.config, img_copy)
        
        return {
            'image': img.type(torch.FloatTensor),
            'mask': mask.type(torch.FloatTensor),
            'hog_f':  hog_f.type(torch.FloatTensor),
            'idx' : img_file.split('/')[-1].split('.')[0]
        }



    
