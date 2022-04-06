
import os
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from utils.utils import *




def create_loader(dataset, opt, is_train):
    if is_train == True:
        return DataLoader(
            dataset,
            batch_size=opt['batch_size'],
            shuffle=opt['shuffle'],
            num_workers=opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)


        
def get_image_paths(img_path, image_ext):

    assert os.path.isdir(img_path), f'{img_path} is not a valid directory'
    
    image_path_list = list(img_path.glob(f'*.{image_ext}'))
                
    assert image_path_list, '{:s} has no valid image file'.format(img_path)
    
    return image_path_list

class SegDataset(Dataset):

    def __init__(self, opt, dataset, task, dim=None, is_train=True):

        self.opt = opt
        self.dataset = dataset
        self.task = task
        self.hog_dim = dim
        self.is_train = is_train
        self.label_suffix = self.opt["label_suffix"] if self.opt["label_suffix"] is not None else ''
        
        self.img_path = Path(self.opt['images'])
        self.lbl_path = Path(self.opt['labels'])

        self.imgs_list = get_image_paths(self.img_path, self.opt['image_ext'])

    def __len__(self):
        return len(self.imgs_list)

    
    def __getitem__(self, idx):

        image_file = self.imgs_list[idx]
        mask_file = Path(self.opt['labels']) / f'{image_file.stem}{self.label_suffix}.{self.opt["label_ext"]}'

        mask = Image.open(mask_file)
        img = Image.open(image_file)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        if self.is_train == True:
            img, mask  = transform_data(img, mask, augment=True)

        image = transforms.ToTensor()(img)        
        mask = np.array(mask)
        if self.dataset == 'Cadis':
            mask = get_mask(self.task, mask)
        mask = torch.from_numpy(mask)

        data = {
            'image': image.type(torch.FloatTensor),
            'mask': mask.type(torch.long),
            'idx' : image_file.stem
        }
        
        if self.hog_dim is not None:
            img_copy = img.copy()
            hog_f = get_hog_f(img_copy, self.hog_dim)
            data['hog_f'] = hog_f.type(torch.long)

        return data
        
