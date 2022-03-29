
import os
import numpy as np
from PIL import Image
from pathlib import Path

from skimage.feature import hog

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


def get_hog_f(img):

    hog_f = hog(img.resize((64, 128)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=True)
    
    return torch.from_numpy(hog_f)

        
def get_image_paths(img_path, image_ext):

    assert os.path.isdir(img_path), f'{img_path} is not a valid directory'
    
    image_path_list = list(img_path.glob(f'*.{image_ext}'))
                
    assert image_path_list, '{:s} has no valid image file'.format(img_path)
    
    return image_path_list

class SegDataset(Dataset):

    def __init__(self, opt, dataset, task, is_train):

        self.opt = opt
        self.dataset = dataset
        self.task = task
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


        img_copy = img.copy()
        hog_f = get_hog_f(img_copy)

        image = transforms.ToTensor()(img)        
        mask = np.array(mask)
        if self.dataset == 'Cadis':
            mask = get_mask(self.task, mask)
        mask = torch.from_numpy(mask)
        
        return {
            'image': image.type(torch.FloatTensor),
            'mask': mask.type(torch.long),
            'hog_f':  hog_f.type(torch.long),
            'idx' : image_file.name
        }
        
