from os.path import splitext
import os
from os import listdir
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as tf

from skimage.feature import hog

def get_hog_f(img):
#     print(img.shape)
#     print(type(img))
#     img_copy = image.copy()
#         hog_feature = hog(img_copy.resize((64, 128)), orientations=4, pixels_per_cell=(), cells_per_block=(2, 2), visualize=False, multichannel=True)
#     ip = img.numpy().transpose(1,2,0)
#     print(f'Transposed Img {ip.shape}')
    hog_f = hog(img.resize((64, 128)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=True)
    return torch.from_numpy(hog_f)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['png','jpg'])
        
def get_image_paths(dir_list, data_root):
    
    assert os.path.isdir(data_root), '{:s} is not a valid directory'.format(data_root)
    
    image_path_list = []
    mask_path_list = []
    
    for path in dir_list:
        image_path = glob.glob(os.path.join(data_root, path, 'images', '*'))
        mask_path = glob.glob(os.path.join(data_root, path, 'labels', '*'))
        for img_fname, mask_fname in zip(image_path, mask_path):
            if is_image_file(img_fname) and is_image_file(mask_fname):
                image_path_list.append(img_fname)
                mask_path_list.append(mask_fname)
                
    assert image_path_list, '{:s} has no valid image file'.format(data_root)
    assert mask_path_list, '{:s} has no valid mask file'.format(data_root)
    
    return image_path_list, mask_path_list


class SegDataset(Dataset):
    def __init__(self, root_dir, dir_list, is_train, mask_suffix=''):
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
        self.is_train = is_train
        
        self.imgs_dir, self.masks_dir = get_image_paths(dir_list, root_dir)
#         self.scale = scale
        self.mask_suffix = mask_suffix
#         assert 0 < scale <= 1, 'Scale must be between 0 and 1'

#         self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
#                     if not file.startswith('.')]
#         logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.imgs_dir)

    @classmethod
    def preprocess(cls, image, mask, is_train):
        
        rn = torch.rand(1).item()        
        if rn > 0.15 and is_train:  
            image = transforms.Resize(448, InterpolationMode.NEAREST)(image)
            mask = transforms.Resize(448, InterpolationMode.NEAREST)(mask)
#             print('Random transform')
            rn = torch.rand(1).item()
            if rn>0.2:
                #random rotation
                rotation_params = transforms.RandomRotation.get_params([-45,45])
                image = transforms.functional.rotate(image, rotation_params)
                mask = transforms.functional.rotate(mask, rotation_params)
#             print(rotation_params)
            #random flip
            rn = torch.rand(1).item()
            if rn>0.3: 
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            rn = torch.rand(1).item()
            if rn>0.4: 
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
                
            #adjust brightness 
            scale = torch.rand(1).item()/5
            if torch.rand(1).item()>0.5:
                image = tf.adjust_brightness(image, 1-scale)
                   
            else:
                image = tf.adjust_brightness(image, 1+scale)
                
            #adjust contrast
            scale = torch.rand(1).item()/10
            if torch.rand(1).item()>0.5:
                image = tf.adjust_contrast(image, 1-scale)
                   
            else:
                image = tf.adjust_contrast(image, 1+scale)
                
#             #adjust saturation
#             rn = torch.rand(1).item()
#             if rn<0.2:
#                 scale = rn + 1
#                 image = tf.adjust_saturation(image, scale)
#             elif rn>0.8:
#                 image = image = tf.adjust_saturation(image, rn)
                    
                    
        #Crop
        if is_train:
            rn = torch.rand(1).item()         
            if rn<0.8:
                #Random Crop
                crop_params = transforms.RandomCrop.get_params(image, (256, 256))

        #         print(crop_params)      
                image = transforms.functional.crop(image, *crop_params)
                mask = transforms.functional.crop(mask, *crop_params)

            else:
                #Center Crop
                image = transforms.CenterCrop(256)(image)
                mask = transforms.CenterCrop(256)(mask)
#         else:
#             image = transforms.Resize(256, InterpolationMode.NEAREST)(image)
#             mask = transforms.Resize(256, InterpolationMode.NEAREST)(mask)
            
        img_c = image.copy()
        hog_f = get_hog_f(img_c)
        
        to_tensor = transforms.ToTensor()        
        image = to_tensor(image)
        
        mask = np.array(mask)       
#         mask = np.expand_dims(mask, 0)
        mask = torch.from_numpy(mask)
        
        return image, mask, hog_f
        


    def __getitem__(self, idx):
#         idx = self.ids[i]
#         print(glob(self.masks_dir +'/'+ idx + self.mask_suffix + '.*') )
#         print(idx)
        mask_file = self.masks_dir[idx]   
        img_file = self.imgs_dir[idx]
#         print(img_file)
#         assert len(mask_file) == 1, \
#             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
#         assert len(img_file) == 1, \
#             f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img, mask, hog_f  = self.preprocess(img, mask, self.is_train)
#         mask = self.preprocess(mask, self.scale)
        
        return {
            'image': img.type(torch.FloatTensor),
            'mask': mask.type(torch.FloatTensor),
            'hog_f':  hog_f.type(torch.FloatTensor),
            'idx' : img_file.split('/')[-1].split('.')[0]
        }