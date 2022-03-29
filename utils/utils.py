import random
import glob
import torch
import numpy as np

import os
import yaml
import logging
from PIL import Image
from datetime import datetime

import torchvision.transforms.functional as TF
from torchvision import transforms

from models import UNet, U2Net, UNet_HOG_Decoder, U2Net_HOG_Decoder
from utils.cadis_vis import *


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, level=logging.INFO, screen=False):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, logger_name + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def get_logger(path):
    mkdirs(path)
    setup_logger('train', path, level=logging.INFO, screen=True)
    setup_logger('val', path, level=logging.INFO, screen=False)
    return logging.getLogger('train'), logging.getLogger('val')


def resume_model(opt, seg_net, hog_decoder=None):
    if opt['seg_net']['resume_path'] is not None and seg_net is not None:
        seg_net.load_state_dict(torch.load(opt['seg_net']['resume_path'], map_location='cpu'))
    if opt['hog_decoder']['resume_path'] is not None and hog_decoder is not None:
        hog_decoder.load_state_dict(torch.load(opt['hog_decoder']['resume_path'], map_location='cpu'))

    return seg_net, hog_decoder


def define_networks(opt):
    
    if opt['network'] == 'U2Net+HOG':

        seg_net = U2Net(opt['seg_net']['in_nc'], opt['seg_net']['out_nc'])
        hog_decoder = U2Net_HOG_Decoder(opt['hog_decoder']['out_dim'])

        seg_net, hog_decoder = resume_model(opt, seg_net, hog_decoder)

        seg_net.to(opt['device'])
        seg_net.train()
        hog_decoder.to(opt['device'])
        hog_decoder.train()

        return seg_net, hog_decoder

    else:
        print('Network type is not defined.')

def augmentation(image, mask):
    
    hflip = True and random.random() < 0.5
    vflip = True and random.random() < 0.5
    rot = True and random.random() < 0.5

    def _augment(img, mask):
        if hflip:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if vflip:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if rot:
            angle = torch.randint(-45, 45, (1, )).item()
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        return img, mask

    return _augment(image, mask) 

def transform_data(image, mask, augment=True):
    
    resize = transforms.Resize(size=(418, 418), interpolation=Image.NEAREST)
    image = resize(image)
    mask = resize(mask)
    
    if augment == True:
        image, mask = augmentation(image, mask)
    
        params = transforms.RandomCrop.get_params(
            image, output_size=(256, 256))
        image = TF.crop(image, *params)
        mask = TF.crop(mask, *params)
    return image, mask


def get_mask(task, mask):

    if task == 1:
        mask = remap_mask(mask, class_remapping=class_remapping_exp1)
    elif task == 2:
        mask = remap_mask(mask, class_remapping=class_remapping_exp2)
    elif task == 3:
        mask = remap_mask(mask, class_remapping=class_remapping_exp3)
    return mask

def save_validation(pred, imgs, true_masks, idx, current_step, save_path_img, dataset, task):

    if dataset == 'Cadis':
        if task == 1:
            colormap = get_remapped_colormap(class_remapping_exp1)
        elif task == 2:
            colormap = get_remapped_colormap(class_remapping_exp2)
        elif task == 3:
            colormap = get_remapped_colormap(class_remapping_exp3)
    
    pred_img = mask_to_colormap(pred.squeeze().float().cpu(), colormap)
    temp_inp = imgs.squeeze().float().cpu().numpy().transpose(1,2,0)*255
    gt_mask = mask_to_colormap(true_masks.squeeze().float().cpu(), colormap)
    border = np.ones((gt_mask.shape[0], 5, 3))*255
    imgs_comb = np.hstack((temp_inp, border.astype(np.uint8), gt_mask, border.astype(np.uint8), pred_img))
    imgs_pil = Image.fromarray(imgs_comb.astype(np.uint8))
    imgs_pil.save(os.path.join(save_path_img, f'{idx}__{current_step}.png'))

class_remapping_exp1 = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [4],
        5: [5],
        6: [6],
        7: [
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
        ],
    }

class_remapping_exp2 = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [4],
        5: [5],
        6: [6],
        7: [7, 8, 10, 27, 20, 32],
        8: [9, 22],
        9: [11, 33],
        10: [12, 28],
        11: [13, 21],
        12: [14, 24],
        13: [15, 18],
        14: [16, 23],
        15: [17],
        16: [19],
        17: [25, 26, 29, 30, 31, 34, 35],
    }

class_remapping_exp3 = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [4],
        5: [5],
        6: [6],
        7: [7],
        8: [8],
        9: [9],
        10: [10],
        11: [11],
        12: [12],
        13: [13],
        14: [14],
        15: [15],
        16: [16],
        17: [17],
        18: [18],
        19: [19],
        20: [20],
        21: [21],
        22: [22],
        23: [23],
        24: [24],
        25: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
