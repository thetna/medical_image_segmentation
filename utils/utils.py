import os
import random
import torch
import glob
import numpy as np
import yaml
from datetime import datetime
import logging

import torchvision.transforms.functional as TF
from torchvision import transforms


from skimage.feature import hog

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
    # n_dim = tensor.dim()

    tensor = tensor.squeeze().float().cpu()

    mask_rgb = np.zeros(tensor.shape[:2] + (3,), dtype=np.uint8)
    for cnt in range(len(colormap)):
        mask_rgb[tensor == cnt] = colormap[cnt]

    return mask_rgb.astype(out_type)

def save_model(cfg, model_dict, current_step):
    save_dir = cfg["path"]["checkpoints"]["ckpt_dir"]
    makedirs(save_dir)
    torch.save(model_dict["u2net"].state_dict(), save_dir + "u2net%d.pth" % (current_step))
    
    if cfg["with_HOG"]:
        torch.save(model_dict["hog_reg"].state_dict(), save_dir + "hog_reg%d.pth" % (current_step))

    
    
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def get_dir_lists(fold):
    
    all_dir_list = ['Video001', 'Video002', 'Video003', 'Video004', 'Video005', 'Video006',  'Video007', 'Video008', 'Video009', 'Video011', 'Video013', 'Video014', 'Video016', 'Video017', 'Video018', 'Video019', 'Video022', 'Video023']
    
    if fold == 1:
        val_dir_list = ['Video001', 'Video006', 'Video016']
    
    elif fold == 2 :
        val_dir_list = ['Video002', 'Video011', 'Video018']
    
    elif fold == 3:
        val_dir_list = ['Video004', 'Video019', 'Video023', ]
        
    elif fold == 4:
        val_dir_list = ['Video003', 'Video005', 'Video014']
        
    elif fold == 5:
        val_dir_list = ['Video007', 'Video008', 'Video022']
        
    elif fold == 6:
        val_dir_list = ['Video009', 'Video013', 'Video017']
        
    else:
        print("Fold not found.")
        
        return [], []
    
    train_dir_list = [d for d in all_dir_list if d not in val_dir_list]
    
    print(train_dir_list, val_dir_list)
    
    return train_dir_list, val_dir_list
    
def augmentation(opt, images):
    
    image, mask = images

    hflip = opt['train']['flip'] and random.random() < 0.5
    vflip = opt['train']['flip'] and random.random() < 0.5
    rot = opt['train']['rotation'] and random.random() < 0.5
    brightness = opt['train']['brightness'] and random.random() < 0.5
    contrast = opt['train']['contrast'] and random.random() < 0.5

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
        if brightness:
            bf = torch.randint(80, 120, (1, )).item() / 100
            img = TF.adjust_brightness(img, brightness_factor=bf)
        if contrast:
            cf = torch.randint(90, 110, (1, )).item() / 100
            img = TF.adjust_contrast(img, contrast_factor=cf)
        return img, mask

    return _augment(image, mask) 

def transform(opt, images, augment=True):
    
    image, mask = images
    resize = transforms.Resize(size=(opt['train']['resize'], opt['train']['resize']))
    image = resize(image)
    mask = resize(mask)
    
    if augment == True:
        image, mask = augmentation(opt, [image, mask])
        
        #Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(opt['train']['crop'], opt['train']['crop']))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
    return image, mask

def get_hog_f(config, img):

    hog_bins = config["train"]["hog_bins"]
    ppc = config["train"]["pix_per_cell"]
    cpb =  config["train"]["cells_per_block"]
    hog_f = hog(img.resize((64, 128)), orientations= hog_bins, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb), visualize=False, multichannel=True)
    return torch.from_numpy(hog_f)