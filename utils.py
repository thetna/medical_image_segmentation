import random
import torch
import numpy as np

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

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)