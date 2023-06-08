# Import necessary libraries
import old_model.dataloader_reg
from utils.data_utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import cv2
import os
import glob
from regions import Regions

# hard code parameter for file and image and diretory
size = 180
image_shape = 16740

HII_folder_path = './drive/MyDrive/Research/LMC/HII_boundaries'
SNR_folder_path = './drive/MyDrive/Research/LMC/SNR_boundaries'

HII_reg_files = glob.glob(os.path.join(HII_folder_path, '*.reg'))
SNR_reg_files = glob.glob(os.path.join(SNR_folder_path, '*.reg'))


# -------------------------------------------------------------------------------------
# Define a function to remove skycoord regions
def remove_skycoord_regions():
    print('removing skycoord regions')
    count1 = 0
    count2 = 0
    removed = []
    for file in HII_reg_files:
        if not contains_image(file):
            HII_reg_files.remove(file)
            removed.append(file)
            count1 += 1
    for file in SNR_reg_files:
        if not contains_image(file):
            SNR_reg_files.remove(file)
            removed.append(file)
            count2 += 1
    print('Removed ' + str(count1) + ' files from HII and ' + str(count2) + ' files from SNR. ' + str(count1+count2) + ' items in total:\n' + str(removed))

# define a function to check if the provided region file uses an image coordinate system
def contains_image(fi):
    f = open(fi)
    for line in f:
        if 'image' in line:
            return True
    return False

# -------------------------------------------------------------------------------------

remove_skycoord_regions()

# Have each 180x180 image in a list ready to pass to sam in a loop.
ann = dataloader_reg.generate_annotation()
print(ann)



