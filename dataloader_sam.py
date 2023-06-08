# Import necessary libraries
#from old_model.dataloader_reg import generate_annotation, annotate_reg, center_arr, get_center, point_in_polygon
#from utils.data_utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import cv2
import os
import glob
from regions import Regions


# -------------------------------------------------------------------------------------
# Define a function to remove skycoord regions
def remove_skycoord_regions(files, coord_type):
    count = 0
    removed = []
    for file in files:
        if not contains_image(file):
            files.remove(file)
            removed.append(file)
            count += 1
    print('Removed ' + str(count) + coord_type + ' files.')

# define a function to check if the provided region file uses an image coordinate system
def contains_image(fi):
    f = open(fi)
    for line in f:
        if 'image' in line:
            return True
    return False

# -------------------------------------------------------------------------------------


def get_centers(files):
    centers = []
    for file in files:
        regions = Regions.read(file, format='ds9')
        Xs = regions[0].vertices.x
        Ys = regions[0].vertices.y
        length = len(Xs)
        left = float('inf')
        right = float('-inf')
        bot = float('inf')
        top = float('-inf')
        for i in range(length):
            if Xs[i] > right:
                right = Xs[i]
            if Xs[i] < left:
                left = Xs[i]
            if Ys[i] < bot:
                bot = Ys[i]
            if Ys[i] > top:
                top = Ys[i]
        centers.append(((left + right) // 2, (top + bot) // 2))
    return centers

def generate_images(path, regs):
    #img_data = fits2matrix(path)[0][0]
    #img_data[np.isnan(img_data)] = -1
    image_bgr = cv2.imread(path)
    images = []
    for file in regs:
        center = get_center


if __name__ == "__main__":

    # Define cropped image size
    size = 180
    
    # Define full image shape
    image_shape = 16740

    # Define data paths
    HII_folder_path = './drive/MyDrive/Research/LMC/HII_boundaries'
    SNR_folder_path = './drive/MyDrive/Research/LMC/SNR_boundaries'
    image_path = './drive/MyDrive/Research/LMC/lmc_askap_aconf.fits'

    # Define lists of region files
    HII_reg_files = glob.glob(os.path.join(HII_folder_path, '*.reg'))
    SNR_reg_files = glob.glob(os.path.join(SNR_folder_path, '*.reg'))
    
    # Remove files with invalid coordinates
    print('removing skycoord regions')
    remove_skycoord_regions(HII_reg_files, ' HII ')
    remove_skycoord_regions(SNR_reg_files, ' SNR ')

    # Generate x,y coordinates for each region file
    hii_centers = get_centers(HII_reg_files)
    snr_centers = get_centers(SNR_reg_files)

    # Generate a cropped image for each region
    #hii_images = generate_images(image_path, HII_reg_files)
    #snr_images = generate_images(image_path, SNR_reg_files)



