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

# Define cropped image size
cropped_size = 180
# Define full image shape
image_shape = 16740
# Define the crop image ID index
crop_id = 0


# Define a class to store cropped images
class Cropped_Image():
    def __init__(self, cen, file_type) -> None:
        # Define the center of the cropped image
        self.center = cen
        # Define the boundary of the cropped image
        self.box = generate_boundary(cen)
        # Define the type of image
        self.type = file_type
        # Define the image id
        self.id = crop_id
        crop_id += 1
    def get_center(self):
        return self.center
    def get_box(self):
        return self.box
    def get_type(self):
        return self.type
    def __str__(self) -> str:
        s = 'ID: ' + str(self.id) + '\nType: ' + self.type + '\nCenter: ' + str(self.center) + '\nBox: ' + str(self.box)
        return s
    def generate_boundary(self) -> None:
        # Define a function to generate the cropped images to pass into the model.
        # Input: the center coordinates of the region to be cropped
        # Returns: the boxed region
        #
        # x1,y1-----------*
        #  |              |
        #  |              |
        #  |              |
        #  |              |
        #  *------------x2,y2
        #
        x, y = center
        radius = cropped_size / 2
        x1 = x - radius
        if x1 < 0:
            x1 = 0
        y1 = y - radius
        if y1 < 0:
            y1 = 0
        x2 = x + radius
        if x2 > image_shape:
            x2 = image_shape
        y2 = y + radius
        if y2 > image_shape:
            y2 = image_shape
        b = {'p1':(x1,y1), 'p2':(x2,y2)}
        return b



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

# Returns a list of (x,y) coordinates
# reference annotate_reg in dataloader_reg to see how to annotate in easily in this function.
def gen_images(files, file_type):
    imgs = []
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
        center = ((left + right) // 2, (top + bot) // 2)
        imgs.append(Cropped_Image(center, file_type))
    return imgs



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
    remove_skycoord_regions(HII_reg_files, ' HII')
    remove_skycoord_regions(SNR_reg_files, ' SNR')

    # Define an image holder to hold all of the cropped images
    image_holder = []

    # Generate and append the cropped images to the image holder
    image_holder.extend(gen_images(HII_reg_files, 'HII'))
    image_holder.extend(gen_images(SNR_reg_files, 'SNR'))

    # Print out the image holder
    for image in image_holder:
        print(image)


