# Import necessary libraries
import sys
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import cv2
import os
import glob
import pickle
from regions import Regions
from astropy.io import fits



# define a function to check if the provided region file uses an image coordinate system
# Used in the Image_Holder initialization to help remove the invalid coordinate region files
def contains_image(fi):
    f = open(fi)
    for line in f:
        if 'image' in line:
            return True
    return False





# Define an image holder class
class Image_Holder():
    def __init__(self, size, image_shape, hii_folder_path, snr_folder_path, hii_csv_path, snr_csv_path, img_path, mode) -> None:
        # Define the mode for the image holder
        self.mode = mode

        # Define the cropped image size
        self.image_size_crop = size

        # Define the full image size
        self.image_size_full = image_shape
        
        # Define the HII and SNR region folder paths
        self.HII_folder_path = hii_folder_path
        self.SNR_folder_path = snr_folder_path

        # Define the hii and snr csv paths
        self.HII_csv_path = hii_csv_path
        self.SNR_csv_path = snr_csv_path

        # Define the path to the full image
        self.image_path = img_path

        # Additional initialization
        # Define the HII and SNR region files
        self.HII_reg_files = glob.glob(os.path.join(self.HII_folder_path, '*.reg'))
        self.SNR_reg_files = glob.glob(os.path.join(self.SNR_folder_path, '*.reg'))

        # Remove any regions using an invalid coordinate system
        self.remove_skycoord_regions()

        # Define the list to hold the cropped image objects
        if mode == 'region':
            self.images = self.finish_init_region()
        elif mode == 'csv':
            self.images = self.finish_init_csv()
        elif mode == 'all':
            self.images = self.finish_init_region()
            self.images.extend(self.finish_init_csv)
        else:
            print('Invalid mode.')
            exit(1)
        print('Created ' + str(len(self.images)) + ' images.')

    def get_image_size_crop(self) -> int:
        return self.image_size_crop
    def get_image_size_full(self) -> int:
        return self.image_size_full
    def get_images(self) -> list:
        return self.images
    def __str__(self) -> str:
        s = ''
        for image in image_holder.get_images():
            s += str(image)
        return s

    # Define a function to finish the initialization of the image holder. Returns the full list of cropped image objects.
    # This function reads in the data from the provided csv files.
    def finish_init_csv(self) -> list:
        imgs = []
        with open(self.HII_csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                imgs.append(CSV_Image(row, 'HII', self.image_size_crop, self.image_size_full))
        with open(self.SNR_csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                imgs.append(CSV_Image(row, 'SNR', self.image_size_crop, self.image_size_full))
        return imgs


    # Define a function to finish the initialization of the image holder. Returns the full list of cropped image objects.
    # This function loads computes the centers of each image via the region files.
    def finish_init_region(self) -> list:
        imgs = []
        for file in self.HII_reg_files:
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
            imgs.append(Region_Image(center, 'HII', self.image_size_crop, self.image_size_full, {}))
        for file in self.SNR_reg_files:
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
            imgs.append(Region_Image(center, 'SNR', self.image_size_crop, self.image_size_full, {}))
        return imgs

    # Define a function to generate and save the actual cropped image data (not just the boundaries) for each image in the holder.
    def generate_images(self):
        #img_data = fits.getdata(self.image_path)[0][0]
        #img_data[np.isnan(img_data)] = -1
        img_data = fits.getdata(self.image_path)
        for image in self.images:
            curr_box = image.get_box()
            x1,y1 = curr_box['p1']
            x2,y2 = curr_box['p2']
            image.set_image(img_data[y1:y2,x1:x2])
    
    # Define a function to clear the image data for all images (for saving)
    def clear_images(self):
        for image in self.images:
            image.clear_image()

    # Define a function to remove skycoord regions
    def remove_skycoord_regions(self) -> None:
        count = 0
        removed = []
        for file in self.HII_reg_files:
            if not contains_image(file):
                self.HII_reg_files.remove(file)
                removed.append(file)
                count += 1
        for file in self.SNR_reg_files:
            if not contains_image(file):
                self.SNR_reg_files.remove(file)
                removed.append(file)
                count += 1
        print('Removed ' + str(count) + ' invalid files: ' + str(removed))








# Define classes.
# Parent class: Cropped_Image
# Child classes: Region_Image, CSV_Image, tic tac toe image


# Define a Cropped_Image parent class.
class Cropped_Image():
    index = 0
    def __init__(self, cen, file_type, crop_size, image_shape) -> None:
        # Define the center of the cropped image
        self.center = cen

        # Define the boundary of the cropped image
        self.box = self.generate_boundary(crop_size, image_shape)

        # Define the type of image
        self.type = file_type

        # Define the image id
        self.id = Cropped_Image.index
        Cropped_Image.index += 1

        # Define where the image data is stored.
        # The image data is generated each time it needs to be used via file path to the whole image
        self.image = None

        # Define the mask for the image
        self.mask = None

    def get_center(self):
        return self.center
    def get_box(self):
        return self.box
    def get_type(self):
        return self.type
    def get_id(self):
        return self.id
    def get_image(self):
        return self.image
    def get_mask(self):
        return self.mask
    def __str__(self) -> str:
        s = '\nID: ' + str(self.id) + '\nType: ' + self.type + '\nCenter: ' + str(self.center) + '\nBox: ' + str(self.box) + '\nSize of image: ' + str(sys.getsizeof(self.image)) + '\nMask: \n' + str(self.mask)
        return s
    def set_image(self, img) -> None:
        self.image = img.copy()
    def clear_image(self) -> None:
        self.image = None
    def set_mask(self, m) -> None:
        self.mask = m
    def generate_boundary(self, crop_size, full_size) -> None:
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
        x, y = self.center
        radius = int(crop_size / 2)
        x1 = x - radius
        if x1 < 0:
            x1 = 0
        y1 = y - radius
        if y1 < 0:
            y1 = 0
        x2 = x + radius
        if x2 > full_size:
            x2 = full_size
        y2 = y + radius
        if y2 > full_size:
            y2 = full_size
        b = {'p1':(int(x1),int(y1)), 'p2':(int(x2),int(y2))}
        return b





# Create a Region_Image child class.
class Region_Image(Cropped_Image):
    index = 0
    def __init__(self, cen, file_type, crop_size, image_shape) -> None:
        super().__init__(cen, file_type, crop_size, image_shape)
    def __str__(self) -> str:
        s = '\nRegion file.'
        s += '\nID: ' + str(self.id)
        s += '\nType: ' + self.type
        s += '\nCenter: ' + str(self.center)
        s += '\nBox: ' + str(self.box)
        s += '\nSize of image: ' + str(sys.getsizeof(self.image))
        s += '\nMask: \n' + str(self.mask)
        return s




# Create a CSV_Image child class.
class CSV_Image(Cropped_Image):
    index = 0
    def __init__(self, csv, file_type, crop_size, image_shape) -> None:
        # Init Super
        super().__init__((int(csv['X_center']),int(csv['Y_center'])), file_type, crop_size, image_shape)

        # Define the raw csv dictionary data
        # {'': '0', 'Name': 'MCELS-L1', 'X_center': '13668.075', 'Y_center': '7012.1650', 'Radius': '12.500000'}
        self.csv_data = csv
    
    def get_name(self):
        return self.csv_data['Name']
    def get_X_center(self):
        return float(self.csv_data['X_center'])
    def get_Y_center(self):
        return float(self.csv_data['Y_center'])
    def get_radius(self):
        return float(self.csv_data['Radius'])
    def __str__(self) -> str:
        s = '\nCSV file'
        s = '\nID: ' + str(self.id)
        s += '\nType: ' + self.type
        s += '\nCenter: ' + str(self.center)
        s += '\nRadius: ' +  str(self.get_radius())
        s += '\nBox: ' + str(self.box)
        s += '\nSize of image: ' + str(sys.getsizeof(self.image))
        s += '\nMask: \n' + str(self.mask)
        return s
















