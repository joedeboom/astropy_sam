# Import necessary libraries
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
train_snrs, test_snrs = [], []
train_hii, test_hii = [], []

HII_folder_path = './drive/MyDrive/Astropy/LMC/HII_boundaries'
SNR_folder_path = './drive/MyDrive/Astropy/LMC/SNR_boundaries'

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




# Define a function to check if a point is inside a polygon
def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

# Define a function to get the center of a region
# This function reads a region file, extracts the x and y coordinates of the vertices,
# then calculates and returns the center coordinates
def get_center(path):
    regions = Regions.read(path, format='ds9')
    # print(path)
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
    return ((left + right) // 2, (top + bot) // 2)

# Define a function to get the center of a region
# This function reads a region file, extracts the x and y coordinates of the vertices,
# then calculates and returns the center coordinates
def center_arr():
    arr = []
    for file in HII_reg_files:
        center = annotate_reg(file, ann, 1)
        if count == 4:
            count = 0
            test_hii.append(center)
        else:
            train_hii.append(center)
            count += 1

    count = 0
    for file in SNR_reg_files:
        center = annotate_reg(file, ann, 2)
        if count == 4:
            count = 0
            test_snrs.append(center)
        else:
            train_snrs.append(center)
            count += 1

    return ann

# Define a function to annotate the regions and return their centers
# This function reads a region file, gets the bounding box of the region,
# then iterates over the pixels within the bounding box, labeling those that are inside the polygon
# It also returns the center of the region
def annotate_reg(path, arr, label):
    regions = Regions.read(path, format='ds9')
    #print('\n'+path)
    #print(regions)
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

    poly = list(zip(Xs, Ys))
    for x in range(round(left), round(right) + 1):
        for y in range(round(bot), round(top) + 1):
            if point_in_polygon(x, y, poly):
                if arr[y][x] == 1 and label == 2:
                    arr[y][x] = 3
                else:
                    arr[y][x] = label

    return ((left + right) // 2, (top + bot) // 2)

# Define a function to generate annotation for regions
# This function initializes an empty annotation array, then iterates over the region files,
# calling the annotate_reg function for each file and updating the annotation array
# It also maintains lists of the centers of the regions for training and testing
def generate_annotation():
    ann = np.zeros([image_shape, image_shape])
    count = 0
    
    num1 = 0
    num2 = 0
    #print('length of HII_reg_files: ' + str(len(HII_reg_files)))
    #print('length of SNR_reg_files: ' + str(len(SNR_reg_files))) 
    #print(HII_reg_files)
    #print(SNR_reg_files)
    remove_skycoord_regions()
    #print('length of HII_reg_files: ' + str(len(HII_reg_files)))
    #print('length of SNR_reg_files: ' + str(len(SNR_reg_files)))

    for file in HII_reg_files:
        num1 += 1
        center = annotate_reg(file, ann, 1)
        if count == 4:
            count = 0
            test_hii.append(center)
        else:
            train_hii.append(center)
            count += 1

    count = 0
    for file in SNR_reg_files:
        num2 += 1
        center = annotate_reg(file, ann, 2)
        if count == 4:
            count = 0
            test_snrs.append(center)
        else:
            train_snrs.append(center)
            count += 1

    print('num1 = ' + str(num1) + '\nnum2 = ' + str(num2))
    return ann

# Define a function to get the data and labels for training and testing
# This function reads an image file, generates the annotation for the regions,
# then extracts patches of the image and the corresponding labels around the center of each region
# for both training and testing. The patches and labels are returned as numpy arrays
def get_data_and_label(path, rgb=False):
    img_data = fits2matrix(path)[0][0]
    # img_data = fits2matrix(path)
    img_data[np.isnan(img_data)] = -1
    ann = generate_annotation()

    x_train, y_train, x_test, y_test = [], [], [], []

    for x, y in train_snrs:
        x = int(x)
        y = int(y)
        x_train.append(img_data[y - size:y, x - size:x])
        x_train.append(img_data[y:y + size, x:x + size])
        x_train.append(img_data[y:y + size, x - size:x])
        x_train.append(img_data[y - size:y, x:x + size])
        x_train.append(img_data[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

        y_train.append(ann[y - size:y, x - size:x])
        y_train.append(ann[y:y + size, x:x + size])
        y_train.append(ann[y:y + size, x - size:x])
        y_train.append(ann[y - size:y, x:x + size])
        y_train.append(ann[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

    for x, y in train_hii:
        x = int(x)
        y = int(y)
        x_train.append(img_data[y - size:y, x - size:x])
        x_train.append(img_data[y:y + size, x:x + size])
        x_train.append(img_data[y:y + size, x - size:x])
        x_train.append(img_data[y - size:y, x:x + size])
        x_train.append(img_data[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

        y_train.append(ann[y - size:y, x - size:x])
        y_train.append(ann[y:y + size, x:x + size])
        y_train.append(ann[y:y + size, x - size:x])
        y_train.append(ann[y - size:y, x:x + size])
        y_train.append(ann[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

    for x, y in test_snrs:
        x = int(x)
        y = int(y)
        x_test.append(img_data[y - size:y, x - size:x])
        x_test.append(img_data[y:y + size, x:x + size])
        x_test.append(img_data[y:y + size, x - size:x])
        x_test.append(img_data[y - size:y, x:x + size])
        x_test.append(img_data[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

        y_test.append(ann[y - size:y, x - size:x])
        y_test.append(ann[y:y + size, x:x + size])
        y_test.append(ann[y:y + size, x - size:x])
        y_test.append(ann[y - size:y, x:x + size])
        y_test.append(ann[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

    for x, y in test_hii:
        x = int(x)
        y = int(y)
        x_test.append(img_data[y - size:y, x - size:x])
        x_test.append(img_data[y:y + size, x:x + size])
        x_test.append(img_data[y:y + size, x - size:x])
        x_test.append(img_data[y - size:y, x:x + size])
        x_test.append(img_data[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

        y_test.append(ann[y - size:y, x - size:x])
        y_test.append(ann[y:y + size, x:x + size])
        y_test.append(ann[y:y + size, x - size:x])
        y_test.append(ann[y - size:y, x:x + size])
        y_test.append(ann[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

# Define a function to get the datasets for training and testing
# This function calls the get_data_and_label function to get the data and labels,
# then wraps them in PyTorch TensorDataset objects
def get_set(path):
    x_train, y_train, x_test, y_test = get_data_and_label(path)
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
    x_train = torch.unsqueeze(x_train, 1)
    x_test = torch.unsqueeze(x_test, 1)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    return train_dataset, test_dataset

# Define a function to get the data loaders for training and testing
# This function calls the get_set function to get the datasets, then wraps them in PyTorch DataLoader objects,
# which can be used to iterate over the data in mini-batches
def get_dataloader(path, batch_size=8, shuffle=True):
    train_dataset, test_dataset = get_set(path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return train_loader, test_loader





# train, test = get_dataloader('./LMC/lmc_askap_aconf.fits')
# for x, y in test:
#     print(x.size())
#     for i in range(8):
#         img = torch.squeeze(x[i], 0)
#         img = torch.stack([img, img, img], 2).numpy()
#         print(img.shape)
#         cv2.imshow('1', img * 255)
#         cv2.waitKey(0)
#     break
