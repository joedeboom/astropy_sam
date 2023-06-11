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
import matplotlib.pyplot as plt
from tqdm import tqdm

# define a function to check if the provided region file uses an image coordinate system
# Used in the Image_Holder initialization to help remove the invalid coordinate region files
def contains_image(fi):
    f = open(fi)
    for line in f:
        if 'image' in line:
            return True
    return False

# Define helper functions
def show_mask(mask, ax, random_color=False) :
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white')  # line=?
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white')    # line=?

def show_box(box, ax) :
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0) , lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)





# Define an image holder class
class Image_Holder():
    def __init__(self, size, image_shape, paths, mode, scale_factor, data_reduction) -> None:
        # Define the mode for the image holder
        self.mode = mode

        # Define the cropped image size
        self.image_size_crop = size
        
        # Define the full image size
        self.image_size_full = image_shape

        # Define the radius scale factor
        # Images will be cropped by scale factor * radius
        # If scale factor = 1, the image size crop will be used by default
        self.scale_factor = scale_factor

        # Define the data reduction factor
        self.data_reduction = data_reduction
        
        # Define the HII and SNR region folder paths
        self.HII_folder_path = paths['HII_folder_path']
        self.SNR_folder_path = paths['SNR_folder_path']

        # Define the hii and snr csv paths
        self.HII_csv_path = paths['HII_csv_path']
        self.SNR_csv_path = paths['SNR_csv_path']

        # Define the path to the full image
        self.image_path = paths['image_path']
        
        # Check path  validity
        self.check_paths()

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
        elif mode == 'grid':
            self.images = self.finish_init_grid()
        elif mode == 'all':
            self.images = self.finish_init_region()
            self.images.extend(self.finish_init_csv)
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
    def __str__(self, abrv=True) -> str:
        s = self.print_stats()
        for image in self.images:
            if abrv:
                if image.get_id() % (50 // self.data_reduction) == 0:
                    s += str(image)
            else:
                s += str(img)
        return s
    def print_stats(self) -> str:
        s = '\n\nImage count: ' + str(len(self.images))
        s += '\nImage crop size: ' + str(self.image_size_crop)
        s += '\nScale factor: ' + str(self.scale_factor)
        count = 0.0
        if self.images[0].get_mask() is not None:
            s += '\nAverage mask count per image: ' + str(self.ave_masks()) + '\n'
        return s
    def ave_masks(self):
        count = 0.0
        for image in self.images:
            count += len(image.get_mask())
        ave = count / len(self.images)
        return ave

    # Define a function to finish the initialization of the image holder. Returns the full list of cropped image objects.
    # This function will create images from a grid of the original
    def finish_init_grid(self) -> list:
        imgs = []
        valid_dim = [1,2,3,4,5,6,9,10,12,15,18,20,27,30,31,36,45,54,60,62,90,93,108,124,135,155,180,186,270,279,310,372,465,540,558,620,837,930,1116,1395,1674,1860,2790,3348,4185,5580,8370,16740]
        
        side_len = 180
        if self.image_size_full % side_len != 0:
            print('Error. Invalid grid side length.')
            exit(1)

        for x in range(side_len // 2, self.image_size_full, side_len):
            for y in range(side_len // 2, self.image_size_full, side_len):
                imgs.append(Grid_Image((x,y), 'GRID', side_len, self.image_size_full))
        return imgs

    # Define a function to finish the initialization of the image holder. Returns the full list of cropped image objects.
    # This function reads in the data from the provided csv files.
    def finish_init_csv(self) -> list:
        imgs = []
        with open(self.HII_csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            count = self.data_reduction
            for row in reader:
                if count % self.data_reduction == 0:
                    imgs.append(CSV_Image(row, 'HII', self.image_size_crop, self.image_size_full, self.scale_factor))
                count += 1
        with open(self.SNR_csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            count = self.data_reduction
            for row in reader:
                if count % self.data_reduction == 0:
                    imgs.append(CSV_Image(row, 'SNR', self.image_size_crop, self.image_size_full, self.scale_factor))
                count += 1
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
            imgs.append(Region_Image(center, 'HII', self.image_size_crop, self.image_size_full))
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
            imgs.append(Region_Image(center, 'SNR', self.image_size_crop, self.image_size_full))
        return imgs

    # Define a function to generate and save the actual cropped image data (not just the boundaries) for each image in the holder.
    def generate_images(self):
        img_data = fits.getdata(self.image_path)[0][0]
        img_data[np.isnan(img_data)] = -1
        #img_data = fits.getdata(self.image_path)
        for image in self.images:
            curr_box = image.get_box()
            x1,y1 = curr_box['p1']
            x2,y2 = curr_box['p2']
            image.set_image(img_data[y1:y2,x1:x2])
            #if image.get_id() % 50 == 0:
                #print('raw image data for #' + str(image.get_id()))
                #print(img_data[y1:y2,x1:x2])


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

    # Define a function to check the validity of provided paths
    def check_paths(self):
        paths = [self.HII_folder_path, self.SNR_folder_path, self.HII_csv_path, self.SNR_csv_path, self.image_path]
        print('Checking paths...',end='\t')
        for path in paths:
            if not os.path.exists(path):
                print('Error. Path does not exist: ' + path)
                exit(1)
        print('Success! All paths exist.')

    # Define a function to save comparison plots of all images.
    # This function will create a new directory inside the provided path and save the images inside it.
    def save_plots(self, path):
        full_path = path + '/' + self.mode + '_scale-' + str(self.scale_factor).replace('.','-') + '_maskcount-' + str(round(self.ave_masks(),5)).replace('.','-') + '/'
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        multi_mask_images = []
        for  image in tqdm(self.images):
            image.generate_plot(full_path)
            if len(image.get_mask()) > 1:
                multi_mask_images.append(image)

        f = open(full_path + 'stats.txt', 'w')
        s = 'Files with multiple masks:'
        for image in multi_mask_images:
            s += image.get_name() + '\n'
        s += '\n\nImage Holder:\n'
        s += str(self, abrv=False)
        f.write(s)
        f.close()








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
    def print_image(self):
        if self.image is None:
            return 'Not generated'
        else:
            return self.image
    def get_mask(self):
        return self.mask
    def print_mask(self):
        if self.mask is None:
            return 'Not generated yet.'
        else:
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




# Create a Grid_Image child class.
class Grid_Image(Cropped_Image):
    def __init__(self, cen, file_type, crop_size, image_shape) -> None:
        super().__init__(cen, file_type, crop_size, image_shape)
    def __str__(self) -> str:
        s = '\n\nRegion file.'
        s += '\nID: ' + str(self.id)
        s += '\nType: ' + self.type
        s += '\nCenter: ' + str(self.center)
        s += '\nBox: ' + str(self.box)
        s += '\nSize of image: ' + str(sys.getsizeof(self.image))
        s += '\nImage: ' + str(self.print_image())
        s += '\nMask: ' + str(self.print_mask())
        return s





# Create a Region_Image child class.
class Region_Image(Cropped_Image):
    def __init__(self, cen, file_type, crop_size, image_shape) -> None:
        super().__init__(cen, file_type, crop_size, image_shape)
    def __str__(self) -> str:
        s = '\n\nRegion file.'
        s += '\nID: ' + str(self.id)
        s += '\nType: ' + self.type
        s += '\nCenter: ' + str(self.center)
        s += '\nBox: ' + str(self.box)
        s += '\nSize of image: ' + str(sys.getsizeof(self.image))
        s += '\nImage: ' + str(self.print_image())
        s += '\nMask: ' + str(self.print_mask())
        return s




# Create a CSV_Image child class.
class CSV_Image(Cropped_Image):
    def __init__(self, csv, file_type, crop_size, image_shape, scale_factor) -> None:
        # Init Super
        super().__init__((float(csv['X_center']),float(csv['Y_center'])), file_type, crop_size, image_shape)

        # Define the raw csv dictionary data
        # {'': '0', 'Name': 'MCELS-L1', 'X_center': '13668.075', 'Y_center': '7012.1650', 'Radius': '12.500000'}
        self.csv_data = csv
        
        if scale_factor != 1:
            self.box = self.regenerate_boundary(image_shape, scale_factor)    

    def get_name(self):
        return self.csv_data['Name']
    def get_X_center(self):
        return float(self.csv_data['X_center'])
    def get_Y_center(self):
        return float(self.csv_data['Y_center'])
    def get_radius(self):
        return float(self.csv_data['Radius'])
    def __str__(self) -> str:
        s = '\n\nCSV file'
        s += '\nID: ' + str(self.id)
        s += '\nType: ' + self.type
        s += '\nName: ' + self.get_name()
        s += '\nCenter: ' + str(self.center)
        s += '\nRadius: ' +  str(self.get_radius())
        s += '\nBox: ' + str(self.box)
        s += '\nSize of image: ' + str(sys.getsizeof(self.image))
        s += '\nImage: ' + str(self.print_image())
        s += '\nMask: ' + str(self.print_mask())
        return s

    def regenerate_boundary(self, full_size, scale_factor) -> None:
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
        radius = self.get_radius() * scale_factor
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

    # Define a function to generate and save plots to the corresponding file paths
    def generate_plot(self, path):    
        dest_name = path + self.type + '_' + str(len(self.mask)) + '_' + self.get_name() + '.png'
        if os.path.isfile(dest_name):
            return

        plt.subplots(figsize=(14,7))
        chart_title = self.get_name() + '  ' + self.type + '\nCenter: ' + str(self.center) + '   Radius: ' + str(self.get_radius()) + '   Mask Count: ' + str(len(self.mask))
        plt.suptitle(chart_title)

        plt.subplot(1,2,1)
        plt.imshow(self.image.copy())
        plt.axis('On')
        plt.title('Source')

        plt.subplot(1,2,2)
        #plt.imshow(self.image)
        show_anns(self.mask)
        plt.axis('On')
        plt.title('Segmented')
    
        #plt.show()
        dest_name = path + self.type + '_' + str(len(self.mask)) + '_' + self.get_name() + '.png'
        #print('Saving to ' + dest_name)
        plt.savefig(dest_name, dpi='figure', bbox_inches='tight', pad_inches=0.1, facecolor='auto', edgecolor='auto')
    
        plt.close()











