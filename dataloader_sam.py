# Import necessary libraries
import sys
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import cv2
import os
import math
import copy
import time
import glob
import pickle
from regions import Regions
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

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
def get_center(poly):
    left = float('inf')
    right = float('-inf')
    bot = float('inf')
    top = float('-inf')

    for point in poly:
        x, y = point
        if x > right:
            right = x
        if x < left:
            left = x
        if y < bot:
            bot = y
        if y > top:
            top = y
    return ((left + right) // 2, (top + bot) // 2)




# Define an image holder class
class Image_Holder():
    def __init__(self, size, image_shape, paths, mode, scale_factor, data_reduction, normalization, SAM_mode, brightness_factors) -> None:
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

        # Define the image normalization technique
        self.normalization = normalization
        
        # Define the brightness factors to be applied to the images
        self.brightness_factors = brightness_factors

        # Define the mode SAM will process the images
        self.SAM_mode = SAM_mode
        
        # Define the HII and SNR region folder paths
        self.HII_folder_path = paths['HII_folder_path']
        self.SNR_folder_path = paths['SNR_folder_path']

        # Define the hii and snr csv paths
        self.HII_csv_path = paths['HII_csv_path']
        self.SNR_csv_path = paths['SNR_csv_path']

        # Define the path to the full image
        self.image_path = paths['image_path']
        
        # Check path validity
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
    def __str__(self) -> str:
        s = self.print_stats()
        for image in self.images:
            if self.data_reduction < 51:
                if image.get_id() % (50 // self.data_reduction) == 0:
                    s += str(image)
            else:
                s += str(image)
        return s
    def print_stats(self) -> str:
        s = '\n\nImage count: ' + str(len(self.images))
        s += '\nImage crop size: ' + str(self.image_size_crop)
        s += '\nScale factor: ' + str(self.scale_factor)
        s += '\nData reduction: ' + str(self.data_reduction)
        s += '\nNormalization: ' + self.normalization
        s += '\nSAM mode: ' + self.SAM_mode
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
    # This function computes the centers of each image via the region files.
    def finish_init_region(self) -> list:
        imgs = []
        count = self.data_reduction
        for file in self.HII_reg_files:
            if count % self.data_reduction == 0:
                name = file.split('/')[-1].split('.')[0]
                region = Regions.read(file, format='ds9')
                imgs.append(Region_Image(region, 'HII', self.image_size_crop, self.image_size_full, self.scale_factor, name))
            count += 1
        count = self.data_reduction
        for file in self.SNR_reg_files:
            if count % self.data_reduction == 0:
                name = file.split('/')[-1].split('.')[0]
                region = Regions.read(file, format='ds9')
                imgs.append(Region_Image(region, 'SNR', self.image_size_crop, self.image_size_full, self.scale_factor, name))
            count += 1
        return imgs

    # Define a function to generate and save the actual cropped image data (not just the boundaries) for each image in the holder.
    def generate_images(self):
        new_images = []

        img_data = fits.getdata(self.image_path)[0][0]
        #img_data[np.isnan(img_data)] = -1
        
        for image in self.images:
            curr_box = image.get_box()
            x1,y1 = curr_box['p1']
            x2,y2 = curr_box['p2']
            img = np.array(img_data[y1:y2,x1:x2])

            if self.normalization == 'og':
                # Normalize the image data
                normalized_data = (img - img.min()) / (img.max() - img.min())
                
                # Create new cropped image object duplicates with varying brightnesses
                for bf in self.brightness_factors:
                    new_img = copy.deepcopy(image)

                    # Update the brightness factor
                    new_img.set_brightness_factor(bf)

                    # Adjust the brightness by multiplying with the factor
                    brightness_adjusted = normalized_data * bf
                
                    # Clip the pixel values to the valid range of [0, 1]
                    brightness_adjusted = np.clip(brightness_adjusted, 0, 1)

                    # Convert the adjusted grayscale image to RGB and unit8
                    rgb_image = cv2.cvtColor((brightness_adjusted * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

                    # Set the image
                    new_img.set_image(rgb_image)

                    # Add new image to the new image list
                    new_images.append(new_img)
            
            else:
            
                # Normalize the image data
                normalized_data = (img - img.min()) / (img.max() - img.min())

                # Apply histogram equalization to enhance brightness
                equalized_data = cv2.equalizeHist((normalized_data * 255).astype(np.uint8)) / 255

                # Create new cropped image object duplicates with varying brightnesses
                for bf in self.brightness_factors:
                    new_img = copy.deepcopy(image)
                    
                    # Update the brightness factor
                    new_img.set_brightness_factor(bf)

                    # Adjust the brightness by multiplying with the factor
                    brightness_adjusted = equalized_data * bf

                    # Clip the pixel values to the valid range of [0, 1]
                    brightness_adjusted = np.clip(brightness_adjusted, 0, 1)

                    # Convert the adjusted grayscale image to RGB and unit8
                    rgb_image = cv2.cvtColor((brightness_adjusted * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

                    # Set the image
                    new_img.set_image(rgb_image)

                    # Add new image to the new image list
                    new_images.append(new_img)
        
        # Save new images to image holder
        self.images = new_images

    # Define a function to generate the corresponding annotation image from the region file
    #def generate_image_annotations(self):

    # Define a function to attach region annotations to the correct csv cropped image
    def attach_region_to_csv(self):
        reg_files = self.HII_reg_files.copy()
        reg_files.extend(self.SNR_reg_files.copy())
        for file in reg_files:
            regions = Regions.read(file, format='ds9')
            Xs = regions[0].vertices.x
            Ys = regions[0].vertices.y
            poly = list(zip(Xs, Ys))

            # Attempt to match each image center to a polygon
            for image in self.images:
                x = image.get_X_center()
                y = image.get_Y_center()
                if point_in_polygon(x,y,poly):
                    # Image center is inside of the current polygon
                    if image.get_polygon() is None:
                        # Image has not had a polygon assigned to it yet
                        print('Attaching ' + file + ' to ' + image.get_name())      
                        # Set the polygon to image
                        image.set_polygon(poly)
                    else:
                        # Image already has a polygon. Check if this new polygon is a better fit.
                        q = [x,y]
                        curr_poly_cent = get_center(image.get_polygon())
                        p = [curr_poly_cent[0], curr_poly_cent[1]]
                        curr_dist = math.dist(p,q)
                        proposed_poly_cent = get_center(poly)
                        p = [proposed_poly_cent[0], proposed_poly_cent[1]]
                        proposed_dist = math.dist(p,q)
                        if proposed_dist < curr_dist:
                            # The proposed polygon is a better fit. Update the object with new polygon
                            print('Updating ' + file + ' to ' + image.get_name())
                            image.set_polygon(poly)

        print('Images with no polygons: ')
        for image in self.images:
            if image.get_polygon() is None:
                print(image.get_name())
        print()

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

    # Define a function to output all stats to file
    def write_stats(self, path, multi_mask_images):
        f = open(path, 'w')
        s = 'Files with multiple masks:\n'
        for image in multi_mask_images:
            s += 'ID-' + str(image.get_id()) + '\tMask count: ' + str(image.get_mask_count()) + '\n'
        s += '\n\n'
        s += self.print_stats()
        for image in self.images:
            s += str(image)
        f.write(s)
        f.close()


    # Define a function to save comparison plots of all images.
    # This function will create a new directory inside the provided path and save the images inside it.
    def save_plots(self, path, save_img):
        full_path = path + '/' + self.mode + '_' + self.SAM_mode + '_' + self.normalization + '_' + time.strftime("%Y%m%d-%H%M%S") + '/'
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        multi_mask_images = []
        
        # Create pdf
        pdf = PdfPages(full_path + 'results.pdf')
        for image in tqdm(self.images):
            image.generate_plot(full_path, save_img, pdf, self.SAM_mode)
            if len(image.get_mask()) > 1:
                multi_mask_images.append(image)
        pdf.close()
        print('Saved ' + full_path + 'results.pdf to file.')
        self.write_stats(full_path + 'stats.txt', multi_mask_images)
        print('Saved ' + full_path + 'stats.txt to file.')







# Define image classes.
# Parent class: Cropped_Image
# Child classes: Region_Image, CSV_Image, grid image


# Define a Cropped_Image parent class.
class Cropped_Image():
    index = 0
    def __init__(self, cen, file_type, crop_size, image_shape, polygon=None) -> None:
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
        # The image data is generated at a later time via file path to the whole image
        self.image = None

        # Define the mask for the image
        self.mask = None

        # Define the SAM predictor scores and logits for the image
        self.predict_scores = None
        self.logits = None

        # Define the polygon annotation for the image
        self.polygon = polygon
    
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
    def get_mask_count(self):
        if self.mask is None:
            return 0
        else:
            return len(self.mask)
    def print_mask(self):
        if self.mask is None:
            return 'Not generated yet.'
        else:
            return pprint.pformat(self.mask)
    def get_predict_scores(self):
        return self.predict_scores
    def get_logits(self):
        return self.logits
    def get_polygon(self):
        return self.polygon
    def set_polygon(self, poly):
        self.polygon = poly
    def print_poly(self):
        if self.polygon is None:
            return 'Not generated yet.'
        else:
            return pprint.pformat(self.polygon)
    def __str__(self) -> str:
        s = '\nID: ' + str(self.id)
        s += '\nType: ' + self.type
        s += '\nCenter: ' + str(self.center)
        s += '\nBox: ' + str(self.box)
        s += '\nSize of image: ' + str(sys.getsizeof(self.image))
        s += '\nMask: \n' + str(self.mask)
        return s
    def set_image(self, img) -> None:
        self.image = img.copy()
    def clear_image(self) -> None:
        self.image = None
    def set_mask(self, m) -> None:
        self.mask = m
    def set_predict_scores(self, p):
        self.predict_scores = p
    def set_logits(self, l):
        self.logits = l
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
    def __init__(self, region, file_type, crop_size, image_shape, scale_factor, name) -> None:
        # Define region parameters
        self.region = region
        self.Xs = region[0].vertices.x
        self.Ys = region[0].vertices.y
        poly = list(zip(self.Xs, self.Ys))
        cen = get_center(poly)
        
        # Define the name
        self.name = name.upper()

        # Init super
        super().__init__(cen, file_type, crop_size, image_shape, polygon=poly)
        
        # Define the region box. This is the box the tightly bounds the region, not the boundary of the cropped image.
        # These are the original coordinates on the full image.
        self.region_box = self.get_region_box()

        # Define the scale factor for the image radius
        self.scale_factor = scale_factor

        # Regenerate boundary if necessary
        # For region images, the box is the region box by default (instead of the cropped image size parameter)
        if scale_factor == 1:
            self.box = self.region_box
        else:
            self.box = self.regenerate_boundary(image_shape, scale_factor)
        
        # Define the transformed polygon onto the cropped image
        self.transformed_polygon = self.get_polygon_transformation()

        # Define the transformed region box onto the cropped image
        self.transformed_region_box = self.get_transformed_region_box()
        
        # Define the background SAM input points to assist segmentation
        #self.background_points = self.get_background_points()
    
        # Define the brightness factor for this image. Default is 1
        self.brightness_factor = 1.0

    def get_name(self):
        return self.name
    def get_brightness_factor(self):
        return self.brightness_factor
    def set_brightness_factor(self, bf):
        self.brightness_factor = bf

    # Define a function to return the bounding box of the region in an optimal form to pass into SAM
    def get_region_box_for_SAM(self):
        x1,y1 = self.transformed_region_box['p1']
        x2,y2 = self.transformed_region_box['p2']
        return np.array([x1,y1,x2,y2])

    def get_image_center(self):
        return int(self.get_radius() * self.scale_factor)
    def __str__(self) -> str:
        s = '\n\nRegion file.'
        s += '\nID: ' + str(self.id)
        s += '\nType: ' + self.type
        s += '\nCenter: ' + str(self.center)
        s += '\nTransformed center: ' + str(self.get_image_center())
        s += '\nRadius: ' +  str(self.get_radius())
        s += '\nBox: ' + str(self.box)
        s += '\nRegion box: ' + str(self.region_box)
        s += '\nTransformed region box: ' + str(self.transformed_region_box)
        s += '\nSize of image: ' + str(sys.getsizeof(self.image))
        s += '\nImage: ' + str(self.print_image())
        s += '\nBrightness factor: ' + str(self.brightness_factor)
        s += '\nPolygon: ' + str(self.print_poly())
        s += '\nTransformed polygon: ' + str(pprint.pformat(self.transformed_polygon))
        s += '\nMask count: ' + str(self.get_mask_count())
        s += '\nMask: ' + str(self.print_mask())
        return s

    def get_radius(self):
        x,y = self.center
        x1,y1 = self.region_box['p1']
        return max(x-x1,y-y1)

    # Define a function to compute the polygon linear transformation
    # Returns the new polygon that can be overlayed onto the cropped image
    def get_polygon_transformation(self):
        transformed_poly = []
        x_poly_cen, y_poly_cen = self.center
        x_transform = x_poly_cen - self.get_image_center()
        y_transform = y_poly_cen - self.get_image_center()
        for point in self.polygon:
            xp,yp = point
            x_new = xp - x_transform
            y_new = yp - y_transform
            new_point = (x_new, y_new)
            transformed_poly.append(new_point)
        return transformed_poly

    # Define a function to compute the linear transformation of the region box
    def get_transformed_region_box(self):
        x_poly_cen, y_poly_cen = self.center
        x_transform = x_poly_cen - self.get_image_center()
        y_transform = y_poly_cen - self.get_image_center()
        x1, y1 = self.region_box['p1']
        x2, y2 = self.region_box['p2']
        b = {'p1':(int(x1-x_transform),int(y1-y_transform)), 'p2':(int(x2-x_transform),int(y2-y_transform))}
        return b

    def get_region_box(self):
        # Define a function to generate the bounding box of the region.
        left = float('inf')
        right = float('-inf')
        bot = float('inf')
        top = float('-inf')
        for point in self.polygon:
            x, y = point
            if x > right:
                right = x
            if x < left:
                left = x
            if y < bot:
                bot = y
            if y > top:
                top = y
        b = {'p1':(int(left),int(bot)), 'p2':(int(right),int(top))}
        return b
        
    def regenerate_boundary(self, full_size, scale_factor) -> None:
        # Define a function to regenerate the cropped boundary proportional to the scale factor and the radius of the region.
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
        radius = min(radius, 800)
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
    def generate_plot(self, path, save_img, pdf, SAM_mode):    
        dest_name = path + self.type + '_ID-' + str(self.get_id())
        
        # Determine if it is automatic SAM
        if SAM_mode == 'auto':
            plt.subplots(figsize=(14,7))
            chart_title = self.name + '   ' + self.type + '   Brightness: ' + str(self.brightness_factor) + '\nCenter: ' + str(self.center) + '   Radius: ' + str(self.get_radius()) + '   Mask Count: ' + str(len(self.mask))
            plt.suptitle(chart_title)

            plt.subplot(1,2,1)
            plt.imshow(self.image.copy())
            plt.scatter(*zip(*self.transformed_polygon))
            plt.axis('On')
            plt.title('Source')

            plt.subplot(1,2,2)
            plt.imshow(self.image)
            sorted_anns = sorted(self.mask.copy(), key=(lambda x: x['area']), reverse=True)
        
            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            for ann in sorted_anns:
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[m] = color_mask
            plt.imshow(img)

            plt.axis('On')
            plt.title('Segmented')
    
            sname = dest_name + '.png'
            if save_img:
                plt.savefig(sname, dpi='figure', bbox_inches='tight', pad_inches=0.1, facecolor='auto', edgecolor='auto')
            pdf.savefig()
            plt.close()
            
        else:
            # Is Predictor SAM
            for i, (mask, score) in enumerate(zip(self.mask, self.predict_scores)):
                plt.subplots(figsize=(14,7))
                chart_title = self.name + '   ' + self.type + '   Brightness: ' + str(self.brightness_factor) + '\nCenter: ' + str(self.center) + '   Radius: ' + str(self.get_radius()) + '   Score: '     + str(round(score, 3))
                plt.suptitle(chart_title)

                plt.subplot(1,2,1)
                plt.imshow(self.image.copy())
                plt.scatter(*zip(*self.transformed_polygon))
                plt.axis('On')
                plt.title('Source annotation')

                plt.subplot(1,2,2)
                plt.imshow(self.image)
                show_mask(mask, plt.gca())
                x = self.get_image_center()
                input_point = np.array([[x,x]])
                input_label = np.array([1])
                if SAM_mode == 'point':
                    show_points(input_point, input_label, plt.gca())
                elif SAM_mode == 'box':
                    show_box(self.get_region_box_for_SAM(), plt.gca())
                else:
                    show_points(input_point, input_label, plt.gca())
                    show_box(self.get_region_box_for_SAM(), plt.gca())
                plt.title('Segmentation ' + str(i))
                plt.axis('On')
                sname = dest_name + '_' + str(i) + '.png'
                if save_img:
                    plt.savefig(sname, dpi='figure', bbox_inches='tight', pad_inches=0.1, facecolor='auto', edgecolor='auto')
                pdf.savefig()
                plt.close()

        



# Create a CSV_Image child class.
class CSV_Image(Cropped_Image):
    def __init__(self, csv, file_type, crop_size, image_shape, scale_factor) -> None:
        # Init Super
        super().__init__((float(csv['X_center']),float(csv['Y_center'])), file_type, crop_size, image_shape)

        # Define the raw csv dictionary data
        # {'': '0', 'Name': 'MCELS-L1', 'X_center': '13668.075', 'Y_center': '7012.1650', 'Radius': '12.500000'}
        self.csv_data = csv
        
        # Define the scale factor for the image radius
        self.scale_factor = scale_factor

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
    def get_scale_factor(self):
        return self.scale_factor
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
        s += '\nPolygon: ' + str(self.print_poly())
        s += '\nMask count: ' + str(self.get_mask_count())
        s += '\nMask: ' + str(self.print_mask())
        return s

    def regenerate_boundary(self, full_size, scale_factor) -> None:
        # Define a function to regenerate the cropped boundary proportional to the scale factor and the radius of the region.
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
        radius = min(radius, 800)
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
    
    # Define a function to return the center point of the cropped image
    def get_image_center(self):
        return int(self.get_radius() * self.scale_factor)
    # Define a function to generate and save plots to the corresponding file paths
    def generate_plot(self, path, save_img, pdf):    
        dest_name = path + self.type + '_' + self.get_name() + '_ID-' + str(self.get_id())
        
        # Determine if it is automatic SAM
        if self.predict_scores is None:
            plt.subplots(figsize=(14,7))
            chart_title = self.get_name() + '  ' + self.type + '\nCenter: ' + str(self.center) + '   Radius: ' + str(self.get_radius()) + '   Mask Count: ' + str(len(self.mask))
            plt.suptitle(chart_title)

            plt.subplot(1,2,1)
            plt.imshow(self.image.copy())
            plt.axis('On')
            plt.title('Source')

            plt.subplot(1,2,2)
            plt.imshow(self.image)
            sorted_anns = sorted(self.mask.copy(), key=(lambda x: x['area']), reverse=True)
        
            #plt.set_autoscale_on(False)

            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            for ann in sorted_anns:
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[m] = color_mask
            plt.imshow(img)

            plt.axis('On')
            plt.title('Segmented')
    
            #plt.show()
            sname = dest_name + '.png'
            if save_img:
                plt.savefig(sname, dpi='figure', bbox_inches='tight', pad_inches=0.1, facecolor='auto', edgecolor='auto')
            pdf.savefig()
            plt.close()
            
        else:
            # Is Predictor SAM
            for i, (mask, score) in enumerate(zip(self.mask, self.predict_scores)):
                plt.figure(figsize=(10,10))
                plt.imshow(self.image)
                show_mask(mask, plt.gca())
                x = self.get_image_center()
                input_point = np.array([[x,x]])
                input_label = np.array([1])
                show_points(input_point, input_label, plt.gca())
                chart_title = self.get_name() + '  ' + self.type + '\nCenter: ' + str(self.center) + '   Radius: ' + str(self.get_radius()) + '   Score: '     + str(round(score, 3))
                #plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                plt.title(chart_title)
                plt.axis('On')
                #plt.show()
                sname = dest_name + '_' + str(i) + '.png'
                if save_img:
                    plt.savefig(sname, dpi='figure', bbox_inches='tight', pad_inches=0.1, facecolor='auto', edgecolor='auto')
                pdf.savefig()
                plt.close()

        



