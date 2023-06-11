import sys
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
import supervision as sv
from tqdm import tqdm

from dataloader_sam import Image_Holder, Cropped_Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



# Define main function
if __name__ == "__main__":

    # Define cropped image size
    size = 180
    
    # Define full image shape
    image_shape = 16740
    
    # Define data paths
    HII_folder_path = './drive/MyDrive/Research/LMC/HII_boundaries'
    SNR_folder_path = './drive/MyDrive/Research/LMC/SNR_boundaries'
    HII_csv_path = './astropy_sam/old_model/csv/hii_regions.csv'
    SNR_csv_path = './astropy_sam/old_model/csv/snrs.csv'
    image_path = './drive/MyDrive/Research/LMC/lmc_askap_aconf.fits'
    save_plots_folder_path = './astropy_sam/cropped_imgs'
    
    # Check the save plot folder path
    if not os.path.exists(save_plots_folder_path):
        print('Error. Path does not exist: ' + save_plots_folder_path)
        exit(1)


    # Define the mode. Determine to construct the images from region files, csv files or grid.
    mode = 'csv'

    # Obtain the image holder
    print('Obtaining the image holder...')
    image_holder = Image_Holder(size, image_shape, HII_folder_path, SNR_folder_path, HII_csv_path, SNR_csv_path, image_path, mode)

    # Generate the images in the image holder
    print('Generating the cropped images...')
    image_holder.generate_images()


    # Experiment with fits
    hdul = fits.open(image_path)
    hdul.info()

    hdul.close()

    print(image_holder)


    # SAM ------------------------------------------------------------------------------------------------------
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = 'default'
    print('Creating SAM...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    print('Sending SAM to ' + device)
    sam.to(device=device)
    print('Creating automatic mask generator...')
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    print('\nGenerating masks...')
    for cropped_image in tqdm(image_holder.get_images()):
        img = np.array(cropped_image.get_image())
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_image.set_mask(mask_generator.generate(img))
    print('Mask generation complete!')
    print(image_holder)

    
    # Define the image holder save name
    new_save_name = 'imgholder_save'
    print('Saving the image holder with the updated masks to ' + new_save_name)
    with open(new_save_name, 'wb') as f:
        pickle.dump(image_holder, f)


