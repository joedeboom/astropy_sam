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

import dataloader_sam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



if __name__ == "__main__":

    # Define cropped image size
    size = 180

    # Define full image shape
    image_shape = 16740

    # Define save file name
    save_file_name = 'crop_image_data.obj'

    # Define data paths
    HII_folder_path = './drive/MyDrive/Research/LMC/HII_boundaries'
    SNR_folder_path = './drive/MyDrive/Research/LMC/SNR_boundaries'
    image_path = './drive/MyDrive/Research/LMC/lmc_askap_aconf.fits'
    
    # Obtain the image holder
    image_holder = dataloader_sam.extract_images(size, image_shape, save_file_name, HII_folder_path, SNR_folder_path, image_path)

    

    # SAM -------------------------------------------------
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    device = 'cuda'
    model_type = 'default'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    img = image_holder[0].get_image()
    print('shape:')
    print(str(img.shape))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sam_result = mask_generator.generate(image_rgb)
    plt.figure(figsize=(20,20))
    plt.imshow(image_rgb)
    show_anns(sam_result)
    plt.axis('off')
    plt.show()

