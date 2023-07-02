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

def run_auto_SAM(sam, image_holder):
    print('Creating automatic mask generator...')
    mask_generator = SamAutomaticMaskGenerator(sam)
        #model=sam,
        #points_per_side=32,
        #pred_iou_thresh=0.9,
        #stability_score_thresh=0.85,
        #crop_n_layers=0,
        #crop_n_points_downscale_factor=2,
        #min_mask_region_area=100,  # Requires open-cv to run post-processing
    #)
    print('\nGenerating masks...')
    for cropped_image in tqdm(image_holder.get_images()):
        img = cropped_image.get_image()
        cropped_image.set_mask(mask_generator.generate(img))


def run_predict_SAM(sam, image_holder, SAM_mode):
    print('Creating SAM predictor...')
    predictor = SamPredictor(sam)

    print('\nGenerating masks...')
    for cropped_image in tqdm(image_holder.get_images()):
        img = np.array(cropped_image.get_image())
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        x = cropped_image.get_image_center()
        if cropped_image.get_type() == 'HII':
            label = 1
        else:
            label = 1
        input_point = np.array([[x,x]])
        input_label = np.array([label])
        
        if SAM_mode == 'box':
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=cropped_image.get_region_box_for_SAM(),
                multimask_output=False,
            )
        elif SAM_mode == 'point':
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
        else:
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=cropped_image.get_region_box_for_SAM(),
                multimask_output=False,
            )
        
        cropped_image.set_mask(masks)
        cropped_image.set_predict_scores(scores)
        cropped_image.set_logits(logits)

        predictor = SamPredictor(sam)



# Define main function
if __name__ == "__main__":

    # Define the mode. Determine to construct the images from region files, csv files or grid
    # Options are 'region', 'csv', 'grid', and 'all'
    mode = 'region'

    # Define cropped image size
    size = 180

    # Define full image shape
    image_shape = 16740

    # Define the radius scale factor.
    # If scale factor == 1, the default image size (defined above) will be used.
    scale_factor = 2

    # Define the image normalization technique. og or hist
    # 'og' will use my orginal normalization technique
    # 'hist' will use a histogram normalization technique
    normalization = 'og'

    # Define the brightness factors to be applied to the images
    brightness_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    brightness_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    #brightness_factors = [0.5, 1.0, 1.5]
    brightness_factors = [1.0]

    # Define the reduced dataset.
    # The dataset size will be reduced by the factor provided (data_reduction = 1 is the full data set).
    data_reduction = 500
    
    # Define SAM mode
    # 'auto' will use automatic SAM masking.
    # 'point' will use predictor SAM with input points
    # 'box' will use predictor SAM with input boxes
    # 'all' will use predictor SAM with input points and boxes.
    SAM_mode = 'box'

    # Define the image holder save name
    imageholder_save = 'imgholder_save.pkl'

    # Define if the program will run locally
    local = True

    # Define data paths (for cloud)
    paths = {
        'HII_folder_path': './drive/MyDrive/Astropy/LMC/HII_boundaries',
        'SNR_folder_path': './drive/MyDrive/Astropy/LMC/SNR_boundaries',
        'HII_csv_path': './astropy_sam/old_model/csv/hii_regions.csv',
        'SNR_csv_path': './astropy_sam/old_model/csv/snrs.csv',
        'image_path': './drive/MyDrive/Astropy/LMC/lmc_askap_aconf.fits',
        'save_plots_folder_path': './astropy_sam/cropped_imgs'
    }


    # Define data paths (for local  machine)
    if local:
        paths = {
            'HII_folder_path': './LMC/HII_boundaries',
            'SNR_folder_path': './LMC/SNR_boundaries',
            'HII_csv_path': './old_model/csv/hii_regions.csv',
            'SNR_csv_path': './old_model/csv/snrs.csv',
            'image_path': './LMC/lmc_askap_aconf.fits',
            'save_plots_folder_path': './cropped_imgs'
        }

    
    s = '\nParameters'
    s += '\nMode: ' + mode
    s += '\nCrop size: ' + str(size)
    s += '\nImage shape: ' + str(image_shape)
    s += '\nScale factor: ' + str(scale_factor)
    s += '\nData reduction: ' + str(data_reduction)
    s += '\nNormalization: ' + normalization
    s += '\nBrightness factors: ' + str(brightness_factors)
    s += '\nSAM mode: ' + SAM_mode
    print(s)

    # Check the save plot folder path. Remaining paths are checked in image holder constructor.
    if not os.path.exists(paths['save_plots_folder_path']):
        print('Error. Path does not exiskt: ' + paths['save_plots_folder_path'])
        exit(1)



    # Obtain the image holder
    print('Obtaining the image holder...')
    image_holder = Image_Holder(size, image_shape, paths, mode, scale_factor, data_reduction, normalization, SAM_mode, brightness_factors)

    # Generate the images in the image holder
    print('Generating the cropped images...')
    image_holder.generate_images()


    # Experiment with fits
    #hdul = fits.open(paths['image_path'])
    #hdul.info()
    #hdul.close()

    print(image_holder)



    # SAM -----------------------------------------------------------------------------------------------------------------------------

    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = 'default'
    print(s)
    print('\nCheckpoint: ' + sam_checkpoint + '\nDevice: ' + device + '\nModel type: ' + model_type + '\n\nCreating SAM...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    print('Sending SAM to ' + device)
    sam.to(device=device)
    
    if SAM_mode == 'auto':
        run_auto_SAM(sam, image_holder)
    else:
        run_predict_SAM(sam, image_holder, SAM_mode)


    print('Mask generation complete!')
    print(image_holder)

    
    print('Saving the image holder with the updated masks to ' + imageholder_save)
    with open(imageholder_save, 'wb') as f:
        pickle.dump(image_holder, f)


