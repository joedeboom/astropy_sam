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
    
    save_plots_folder_path = './astropy_sam/cropped_imgs'

    # Obtain the image holder
    save_file_path = './imgholder_save'
    print('Loading the image holder from ' + save_file_path + '...',end='\t')
    image_holder = pickle.load(open(save_file_path, 'rb'))
    if image_holder is not None:
        print('Success!')
    else:
        print('Error loading the image holder.')
    
    # Save the plots to folder
    print('Generating plots and saving to ' + save_plots_folder_path + '...')
    image_holder.save_plots(save_plots_folder_path)


