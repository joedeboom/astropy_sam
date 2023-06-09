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


from dataloader_sam import Image_Holder, Cropped_Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



if __name__ == "__main__":

    # Define cropped image size
    size = 180
    
    # Define full image shape
    image_shape = 16740
    
    # Define data paths
    HII_folder_path = './drive/MyDrive/Research/LMC/HII_boundaries'
    SNR_folder_path = './drive/MyDrive/Research/LMC/SNR_boundaries'
    image_path = './drive/MyDrive/Research/LMC/lmc_askap_aconf.fits'



    # Determine mode. 
    # If a save file is present, will load the image holder from the provided path
    # If not present, create a new one and save
    
    save_file_path = sys.argv[1]

    
    if os.path.exists(save_file_path):
        # load an old image holder from path
        print('Loading the image holder from ' + save_file_path)
        image_holder = pickle.load(open(save_file_path, 'rb'))
    else:
        # create a new image holder from scratch

        # Obtain the image holder
        print('Obtaining the image holder...')
        image_holder = Image_Holder(size, image_shape, HII_folder_path, SNR_folder_path, image_path)

        # Generate the images in the image holder
        print('Generating the cropped images...')
        image_holder.generate_images()

        # Save the image holder to file
        print('Saving the image holder to ' + save_file_path)
        with open(save_file_path, 'wb') as f:
            pickle.dump(image_holder, f)


    # Experiment with fits
    hdul = fits.open(image_path)
    hdul.info()

    hdul.close()



    # SAM ------------------------------------------------------------------------------------------------------
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    device = 'cuda'
    model_type = 'default'
    print('Creating SAM...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    print('Sending SAM to ' + device)
    sam.to(device=device)
    print('Creating automatic mask generator...')
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    print('\nLooping through the images...')
    for cropped_image in image_holder.get_images():
        print(cropped_image)
        img = np.array(cropped_image.get_image())

        print(img)
        print(img.shape)

        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        sam_result = mask_generator.generate(img)
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        show_anns(sam_result)
        plt.axis('off')
        plt.show()

        print('Yip yip yippie!')

        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        sv.plot_images_grid(
            images=[image_bgr, annotated_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image']
        )



