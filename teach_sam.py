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



# Define main function
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
    
    print('\nGenerating masks...')
    for cropped_image in tqdm(image_holder.get_images()):
        #print(cropped_image)
        img = np.array(cropped_image.get_image())
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_image.set_mask(mask_generator.generate(img))
        
        """
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        show_anns(sam_result)
        plt.axis('off')
        plt.show()

        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
        sv.plot_images_grid(
            images=[img, annotated_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image']
        )
        """

    # Save the image holder (with updated masks) to file
    new_save_name = save_file_path + '_with_masks'
    print('Saving the image holder with the updated masks to ' + new_save_name)
    with open(new_save_name, 'wb') as f:
        pickle.dump(image_holder, f)


