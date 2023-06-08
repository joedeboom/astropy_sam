import copy

import cv2
from models.pspnet import PSPNet
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from utils.data_utils import fits2matrix
import glob
import os
from dataloader_reg import get_center

image_path = './drive/MyDrive/Astropy/LMC/lmc_askap_aconf.fits'
HII_folder_path = './drive/MyDrive/Astropy/LMC/HII_boundaries'
SNR_folder_path = './drive/MyDrive/Astropy/LMC/SNR_boundaries'
HII_reg_files = glob.glob(os.path.join(HII_folder_path, '*.reg'))
SNR_reg_files = glob.glob(os.path.join(SNR_folder_path, '*.reg'))
#state2load = 'new_old_ep066-loss0.212-val_loss0.240.pth'                                   xuanhan og
state2load = '/content/logs/loss_2023_06_02_17_43_36/ep052-loss0.211-val_loss0.235.pth'     #mine


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

remove_skycoord_regions()

colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0)]

img_data2 = fits2matrix(image_path)
print(img_data2.shape)
img_data2 = img_data2[0][0]
img_data2[np.isnan(img_data2)] = -1
pspnet = PSPNet(4, 8)
df2 = pd.read_csv('./astropoly/csv/snrs.csv')

pspnet.load_state_dict(torch.load(state2load, map_location=torch.device('cpu')))
pspnet.eval()
img = copy.deepcopy(img_data2).astype(np.float32)
img = torch.from_numpy(img)
img = torch.stack([img, img, img], 2)
img = np.array(img)
count = 0
# for arr in df2.values[:, 2:]:
for file in HII_reg_files:
    # print(arr)
    x, y = get_center(file)
    count += 1
    # x, y = int(arr[0]), int(arr[1])
    x, y = int(x), int(y)
    # radius = int(arr[2])
    # cv2.circle(img_data2, (x, y), radius, color=(0, 0, 255), thickness=1)
    cur = img_data2[y - 90: y + 90, x - 90:x + 90].astype(np.float32)
    # img = cv2.circle(img, (x, y), radius, (0, 0, 255), thickness=1)
    b = img[y - 90: y + 90, x - 90:x + 90] * 255
    # b = img[y - 9 0: y + 90, x - 90:x + 90]

    a = torch.from_numpy(cur)
    print(a.size())
    # a = torch.unsqueeze(a, 0)
    a = torch.stack([a, a, a], 0)

    a = torch.unsqueeze(a, 0)
    output = pspnet(a)[1][0]
    output = F.softmax(output.permute(1, 2, 0), dim=-1).cpu().detach().numpy()
    output = output.argmax(axis=-1)

    seg_img = np.zeros((np.shape(output)[0], np.shape(output)[1], 3))
    for c in range(4):
        seg_img[:, :, 0] += ((output[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((output[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((output[:, :] == c) * (colors[c][2])).astype('uint8')
    print(output.shape)
    b = cv2.addWeighted(b.astype(np.float64), 0.2, seg_img, 0.8, 0)
    b *= 255.0
    cv2.imwrite('imgs/ratio/' + str(count) + '_test.jpg', b)
    # cv2.imshow('1', b)
    # cv2.waitKey(0)
