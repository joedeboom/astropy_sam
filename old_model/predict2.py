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
import dataloader_reg


HII_folder_path = './reg/HII_boundaries'
SNR_folder_path = './reg/SNR_boundaries'
HII_reg_files = glob.glob(os.path.join(HII_folder_path, '*.reg'))
SNR_reg_files = glob.glob(os.path.join(SNR_folder_path, '*.reg'))

colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0)]

img_data2 = fits2matrix('./LMC/lmc_askap_aconf.fits')
print(img_data2.shape)
img_data2 = img_data2[0][0]
img_data2[np.isnan(img_data2)] = -1
pspnet = PSPNet(4, 8)

pspnet.load_state_dict(torch.load('new_old_ep066-loss0.212-val_loss0.240.pth', map_location=torch.device('cpu')))
pspnet.eval()
img = copy.deepcopy(img_data2).astype(np.float32)
img = torch.from_numpy(img)
img = torch.stack([img, img, img], 2)
img = np.array(img)
count = 0



for file in HII_reg_files:
    x, y = get_center(file)
    count += 1
    x, y = int(x), int(y)
    cur = img_data2[y - 90: y + 90, x - 90:x + 90].astype(np.float32)
    b = img[y - 90: y + 90, x - 90:x + 90] * 255
    a = torch.from_numpy(cur)
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
