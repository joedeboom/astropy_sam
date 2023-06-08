from regions import Regions
import numpy as np
from utils.data_utils import fits2matrix
import cv2
import torch
from dataloader_reg import get_center, generate_annotation
# from dataloader_reg import get_center
# from dataloader_utils2 import generate_annotation
import glob
import os

img = fits2matrix('./LMC/lmc_askap_aconf.fits')
img = img[0][0]
img[np.isnan(img)] = -1
img = img.astype(np.float32)
img = torch.from_numpy(img)
img = torch.stack([img, img, img], 2)
img = np.array(img)

HII_folder_path = './reg/HII_boundaries'
SNR_folder_path = './reg/SNR_boundaries'
HII_reg_files = glob.glob(os.path.join(HII_folder_path, '*.reg'))
SNR_reg_files = glob.glob(os.path.join(SNR_folder_path, '*.reg'))

colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0)]
size = 180
image_shape = 16740
# img = np.zeros((16740, 16740, 3), dtype=np.uint8)
# ann = generate_annotation()
ann = np.zeros([image_shape, image_shape])
count = 0
for file in HII_reg_files:
    count += 1
    regions = Regions.read(file, format='ds9')
    Xs = regions[0].vertices.x
    Ys = regions[0].vertices.y
    length = len(Xs)
    polys = []
    for i in range(length):
        x = round(Xs[i])
        y = round(Ys[i])
    #     # polys.append([y, x])
        ann[y][x] = 1
    # polys = np.array(polys, dtype=np.int32)
    x, y = get_center(file)
    x, y = int(x), int(y)
    # points = np.array([[y, x], [y+10, x+10], [y-10,x-10]], dtype=np.int32)
    # cv2.fillPoly(img, [points], color=(255, 0, 0))
    b = img[y - 90: y + 90, x - 90:x + 90] * 255
    # print(points)
    temp = ann[y - 90: y + 90, x - 90:x + 90]
    seg_img = np.zeros((np.shape(b)[0], np.shape(b)[1], 3))
    for c in range(4):
        seg_img[:, :, 0] += ((temp[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((temp[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((temp[:, :] == c) * (colors[c][2])).astype('uint8')

    b = cv2.addWeighted(b.astype(np.float64), 0.5, seg_img, 0.5, 0)
    b *= 255.0
    cv2.imwrite('imgs/ratioANN/' + str(count) + '_test.jpg', b)
    # cv2.imwrite('imgs/ANN/' + str(count) + '_test.jpg', b)
    # cv2.imshow('1', b)
    # cv2.waitKey(0)