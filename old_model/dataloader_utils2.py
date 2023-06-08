from utils.data_utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import cv2
size = 180
image_shape = 16740
hii = pd.read_csv('./astropoly/csv/hii_regions.csv').values[:, 2:]
snrs = pd.read_csv('./astropoly/csv/snrs.csv').values[:-1, 2:]

train_snrs, test_snrs = [], []
train_hii, test_hii = [], []
count = 0
for i in range(len(snrs)):
    if count == 4:
        count = 0
        test_snrs.append(snrs[i])
    else:
        train_snrs.append(snrs[i])
        count += 1

count = 0
for i in range(len(hii)):
    if count == 4:
        test_hii.append(hii[i])
        count = 0
    else:
        train_hii.append(hii[i])
        count += 1


def generate_annotation():
    ann = np.zeros([image_shape, image_shape])
    for arr in hii:
        [x, y, radius] = arr[0], arr[1], arr[2]
        if x < 0 or y < 0:
            continue
        for i in range(int(y - radius), int(y + radius + 1)):
            for j in range(int(x - radius), int(x + radius + 1)):
                if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                    ann[i][j] = 1

    for arr in snrs:
        x, y, radius = arr[0], arr[1], arr[2]
        if x < 0 or y < 0:
            continue
        for i in range(int(y - radius), int(y + radius + 1)):
            for j in range(int(x - radius), int(x + radius + 1)):
                if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                    if ann[i][j] == 1:
                        ann[i][j] = 3
                    else:
                        ann[i][j] = 2
    return ann


def get_data_and_label(path, rgb=False):
    img_data = fits2matrix(path)[0][0]
    img_data[np.isnan(img_data)] = -1
    ann = generate_annotation()

    x_train, y_train, x_test, y_test = [], [], [], []

    for x, y, radius in train_snrs:
        x = int(x)
        y = int(y)
        x_train.append(img_data[y - size:y, x - size:x])
        x_train.append(img_data[y:y + size, x:x + size])
        x_train.append(img_data[y:y + size, x - size:x])
        x_train.append(img_data[y - size:y, x:x + size])
        x_train.append(img_data[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

        y_train.append(ann[y - size:y, x - size:x])
        y_train.append(ann[y:y + size, x:x + size])
        y_train.append(ann[y:y + size, x - size:x])
        y_train.append(ann[y - size:y, x:x + size])
        y_train.append(ann[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

    for x, y, radius in train_hii:
        x = int(x)
        y = int(y)
        x_train.append(img_data[y - size:y, x - size:x])
        x_train.append(img_data[y:y + size, x:x + size])
        x_train.append(img_data[y:y + size, x - size:x])
        x_train.append(img_data[y - size:y, x:x + size])
        x_train.append(img_data[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

        y_train.append(ann[y - size:y, x - size:x])
        y_train.append(ann[y:y + size, x:x + size])
        y_train.append(ann[y:y + size, x - size:x])
        y_train.append(ann[y - size:y, x:x + size])
        y_train.append(ann[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

    for x, y, radius in test_snrs:
        x = int(x)
        y = int(y)
        x_test.append(img_data[y - size:y, x - size:x])
        x_test.append(img_data[y:y + size, x:x + size])
        x_test.append(img_data[y:y + size, x - size:x])
        x_test.append(img_data[y - size:y, x:x + size])
        x_test.append(img_data[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

        y_test.append(ann[y - size:y, x - size:x])
        y_test.append(ann[y:y + size, x:x + size])
        y_test.append(ann[y:y + size, x - size:x])
        y_test.append(ann[y - size:y, x:x + size])
        y_test.append(ann[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

    for x, y, radius in test_hii:
        x = int(x)
        y = int(y)
        x_test.append(img_data[y - size:y, x - size:x])
        x_test.append(img_data[y:y + size, x:x + size])
        x_test.append(img_data[y:y + size, x - size:x])
        x_test.append(img_data[y - size:y, x:x + size])
        x_test.append(img_data[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

        y_test.append(ann[y - size:y, x - size:x])
        y_test.append(ann[y:y + size, x:x + size])
        y_test.append(ann[y:y + size, x - size:x])
        y_test.append(ann[y - size:y, x:x + size])
        y_test.append(ann[y - int(size / 2):y + int(size / 2), x - int(size / 2):x + int(size / 2)])

    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def get_set(path):
    x_train, y_train, x_test, y_test = get_data_and_label(path)
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
    x_train = torch.unsqueeze(x_train, 1)
    x_test = torch.unsqueeze(x_test, 1)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    return train_dataset, test_dataset


def get_dataloader(path, batch_size=8, shuffle=True):
    train_dataset, test_dataset = get_set(path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return train_loader, test_loader


# train, test = get_dataloader('./LMC/lmc_askap_aconf.fits')
# for x, y in test:
#     print(x.size())
#     for i in range(8):
#         img = torch.squeeze(x[i], 0)
#         img = torch.stack([img, img, img], 2).numpy()
#         print(img.shape)
#         cv2.imshow('1', img * 255)
#         cv2.waitKey(0)
#     break
