from utils.data_utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

size = 180
image_shape = 16740


def generate_annotation():
    hii = pd.read_csv('csv/hii_regions.csv').values
    snrs = pd.read_csv('csv/snrs.csv').values
    ann = np.zeros([image_shape, image_shape])
    for arr in hii[:, 2:]:
        x, y, radius = arr[0], arr[1], arr[2]
        if x < 0 or y < 0:
            continue
        for i in range(int(y - radius), int(y + radius + 1)):
            for j in range(int(x - radius), int(x + radius + 1)):
                if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                    ann[i][j] = 1

    for arr in snrs[:, 2:]:
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

    count = 0
    cover = 3
    for i in range(0, 16740 - size, int(size / 3)):
        for j in range(0, 16740 - size, int(size / 3)):
            if count == 12:

                x_test.append(img_data[i:i + size, j:j + size])
                cover -= 1
                if cover == 0:
                    count = 0
                    cover = 3
            else:
                x_train.append(img_data[i:i + size, j:j + size])
                count += 1
    count = 0
    cover = 3

    for i in range(0, 16740 - size, int(size / 3)):
        for j in range(0, 16740 - size, int(size / 3)):
            if count == 12:
                y_test.append(ann[i:i + size, j:j + size])
                cover -= 1
                if cover == 0:
                    count = 0
                    cover = 3
            else:
                y_train.append(ann[i:i + size, j:j + size])
                count += 1
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    print(x_train.shape)
    return x_train, y_train, x_test, y_test


def get_set(path, rgb=False):
    x_train, y_train, x_test, y_test = get_data_and_label(path, rgb)
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    return train_dataset, test_dataset


def get_dataloader(path, batch_size=16, shuffle=True, rgb=False):
    train_dataset, test_dataset = get_set(path, rgb)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return train_loader, test_loader


# train, test = get_dataloader('./LMC/lmc_askap.fits', rgb=True)
#
# for x, y in train:
#     x = torch.stack([x, x, x], 1)
#     print(x.size())
#     print(y.size())
#     break
# print(len(train))
# print(len(test))
