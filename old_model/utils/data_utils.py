from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def read_dat(path):
    return np.fromfile(path)

def fits2matrix(path):
    return fits.getdata(path)


def show_grey_img(array, save_path=None):
    plt.imshow(array, cmap='gray')
    if save_path:
        plt.imsave(save_path, array, cmap='gray')
    plt.colorbar()
    plt.show()
