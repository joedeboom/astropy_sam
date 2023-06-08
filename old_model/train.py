# GIT TESTING #
print('yip, yip, yippie!')

# Import necessary libraries and modules
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.pspnet import PSPNet  # Import PSPNet model
from models.pspnet_training import weights_init  # Import function to initialize model weights
from utils.callbacks import LossHistory  # Import LossHistory for tracking training loss history
from utils.dataloader import PSPnetDataset, pspnet_dataset_collate  # Import dataset and collate function for PSPNet
from utils.utils_fit import fit_one_epoch, fit_one_epoch_rgb  # Import function to fit the model for one epoch
import dataloader_utils2  # Import additional dataloader utilities
import dataloader_reg  # Import region dataloader utilities

import os

# Main function
if __name__ == "__main__":
    # Check if CUDA is available and set variable
    if torch.cuda.is_available():
        print('cuda available')
        Cuda = True
    else:
        Cuda = False

    # Define model parameters
    image_path = './drive/MyDrive/Astropy/LMC/lmc_askap_aconf.fits'
    if os.path.exists(image_path):
        print('Image path exists!')
    else:
        raise ValueError('Image path does not exist')    

    num_classes = 4  # Number of classes for the task
    backbone = "resnet50"  # Backbone network to use in the PSPNet

    # Define whether to use pretrained weights
    pretrained = False

    # Define model path
    model_path = ''

    # Define the downsample factor for the PSPNet
    downsample_factor = 8

    # Define the input shape for the model
    input_shape = [180, 180]

    # Define the initial epoch number
    Init_Epoch = 0
    # Define the number of epochs for which the model is frozen
    Freeze_Epoch = 0
    # Define the batch size while the model is frozen
    Freeze_batch_size = 0
    # Define the learning rate while the model is frozen
    Freeze_lr = 0

    # Define the number of epochs after which the model is unfrozen
    UnFreeze_Epoch = 100
    # Define the batch size after the model is unfrozen
    Unfreeze_batch_size = 8
    # Define the learning rate after the model is unfrozen
    Unfreeze_lr = 5e-6

    # Define whether to use Dice loss
    dice_loss = False

    # Define whether to use Focal loss
    focal_loss = True

    # Define the class weights
    cls_weights = np.ones([num_classes], np.float32)

    # Define whether to use an auxiliary branch in the model
    aux_branch = True

    # Define whether to freeze the model during training
    Freeze_Train = False

    # Define the number of workers for data loading
    num_workers = 2

    # Create the PSPNet model
    #print('before model')
    model = PSPNet(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                   pretrained=pretrained, aux_branch=aux_branch)
    print('model created')

    # Initialize the model weights if not using pretrained weights
    if not pretrained:
        weights_init(model)
    # Load the pretrained weights if model path is given
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()


    loss_history = LossHistory("logs/")

    # If True, start unfreezing the model for training
    if True:
        # Set the batch size and learning rate for the unfrozen model
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        # Set the number of epochs to train the unfrozen model
        end_epoch = UnFreeze_Epoch

        # Define the optimizer as Adam and set its learning rate
        optimizer = optim.Adam(model_train.parameters(), lr)
        # Define the learning rate scheduler
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
        # Get the dataloader for the training and validation data
        print('retreiving dataloader...')
        gen, gen_val = dataloader_reg.get_dataloader(image_path)
        # Calculate the number of steps per epoch for the training and validation data
        epoch_step = len(gen)
        epoch_step_val = len(gen_val)
        # If there is no data, raise an error
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The data set is too small for training. Please expand the data set.")

        # For each epoch in the range of defined epochs
        for epoch in range(0, end_epoch):
            # Fit the model for one epoch and update the learning rate
            fit_one_epoch_rgb(model_train, model, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights,
                          aux_branch, num_classes)
            lr_scheduler.step()
