'''
21.06.2023, I. Schicker

Utils for downscaling of ERA5 2m temperature to CERRA using a UNET for the Code4Earth Challenge


History:
- 23.06.2023, I. Schicker: adding a separate loss function file for custom losses
'''

import os
#from downscaling_utils import create_plots
import numpy as np
import pandas as pd
import xarray as xr

# import tensorflow for seed etc.
import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D)
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.losses import Loss

# Set the seed for reproducibility
seed = 123
##check which tensorflow version
if tf.__version__ == '1.12.0':
    tf.random.set_random_seed(seed)
else:
    tf.random.set_seed(seed)

###########################################WEIGHTED LOSS###############################################
def weighted_loss(preds, targets, weights, loss='l1'):
    """
    weighted loss function
    
    preds:        Tensor/array of size (batch_size, frames_predict, 64, 64) holding model predictions 
    targets:      Tensor/array of size (batch_size, frames_predict, 64, 64) holding targets 
    weights:      Tensor/array of size (batch_size, frames_predict, 64, 64) holding weights as computed by the data loader
    loss:         'l1' for MAE loss and 'l2' for MSE loss
    """            
    
    if loss=='l1':
        return (weights *abs(preds - targets)).mean()
    elif loss=='l2': 
        return (weights *(preds - targets)**2).mean()
    else: 
        raise Exception('loss must be either \'l1\' or \'l2\'!')

###########################################SERA LOSS###############################################        
def sera_loss(preds, targets, relevances):
    """
    SERA loss function
    
    preds:        Tensor of size (batch_size, frames_predict, 64, 64) holding model predictions 
    targets:      Tensor of size (batch_size, frames_predict, 64, 64) holding targets 
    relevances:   Tensor of size (batch_size, frames_predict, 64, 64) holding relevance values as computed by the data loader
    """
    dt = 0.1
    sera = 0.
    for t in np.arange(0., 1.+dt, dt): 
        indices = relevances >= t 
        if indices.sum() > 0: 
            ser_t = ((preds[indices]-targets[indices])**2).mean()
            sera += ser_t*dt 
    return sera

#############################################Inverse weighted#####################################
# Define custom loss function
class InverselyWeightedMSE(Loss):
    def __init__(self):
        super(InverselyWeightedMSE, self).__init__()
 
    def call(self, y_true, y_pred):
        weights = 1.0 / tf.abs(y_true)  # Calculate the weights inversely proportional to y_true
        mse = tf.reduce_mean(tf.square(y_true - y_pred))  # Compute mean squared error
        weighted_mse = tf.reduce_mean(tf.multiply(tf.square(y_true - y_pred), weights))  # Apply weights to MSE
 
        return weighted_mse

