'''
21.06.2023, I. Schicker

The UNET models for downscaling of ERA5 2m temperature to CERRA for the Code4Earth Challenge

History:
- 21.06.2023, I. Schicker: split the test code into separate function files. 

'''
import os
import numpy as np
import pandas as pd
import xarray as xr

# import tensorflow and required stuff from Keras API
import tensorflow as tf
# all the layers used for U-net
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Input, MaxPool2D)
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# Set the seed for reproducibility
seed = 123
##check which tensorflow version
if tf.__version__ == '1.12.0':
    tf.random.set_random_seed(seed)
else:
    tf.random.set_seed(seed)



##################################################UNET-LARGE, NOT YET TESTED!###############################
def unet_model_largedomain(input_shape,z_branch):
    ''' 
    19.06.2023, I. Schicker
    
    This Unet was built for the Code4Earth Challenge for the original grid of ERA52CERRA of 801 x 501 grid points.
    Will be adapted for the smaller grid.
    Also, not yet tested.
    '''
    inputs = tf.keras.Input(shape=input_shape)
 
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_size = (3,3),kernel_initializer="he_normal")(inputs)
    conv1 = layers.BatchNormalization()(conv1)  # Batch normalization
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
 
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool1)
    conv2 = layers.BatchNormalization()(conv2)  # Batch normalization
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
 
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool2)
    conv3 = layers.BatchNormalization()(conv3)  # Batch normalization
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
 
    # Bridge
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool3)
    conv4 = layers.BatchNormalization()(conv4)  # Batch normalization
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv4)
 
    # Decoder
    up5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer="he_normal")(conv4)
    crop5 = layers.Cropping2D(cropping=((0, 0), (0, 1)))(conv3)  # Crop to match shape
    up5 = layers.concatenate([up5, crop5], axis=3)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(up5)
    convconv54 = layers.BatchNormalization()(conv5)  # Batch normalization
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv5)
 
    up6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer="he_normal")(conv5)
    crop6 = layers.Cropping2D(cropping=((0, 0), (0, 2)))(conv2)  # Crop to match shape
    up6 = layers.concatenate([up6, crop6], axis=3)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(up6)
    conv6 = layers.BatchNormalization()(conv6)  # Batch normalization
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv6)
 
    up7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer="he_normal")(conv6)
    crop7 = layers.Cropping2D(cropping=((0, 1), (0, 5)))(conv1)  # Crop to match shape
    up7 = layers.concatenate([up7, crop7], axis=3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(up7)
    conv7 = layers.BatchNormalization()(conv7)  # Batch normalization
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv7)
 
    # Output layer
    ##OLD: outputs = layers.Conv2D(2, 1, activation='sigmoid', kernel_initializer="he_normal")(conv9)
    output_temp = Conv2D(1, (1,1), kernel_initializer="he_normal", name="output_temp")(conv7)
    if z_branch:
        output_z = Conv2D(1, (1, 1), kernel_initializer="he_normal", name="output_z")(conv7)

        model = Model(inputs, [output_temp, output_z], name="t2m_downscaling_unet_large_with_z")
    else:    
        model = Model(inputs, output_temp, name="t2m_downscaling_unet_large")

    ## OLD: model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

##################################################UNET-SMALL, NEEDS OPTIMISATION!###############################
def unet_model_small(input_shape,z_branch):
    ''' 
    20.06.2023, I. Schicker
    
    This Unet was built for the Code4Earth Challenge for the cropped domain agreed on with the spanish team of 240 x 160 grid points.

    Currently tested. 
    
    To be added:
      - Batch normalization
      - kernel_initializer
      - strides_up
      - kernel
      - activation
      
    '''
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(inputs)
    conv1 = layers.BatchNormalization()(conv1)  # Batch normalization
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool1)
    conv2 = layers.BatchNormalization()(conv2)  # Batch normalization
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool2)
    conv3 = layers.BatchNormalization()(conv3)  # Batch normalization
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool3)
    conv4 = layers.BatchNormalization()(conv4)  # Batch normalization
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv4)

    # Decoder
    up5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer="he_normal")(conv4)
    crop5 = layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv3)  # Crop to match shape
    up5 = layers.concatenate([up5, crop5], axis=3)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(up5)
    conv5 = layers.BatchNormalization()(conv5)  # Batch normalization
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv5)

    up6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer="he_normal")(conv5)
    crop6 = layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv2)  # Crop to match shape
    up6 = layers.concatenate([up6, crop6], axis=3)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(up6)
    conv6 = layers.BatchNormalization()(conv6)  # Batch normalization
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv6)

    up7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer="he_normal")(conv6)
    crop7 = layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv1)  # Crop to match shape
    up7 = layers.concatenate([up7, crop7], axis=3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(up7)
    conv7 = layers.BatchNormalization()(conv7)  # Batch normalization
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv7)

    # Output layer
    ##OLD: outputs = layers.Conv2D(2, 1, activation='sigmoid', kernel_initializer="he_normal")(conv9)
    output_temp = Conv2D(1, (1,1), kernel_initializer="he_normal", name="output_temp")(conv7)
    if z_branch:
        output_z = Conv2D(1, (1, 1), kernel_initializer="he_normal", name="output_z")(conv7)

        model = Model(inputs, [output_temp, output_z], name="t2m_downscaling_unet_small_with_z")
    else:    
        model = Model(inputs, output_temp, name="t2m_downscaling_unet_small")

    ## OLD: model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#############################SMALL DOMAIN INCLUDING WINDOW APPROACH####################################################
def unet_model_small_window(input_shape,z_branch):
    inputs = tf.keras.Input(shape=input_shape)
 
    # Contracting path
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(inputs)
    conv1 = layers.BatchNormalization()(conv1)  # Batch normalization
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
 
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool1)
    conv2 = layers.BatchNormalization()(conv2)  # Batch normalization
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
 
    # Add padding to ensure compatibility
    pool2_pad = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(pool2)
   
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool2_pad)
    conv3 = layers.BatchNormalization()(conv3)  # Batch normalization
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
 
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool3)
    conv4 = layers.BatchNormalization()(conv4)  # Batch normalization
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
 
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer="he_normal")(pool4)
    conv5 = layers.BatchNormalization()(conv5)  # Batch normalization
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
 
    # Expanding path
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer="he_normal")(layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_normal")(merge6)
    conv6 = layers.BatchNormalization()(conv6)  # Batch normalization
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv6)
 
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer="he_normal")(layers.UpSampling2D(size=(2, 2))(conv6))
    crop7 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(conv3)  # Crop to match shape
    merge7 = layers.concatenate([crop7, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(merge7)
    conv7 = layers.BatchNormalization()(conv7)  # Batch normalization
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv7)
 
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer="he_normal")(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(merge8)
    conv8 = layers.BatchNormalization()(conv8)  # Batch normalization
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv8)
 
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer="he_normal")(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(merge9)
    conv9 = layers.BatchNormalization()(conv9)  # Batch normalization
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(conv9)
   
    # Output layer
    ##OLD: outputs = layers.Conv2D(2, 1, activation='sigmoid', kernel_initializer="he_normal")(conv9)
    output_temp = Conv2D(1, (1,1), kernel_initializer="he_normal", name="output_temp")(conv9)
    if z_branch:
        output_z = Conv2D(1, (1, 1), kernel_initializer="he_normal", name="output_z")(conv9)

        model = Model(inputs, [output_temp, output_z], name="t2m_downscaling_unet_small_window_with_z")
    else:    
        model = Model(inputs, output_temp, name="t2m_downscaling_unet_small_window")

    ## OLD: model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#################################ECMWF May 2022 Course UNET - Adapated by I. Schicker within the course#################
'''
Based on the maelstrom notebook, this was used in May 2022 for the ECMWF training course. Has already been
adapted during the course by I. Schicker in terms of layers, kernel_init, etc.
Used here, too, as a first shot.
'''
##############################Conv block########################################################
def conv_block(inputs, num_filters: int, kernel: tuple = (3,3), padding: str = "same",
               activation: str = "tanh", kernel_init: str = "he_normal", l_batch_normalization: bool = True):
    """
    A convolutional layer with optional batch normalization
    :param inputs: the input data with dimensions nx, ny and nc
    :param num_filters: number of filters (output channel dimension)
    :param kernel: tuple indictating kernel size
    :param padding: technique for padding (e.g. "same" or "valid")
    :param activation: activation fuction for neurons (e.g. "relu")
    :param kernel_init: initialization technique (e.g. "he_normal" or "glorot_uniform")
    """
    
    x = Conv2D(num_filters, kernel, padding=padding, kernel_initializer=kernel_init)(inputs)
    if l_batch_normalization:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    return x

def conv_block_n(inputs, num_filters, n=2, kernel=(3,3), padding="same", activation="relu", 
                     kernel_init="he_normal", l_batch_normalization=True):
    """
    Sequential application of two convolutional layers (using conv_block).
    """
    
    x = conv_block(inputs, num_filters, kernel, padding, activation,
                   kernel_init, l_batch_normalization)
    for i in np.arange(n-1):
        x = conv_block(x, num_filters, kernel, padding, activation,
                       kernel_init, l_batch_normalization)
    
    return x

def encoder_block(inputs, num_filters, kernel_maxpool: tuple=(2,2), l_large: bool=True):
    """
    One complete encoder-block used in U-net
    this is formed of 1 or 2 convolutional layers
    followed by a MaxPooling to aggregate spatial info
    """
    if l_large:
        x = conv_block_n(inputs, num_filters, n=2)
    else:
        x = conv_block(inputs, num_filters)
        
    p = MaxPool2D(kernel_maxpool)(x)
    
    return x, p

def decoder_block(inputs, skip_features, num_filters, kernel: tuple=(3,3), strides_up: int=2, padding: str= "same", 
                  activation="relu", kernel_init="he_normal", l_batch_normalization: bool=True):
    """
    One complete decoder block used in U-net (reverting the encoder)
    Conv2DTranspose fulfils the opposite role to MaxPool2D, increasing the resolution
    followed by concatenating the features from the down-part of the UNet
    finally more convolutions!
    """
    
    x = Conv2DTranspose(num_filters, (strides_up, strides_up), strides=strides_up, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block_n(x, num_filters, 2, kernel, padding, activation, kernel_init, l_batch_normalization)
    
    return x


def build_unet_sha(input_shape, channels_start=56, z_branch=False):
    """
    Build a u-net for a prescribed input shape, channels & w/wo orography
    input_shape[list] -- shape of input tensor
    channels_start[int] -- number of channels in first convolution layer
    z_branch[logical] -- True: build model to also recover orography field
    """
    inputs = Input(input_shape)
    
    """ encoder """
    s1, e1 = encoder_block(inputs, channels_start, l_large=True)
    s2, e2 = encoder_block(e1, channels_start*2, l_large=False)
    s3, e3 = encoder_block(e2, channels_start*4, l_large=False)
    s4, e4 = encoder_block(e3, channels_start*8, l_large=False)

    """ bridge encoder <-> decoder """
    b1 = conv_block(e4, channels_start*16)
    
    """ decoder """
    d1 = decoder_block(b1, s4, channels_start*8)
    d2 = decoder_block(d1, s3, channels_start*4)
    d3 = decoder_block(d2, s2, channels_start*2)
    d4 = decoder_block(d3, s1, channels_start)
    
    output_temp = Conv2D(1, (1,1), kernel_initializer="he_normal", name="output_temp")(d4)
    if z_branch:
        output_z = Conv2D(1, (1, 1), kernel_initializer="he_normal", name="output_z")(d4)

        model = Model(inputs, [output_temp, output_z], name="t2m_downscaling_unet_with_z")
    else:    
        model = Model(inputs, output_temp, name="t2m_downscaling_unet")
    
    return model

