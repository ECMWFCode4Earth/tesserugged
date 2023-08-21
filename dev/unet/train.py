'''
21.06.2023, I. Schicker

Downscaling of ERA5 2m temperature to CERRA using a UNET for the Code4Earth Challenge - Training

Training Code

History:
- 21.06.2023, I. Schicker: splitting the test skript into a more readable code
'''

import os
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

# Set the seed for reproducibility
seed = 123
##check which tensorflow version
if tf.__version__ == '1.12.0':
    tf.random.set_random_seed(seed)
else:
    tf.random.set_seed(seed)


##########import the model components
import unets 
import generator
import utils as ut


def train_unet(train_feature,train_target,valid_feature, valid_target,lowtfile, hightfile,lowlsmfile,highlsmfile,cropbox, unetname, 
               batch_size, num_epochs, loss_function, optimizer, modelpath, z_branch):
    '''
    21.06.2023, I. Schicker
    
    This is the main training part of the Unets
    '''

    ##loading topography data
    lowtopo = ut.topography_lsm(lowtfile, cropbox, crop=True)
    hightopo = ut.topography_lsm(hightfile, cropbox, crop=True)
    lowlsm = ut.topography_lsm(lowlsmfile, cropbox, crop=True)
    highlsm = ut.topography_lsm(highlsmfile, cropbox, crop=True)
    

    ## Define train and validation steps
    print(type(train_feature))
    
    if type(train_feature) != list:
        total_train_samples = train_feature.sizes['time'] #len(trainperiod)  # Total number of training samples
        total_validation_samples = valid_feature.sizes['time'] #len(validperiod)  # Total number of validation samples
        train_steps = int(np.floor(total_train_samples // batch_size))
        validation_steps = int(np.floor(total_validation_samples // batch_size))
    else:
        total_train_samples = len(train_feature) #len(trainperiod)  # Total number of training samples
        total_validation_samples = len(valid_feature) #len(validperiod)  # Total number of validation samples
        train_steps = int(np.floor(total_train_samples // batch_size))
        validation_steps = int(np.floor(total_validation_samples // batch_size))

    print(total_train_samples, total_validation_samples, train_steps,train_steps)


    ## Define input shape, hardcoded here!
    input_shape = (160, 240, 5) #--> when using
    input_shape_small = (64, 64, 5)
    
    ##Set windowing for the subpatches approach.
    ##MIND: unet_model_small_window expects the data to have 64x64 points, when using overlapping windows this results in:
    ##window_size = 64x64
    ##stride/overlap = 2 x 2
    window_size = (64, 64)
    stride = (4,4)

    ## Create the model
    if unetname == 'unet_model_small':
        model = unets.unet_model_small(input_shape)
        ## Create batch generators for training and validation
        train_generator = generator.batch_generator(train_feature,train_target, [lowtopo,hightopo], [lowlsm,highlsm],len(train_feature), batch_size,cropbox,input_shape,z_branch)
        validation_generator = generator.batch_generator(valid_feature, valid_target, [lowtopo,hightopo], [lowlsm,highlsm],len(valid_feature), batch_size,cropbox,input_shape,z_branch)
    elif unetname == 'unet_model_small_window':
        model = unets.unet_model_small_window(input_shape_small)
        ## Create batch generators for training and validation
        train_generator = generator.batch_generator_2(train_feature,train_target, [lowtopo,hightopo], [lowlsm,highlsm],len(train_feature), batch_size,cropbox,window_size, stride,z_branch)
        validation_generator = generator.batch_generator_2(valid_feature, valid_target, [lowtopo,hightopo], [lowlsm,highlsm],len(valid_feature), batch_size,cropbox,window_size, stride,z_branch)

    elif unetname == 'build_unet_sha':
        # build, compile and train the model
        model = unets.build_unet_sha(input_shape, z_branch=z_branch)
        if type(train_feature) != list:
          train_generator = generator.batch_generator(train_feature,train_target, [lowtopo,hightopo], [lowlsm,highlsm],len(train_feature), batch_size,cropbox,input_shape,z_branch)
          validation_generator = generator.batch_generator(valid_feature, valid_target, [lowtopo,hightopo], [lowlsm,highlsm],len(valid_feature), batch_size,cropbox,input_shape,z_branch)
        elif type(train_feature) == list:
          ## Create batch generators for training and validation
          ##also, resetting the batch_size:
          batch_size = 1
          train_generator = generator.batch_generator_3(train_feature,train_target, [lowtopo,hightopo], [lowlsm,highlsm],len(train_feature), batch_size,cropbox,z_branch)
          validation_generator = generator.batch_generator_3(valid_feature, valid_target, [lowtopo,hightopo], [lowlsm,highlsm],len(valid_feature), batch_size,cropbox,z_branch)
    
    ## Print model summary
    model.summary()

    ## Define checkpoints for early stopping and saving the best model
    checkpoint_path = modelpath ##put the checkpoints into the model directory!
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True)
    if tf.__version__ == '1.12.0':
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
    else:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Define learning rate scheduler
    
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr  # Keep initial learning rate for the first 10 epochs
        elif epoch >= 10 and epoch < 30:
            return lr * tf.math.exp(-0.001)  # Decrease learning rate by a factor of 0.1 after 10 epochs
        elif epoch >= 30:
            return lr    

    lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
    
    ## Compile the model
    model.compile(optimizer=optimizer, loss=loss_function, run_eagerly=True)

    
    # Train the model
    #model.fit(train_generator, steps_per_epoch=train_steps, epochs=num_epochs,
    #          validation_data=validation_generator, validation_steps=validation_steps)
   
    if unetname == 'build_unet_sha':
        if z_branch:
            model.compile(optimizer=optimizer,
                    loss={"output_temp": "mae", "output_z": "mae"}, 
                    loss_weights={"output_temp": 1.0, "output_z": 1.0})
        else:
            model.compile(optimizer=optimizer,
                    loss={"output_temp": loss_function}, run_eagerly=True)
             

        model.fit(train_generator, batch_size=batch_size, epochs=num_epochs,  callbacks=[checkpoint, early_stopping, lr_scheduler_callback], validation_data=validation_generator)

    else:
        model.fit(train_generator, steps_per_epoch=train_steps, epochs=num_epochs,
                  validation_data=validation_generator, validation_steps=validation_steps,
                  callbacks=[checkpoint, early_stopping, lr_scheduler_callback])
    
    
    # Save the trained model to a file
    model.save(modelpath)
