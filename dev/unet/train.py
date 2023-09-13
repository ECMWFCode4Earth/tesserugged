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
               batch_size, num_epochs, loss_function, optimizer, modelpath, z_branch,normalize,normsource,datasource):
    '''
    21.06.2023, I. Schicker
    
    This is the main training part of the Unets
    '''
    #print(normsource)
    ##loading topography data
    lowtopo = ut.topography_lsm(lowtfile, cropbox, crop=True)
    hightopo = ut.topography_lsm(hightfile, cropbox, crop=True)
    lowlsm = ut.topography_lsm(lowlsmfile, cropbox, crop=True)
    highlsm = ut.topography_lsm(highlsmfile, cropbox, crop=True)

    ## Prepare, just in case, the feature and target "climate" files in case we need to normalize.
    ## hopefully saves resources in batching
    ##data path min and max in normsource
    feature_min = normsource[0]
    feature_max = normsource[1]
    target_min = normsource[2]
    target_max = normsource[3]
    ##drop bounds here!
    if 'time_bnds' in feature_min.keys():
        feature_min = feature_min.drop_vars('time_bnds')
        feature_max = feature_max.drop_vars('time_bnds')
        target_min = target_min.drop_vars('time_bnds')
        target_max = target_max.drop_vars('time_bnds')
    #print(feature_min)
    #exit()
    ##crop to domain, target
    if 'latitude' not in list(feature_min.coords.keys()):
        #print('Renaming coordinates to longitude and latitude')
        ##MIND: expects dimensions to be (time,lat,lon) in the coords list!!
        feature_min = feature_min.rename({list(feature_min.coords.keys())[1]:'longitude',list(feature_min.coords.keys())[2]:'latitude'})
        feature_max = feature_max.rename({list(feature_max.coords.keys())[1]:'longitude',list(feature_max.coords.keys())[2]:'latitude'})

        target_min = target_min.rename({list(target_min.coords.keys())[1]:'longitude',list(target_min.coords.keys())[0]:'latitude'})
        target_max = target_max.rename({list(target_max.coords.keys())[1]:'longitude',list(target_max.coords.keys())[0]:'latitude'})

        #print(feature_min)
        
    if datasource != 'residuals':
        feature_min = feature_min.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
        feature_max = feature_max.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
        target_min = target_min.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
        target_max = target_max.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
        #print(feature_min)

    if 'time' in list(feature_min.coords.keys()):
        feature_min = feature_min.drop('time')
        feature_max = feature_max.drop('time')
        target_min = target_min.drop('time')
        target_max = target_max.drop('time')

    if 'crs' in list(feature_min.keys()):
        feature_min = feature_min.drop('crs')
        feature_max = feature_max.drop('crs')
    if 'crs' in list(target_min.keys()):
        target_min = target_min.drop('crs')
        target_max = target_max.drop('crs')

    ##interpolate feature data if coarse resolved
    ##Check the resolution of the min and max feature files, if we need to interpolate to CERRA or not
    resolution = feature_min['longitude'][1].values - feature_min['longitude'][0].values
    print(resolution)
    if resolution > 0.05:
       ##interpolate:
       #print(feature_min)
       feature_min = feature_min.interp(latitude=target_min.latitude.data,longitude = target_min.longitude.data)
       feature_max = feature_max.interp(latitude=target_min.latitude.data, longitude = target_min.longitude.data)
       
    
    normsource = [feature_min, feature_max, target_min, target_max]

    ## Define train and validation steps
    print('TYPE DATA: ', type(train_feature))
    #print(valid_feature)
    #print(valid_target)
    if type(train_feature) != list:
        total_train_samples = train_feature.sizes['time'] #len(trainperiod)  # Total number of training samples
        total_validation_samples = valid_feature.sizes['time'] #len(validperiod)  # Total number of validation samples
        train_steps = int(np.floor(total_train_samples // batch_size))
        validation_steps = int(np.floor(total_validation_samples // batch_size))
        number_of_batches = train_feature.to_array().to_numpy().shape[1] / batch_size
 
    else:
        #print(train_feature)
        #exit()
        total_train_samples = len(train_feature) #len(trainperiod)  # Total number of training samples
        total_validation_samples = len(valid_feature) #len(validperiod)  # Total number of validation samples
        train_steps = int(np.floor(total_train_samples // batch_size))
        validation_steps = int(np.floor(total_validation_samples // batch_size))

    print('Number of total samples in training, validation, and batch_size: ',total_train_samples, total_validation_samples, batch_size)
    print('Number of samples in training, validation, and batch_size: ',train_steps, validation_steps, batch_size)
    
    
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
        model = unets.unet_model_small(input_shape,z_branch)
        ## Create batch generators for training and validation
        train_generator = generator.batch_generator(train_feature,train_target, [lowtopo,hightopo], [lowlsm,highlsm],total_train_samples, batch_size,cropbox,input_shape,z_branch,normalize,normsource)
        validation_generator = generator.batch_generator(valid_feature, valid_target, [lowtopo,hightopo], [lowlsm,highlsm],total_validation_samples, batch_size,cropbox,input_shape,z_branch,normalize,normsource)
    elif unetname == 'unet_model_small_window':
        model = unets.unet_model_small_window(input_shape_small)
        ## Create batch generators for training and validation
        train_generator = generator.batch_generator_2(train_feature,train_target, [lowtopo,hightopo], [lowlsm,highlsm],total_train_samples, batch_size,cropbox,window_size, stride,z_branch)
        validation_generator = generator.batch_generator_2(valid_feature, valid_target, [lowtopo,hightopo], [lowlsm,highlsm],total_validation_samples, batch_size,cropbox,window_size, stride,z_branch)

    elif unetname == 'build_unet_sha':
        # build, compile and train the model
        model = unets.build_unet_sha(input_shape, z_branch=z_branch)
        # define the generator: 
        train_generator = generator.batch_generator(train_feature,train_target, [lowtopo,hightopo], [lowlsm,highlsm],total_train_samples, batch_size,cropbox,input_shape,z_branch,normalize,normsource)
        validation_generator = generator.batch_generator(valid_feature, valid_target, [lowtopo,hightopo], [lowlsm,highlsm],total_validation_samples, batch_size,cropbox,input_shape,z_branch,normalize,normsource)
 
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
    print('##############COMPILE MODEL###############') 
    tf.config.run_functions_eagerly(True)

    if z_branch:
       model.compile(optimizer=optimizer,
                    loss={"output_temp": "mae", "output_z": "mae"}, 
                    loss_weights={"output_temp": 1.0, "output_z": 1.0})
    else:
       model.compile(optimizer=optimizer,
                    loss={"output_temp": loss_function}, run_eagerly=True)
             
    ## FIT the model
    print('#############FITTING ',unetname,'################')
    print(batch_size, num_epochs)
    #exit()
    if unetname == 'build_unet_sha':
        model.fit(train_generator, batch_size=batch_size, epochs=num_epochs,  callbacks=[checkpoint, early_stopping, lr_scheduler_callback], validation_data=validation_generator, verbose=1)

    else:
        print(train_steps, num_epochs, validation_steps, batch_size) #validation_steps=validation_steps, 
        
        ##removing validation data!
        model.fit(train_generator, steps_per_epoch=train_steps, epochs=num_epochs, callbacks=[checkpoint, early_stopping, lr_scheduler_callback],verbose=1)

        ## model.fit(train_generator, steps_per_epoch=train_steps, epochs=num_epochs,validation_data=validation_generator, callbacks=[checkpoint, early_stopping, lr_scheduler_callback],verbose=1)    
    
    # Save the trained model to a file
    model.save(modelpath)

