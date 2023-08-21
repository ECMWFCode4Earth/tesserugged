'''
21.06.2023, I. Schicker

Downscaling of ERA5 2m temperature to CERRA using a UNET for the Code4Earth Challenge - Training

Prediction Code

History:
- 21.06.2023, I. Schicker: splitting the test skript into a more readable code
'''

import os
import numpy as np
import pandas as pd
import rioxarray as riox
import xarray as xr
from rioxarray.merge import merge_datasets
from datetime import datetime

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

##for model plotting:
#import visualkeras

##########import the model components
import unets 
import generator as gen
import utils as ut


################################################################################
def predict(inputlist,modelpath,lowtfile, hightfile, lowlsmfile,highlsmfile,cropbox,unet_name,datasource,file_topo, files_temperature, input_shape, z_branch):
    '''
    21.06.2023, I. Schicker
    # Perform prediction using the loaded model
    # predictions = loaded_model.predict(data)
    # Function for data preprocessing and prediction    
    
    '''

    # Load the saved model
    loaded_model = tf.keras.models.load_model(modelpath)  
    #print(loaded_model.summary())
    #visualkeras.layered_view(loaded_model, legend=True, to_file='Modelsetup_'+unet_name+'.png').show()
    
    ## Preprocess the data
    ##Input file list for testing
    #if type(inputlist) != list:
    predict_data_path_list = inputlist #.values.flatten()
    print(predict_data_path_list)
    ##loading topography data and add time dimension
    lowtopo = ut.topography_lsm(lowtfile, cropbox, crop=True)
    hightopo = ut.topography_lsm(hightfile, cropbox, crop=True)
    lowlsm = ut.topography_lsm(lowlsmfile, cropbox, crop=True)
    highlsm = ut.topography_lsm(highlsmfile, cropbox, crop=True)
    
    ##get time array_
    #data_ds = xr.open_mfdataset(predict_data_path_list)

    
    if unet_name == 'unet_model_small_window':
        # Set the window size
        window_size = (64, 64)
        stride = (4,4)
        
        # Generate sub-windows from the input data for prediction
        generator = gen.batch_generator_2_prediction(predict_data_path_list, [lowtopo,hightopo], [lowlsm, highlsm],cropbox,window_size,stride)
        print(generator)
        # Predict for each sub-window
        predictions = []
        for sub_window in generator:
            #Reshape the sub-window to match the model's input shape
            ##HACKY as hell right now............                    
            data = np.concatenate([sub_window['t2m_normalized'].values[..., np.newaxis], sub_window['orogl_normalized'].values[..., np.newaxis],
                                    sub_window['orog_normalized'].values[..., np.newaxis], sub_window['llsm'].values[..., np.newaxis],
                                    sub_window['lsm'].values[..., np.newaxis]], axis=-1)

            #sub_window = sub_window.reshape(1, window_size[0], window_size[1], 5)
            #print(sub_window.shape)
            # Perform prediction
            prediction = loaded_model.predict(data)
            #print(prediction)
            print(prediction.shape)
            ##transform to xarray
            # create dataset
            # define data with variable attributes
            ##DONE in utils.py
            predictions_ds = ut.predict_2_xarray(prediction, sub_window,z_branch, unet_name)

            
            sub_window.close()

            if datasource == 'minmaxnorm':
                #### Now de-normalize temperature and topography
                predictions_ds_denorm = ut.denormalize_predictions(predictions_ds,file_topo, files_temperature)
                ##close non-necessary files
                predictions_ds.close()
            elif datasource == 'residuals':
                ##########RESIDUALS DENORM:
                ##fc * cerrs.sd + cerra.mu
                ##TBD!!
                predictions_ds_denorm = ut.estandardise_predictions(predictions_ds, flinktemp_mu,flinktemp_std)
                print('TBD')
            
            
            # Append the prediction to the list
            predictions.append(predictions_ds_denorm)
    
        ## Concatenate the predictions into a single array    
        predictions_fulldomain = merge_datasets(predictions, nodata=np.nan)
        #print(predictions)
        print(predictions_fulldomain.shape)
    
    elif unet_name == 'unet_model_small':
        '''
        This is the prediction part for the small (160, 240) window domain. This can be used for either the small_window approach.
        Needs to be adapted!
        
        '''
        # Generate sub-windows from the input data for prediction
        generator = gen.batch_generator_predict(predict_data_path_list, [lowtopo,hightopo], [lowlsm, highlsm],num_samples, batch_size, cropbox, input_shape)
        print(generator)

        predictions = []
        for batch in generator:    
            ## Make prediction using the loaded model
            prediction = loaded_model.predict(data)
    
            print(prediction.shape)
    
           ##transform to xarray
            # create dataset
            # define data with variable attributes
            ##DONE in utils.py
            predictions_ds = ut.predict_2_xarray(prediction, sub_window,z_branch, unet_name)            

            if datasource == 'minmaxnorm':
                #### Now de-normalize temperature and topography
                predictions_ds_denorm = ut.denormalize_predictions(predictions_ds,file_topo, files_temperature)
                ##close non-necessary files
                predictions_ds.close()
                print(predictions_ds_denorm)
            elif datasource == 'residuals':
                ##########RESIDUALS DENORM:
                ##fc * cerrs.sd + cerra.mu
                ##TBD!!
                print('TBD')

        ##########RESIDUALS DENORM:
        ##fc * cerrs.sd + cerra.mu

    
    elif unet_name == 'build_unet_sha':
        '''
        This is the prediction part for the small (160, 240) window domain. This can be used for either the small_window approach.
        Needs to be adapted!
        
        '''
        
        print('We are in USha prediction')
        # Generate sub-windows from the input data for prediction
        if type(predict_data_path_list) != list:
          generator = gen.batch_generator_predict(predict_data_path_list, [lowtopo,hightopo], [lowlsm,highlsm],len(train_feature), batch_size,cropbox,input_shape)
        elif type(predict_data_path_list) == list:
          ## Create batch generators for training and validation
          ##also, resetting the batch_size as in the :
          print(predict_data_path_list)
          print(len(predict_data_path_list))
          batch_size = 2
          generator = gen.batch_generator_predict(predict_data_path_list, [lowtopo,hightopo], [lowlsm,highlsm],len(predict_data_path_list), batch_size,cropbox,input_shape)


        predictions = []
        for batch in generator:    
            ## Make prediction using the loaded model
            prediction = loaded_model.predict(batch)
    
            print(prediction)
    
            ##transform to xarray
            # create dataset
            # define data with variable attributes
            ##DONE in utils.py
            predictions_ds = ut.predict_2_xarray(prediction, sub_window,z_branch, unet_name)
            
            #### Now de-normalize temperature and topography
            print(file_topo)
            print(files_temperature)
            if datasource == 'minmaxnorm':
                #### Now de-normalize temperature and topography
                predictions_ds_denorm = ut.denormalize_predictions(predictions_ds,file_topo, files_temperature)
                ##close non-necessary files
                predictions_ds.close()
                print(predictions_ds_denorm)
            elif datasource == 'residuals':
                ##########RESIDUALS DENORM:
                ##fc * cerrs.sd + cerra.mu
                ##TBD!
                print('TBD')
            
            predictions.append(predictions_ds_denorm)



        ##########RESIDUALS DENORM:
        ##fc * cerrs.sd + cerra.mu
    
    ##convert to xarray and denormalize. Needs path to xmin and xmax files
    # Create dummy coordinates for time, latitude, and longitude
    #time_coords = np.arange(predictions.shape[0])  # Assuming predictions.shape[0] represents the number of sub-windows
    #lat_coords = np.arange(start_latitude, end_latitude, lat_resolution)  # Replace start_latitude, end_latitude, and lat_resolution with the actual values
    #lon_coords = np.arange(start_longitude, end_longitude, lon_resolution)  # Replace start_longitude, end_longitude, and lon_resolution with the actual values

    #prediction = predictions
    
    ##denorm:
    #xmin = xr.open_dataset(xminpath)
    #xmax = xr.open_dataset(xmaxpath)
    #predictions_normal = ut.denormalize(prediction, xmin,xmax)
        

    
    # Close the data file
    lowtopo.close()
    hightopo.close()

    #return predictions
