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
def predict(inputlist,modelpath,lowtfile, hightfile, lowlsmfile,highlsmfile,cropbox,unet_name,datasource,file_topo, files_temperature, input_shape, z_branch, batch_size,normsource,normalise,leadtime):
    '''
    21.06.2023, I. Schicker
    # Perform prediction using the loaded model
    # predictions = loaded_model.predict(data)
    # Function for data preprocessing and prediction    
    
    '''

    # Load the saved model
    loaded_model = tf.keras.models.load_model(modelpath)  
    print(loaded_model.summary())
    #visualkeras.layered_view(loaded_model, legend=True, to_file='Modelsetup_'+unet_name+'.png').show()
    
    ## Preprocess the data
    ##Input file list for testing
    #if type(inputlist) != list:
    predict_data_path_list = inputlist #.values.flatten()
    print(predict_data_path_list)
    print(len(predict_data_path_list))
    print(batch_size)
    
    ##loading topography data and add time dimension
    lowtopo = ut.topography_lsm(lowtfile, cropbox, crop=True)
    hightopo = ut.topography_lsm(hightfile, cropbox, crop=True)
    lowlsm = ut.topography_lsm(lowlsmfile, cropbox, crop=True)
    highlsm = ut.topography_lsm(highlsmfile, cropbox, crop=True)
   
    ## Prepare, just in case, the feature and target "climate" files in case we need to normalize.
    ## hopefully saves resources in batching
    ##data path min and max in normsource
    feature_min = xr.open_dataset(normsource[0]).isel(time=0)
    feature_max = xr.open_dataset(normsource[1]).isel(time=0)
    ##drop bounds here!
    feature_min = feature_min.drop_vars('time_bnds')
    feature_max = feature_max.drop_vars('time_bnds')
    
    ##crop to domain, target
    if 'latitude' not in list(feature_min.coords.keys()):
        #print('Renaming coordinates to longitude and latitude')
        ##MIND: expects dimensions to be (time,lat,lon) in the coords list!!
        feature_min = feature_min.rename({list(feature_min.coords.keys())[1]:'longitude',list(feature_min.coords.keys())[2]:'latitude'})
        feature_max = feature_max.rename({list(feature_max.coords.keys())[1]:'longitude',list(feature_max.coords.keys())[2]:'latitude'})

    if datasource != 'residuals':
        feature_min = feature_min.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
        feature_max = feature_max.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))

    if 'time' in list(feature_min.coords.keys()):
        feature_min = feature_min.drop('time')
        feature_max = feature_max.drop('time')

    if 'crs' in list(feature_min.keys()):
        feature_min = feature_min.drop('crs')
        feature_max = feature_max.drop('crs')
    
    normsource = [feature_min, feature_max]
    ##get time array_
    #data_ds = xr.open_mfdataset(predict_data_path_list)
    print(batch_size)
    
    if unet_name == 'unet_model_small_window':
        # Set the window size
        window_size = (64, 64)
        stride = (4,4)
        
        # Generate sub-windows from the input data for prediction
        generator = gen.batch_generator_2_prediction(predict_data_path_list, [lowtopo,hightopo], [lowlsm, highlsm],len(predict_data_path_list),cropbox,window_size,stride)
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
        generator = gen.batch_generator_predict(predict_data_path_list, [lowtopo,hightopo], [lowlsm, highlsm],len(predict_data_path_list), batch_size, cropbox, input_shape,normsource,normalise)
        print(generator)

        predictions = []
        i = 0
        for batch in generator:    
            ##HACKY as hell right now............                    
            print(i)
            print(batch)
            
            ###ANOTHER HACK: we have the following params:   t2m, oro_low, orog_normalized,lsm_low, lsm

            data = np.concatenate([batch['t2m'].values[..., np.newaxis], batch['oro_low'].values[..., np.newaxis],
                                    batch['orog_normalized'].values[..., np.newaxis], batch['lsm_low'].values[..., np.newaxis],
                                    batch['lsm'].values[..., np.newaxis]], axis=-1)

            ## Make prediction using the loaded model
            prediction = loaded_model.predict(data)
    
            print(prediction.shape)
            #print(prediction)
            #print(data)
            
            ##transform to xarray
            # create dataset
            # define data with variable attributes
            ##DONE in utils.py
            predictions_ds = ut.predict_2_xarray(prediction, batch,z_branch, unet_name)            

            if datasource == 'minmaxnorm':
                #### Now de-normalize temperature and topography
                predictions_ds_denorm = ut.denormalize_predictions(predictions_ds,file_topo, files_temperature)
                ##close non-necessary files
                predictions_ds.close()
                print('#########################PREDICTION################################')
                print(predictions_ds_denorm)
            elif datasource == 'residuals':
                ##########RESIDUALS DENORM:
                ##fc * cerrs.sd + cerra.mu
                ##TBD!!
                print('TBD')
            i += 1
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
            ##the generator gives xarrays ... so we need to transform it to the needed input structure, np.array
            data = batch.to_array().to_numpy()
            print(data.shape)


            ## Make prediction using the loaded model
            prediction = loaded_model.predict(batch)
    
            print(prediction)
    
            ##transform to xarray
            # create dataset
            # define data with variable attributes
            ##DONE in utils.py
            predictions_ds = ut.predict_2_xarray(prediction, batch,z_branch, unet_name)
            
            #### Now de-normalize temperature and topography
            print(file_topo)
            print(files_temperature)
            
            #### Now de-normalize temperature and topography
            predictions_ds_denorm = ut.denormalize_predictions(predictions_ds,file_topo, files_temperature)
            
            ##close non-necessary files
            predictions_ds.close()
            print(predictions_ds_denorm)
            
            
            predictions.append(predictions_ds_denorm)



        ##########CONCAT predictions
        prediction_all = xr.merge(predictions)
        prediction_all.to_netcdf('./PRED/Prediction_'+unet_name+'_'+str(leadtime)+'.nc')
        
        

    
    # Close the data file
    lowtopo.close()
    hightopo.close()

    #return predictions

