'''
21.06.2023, I. Schicker

Generator for downscaling of ERA5 2m temperature to CERRA using a UNET for the Code4Earth Challenge

History:
- 21.06.2023, I. Schicker: split the test code into separate function files. 

'''

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

# Set the seed for reproducibility
seed = 123
##check which tensorflow version
if tf.__version__ == '1.12.0':
    tf.random.set_random_seed(seed)
else:
    tf.random.set_seed(seed)


######################################GENERATOR#################################
def batch_generator(data_path, label_path, terrain_features,lsm_features, num_samples, batch_size, cropbox, input_shape,z_branch):
    '''
    21.06.2023, I. Schicker
    
    This is the generator for the UNET version (but can be used somewhere else, too).
    
    It expects:
      - data_path        = a pandas data frame consisting of rows with the path to the input feature file. 
                           Here, these are the interpolated ERA5 files, monthly files.
      - label_path       = a pandas data frame consisting of rows with the path to the target file.
                           Here, these are the to latlon converted CERRA files. 
      - terrain_features = low resolution ERA5 topography and high resolution CERRA topography [lowres, highres]
                           Xarray data sets.
      - num_samples      = number of overall samples in the training/validation data set. 3hourly*days_per_year*years
      - batch_size       = the batch_size used in the training process.
      - window_size      = and another splitting of the data to fit it into memory, splitting time
      - input_shape      = size of the target domain we defined before.
      - z_branch         = if we use the highres topo as additional target

    Input data in the paths are expected to be netcdf files.
    
    IF adding lsm as additional input feature, make some minor adaptations below.
    '''
    

    while True:
        ## Pre-load the terrain data as they do not have a time dimension

        lowtopo = terrain_features[0]
        hightopo = terrain_features[1]
        lowlsm = lsm_features[0]
        highlsm = lsm_features[1]        
 
        for offset in range(0, num_samples, batch_size):
            
            ##We check what type our data_path is
            ## - if it is an xarray data set we use the xarray batching directly
            ##   also, no need for cropping if it is xarray as then we use, for this C4E challenge, the residuals which are already cropped
            ## - if it is a list, we use first list subsetting, then open the subsetted list
            if type(data_path) != 'list':
                
                data_ds_cropp = data_path.isel(time=slice(offset,offset + batch_size))
                label_ds_cropp = label_path.isel(time=slice(offset,offset + batch_size))
                
                if 'latitude' not in list(data_ds_cropp.coords.keys()):
                    #print('Renaming coordinates to longitude and latitude')
                    ##MIND: expects dimensions to be (time,lon,lat) in the coords list for data_ds and (lon,lat,time) for label_ds!!
                    data_ds_cropp = data_ds_cropp.rename({list(data_ds_cropp.coords.keys())[1]:'longitude',list(data_ds_cropp.coords.keys())[2]:'latitude'})
                    label_ds_cropp = label_ds_cropp.rename({list(label_ds_cropp.coords.keys())[0]:'longitude',list(label_ds_cropp.coords.keys())[1]:'latitude'})
                
                ##check size of domain, if larger then we need to cropp using the cropbox
                if data_ds_cropp.sizes['latitude'] > input_shape[0] or data_ds_cropp.sizes['longitude'] > input_shape[1]:
                    data_ds_cropp = data_ds_cropp.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
                    label_ds_cropp = label_ds_cropp.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))                
                

            elif type(data_path) == 'list':
                ##get the filenames corresponding to the offset range
                data_path_list = data_path[offset:offset + batch_size].values.flatten()
                label_path_list = label_path[offset:offset + batch_size].values.flatten()
                
                ##read the meteorological parameter input feature and the target
                data_ds = xr.open_mfdataset(data_path_list)
                label_ds = xr.open_mfdataset(label_path_list)
                
                ##ATTENTION!!! We decided, as proposed by the spanish team, to use a smaller domain for testing purposes
                ## cropping here to the smaller domain
                data_ds_cropp = data_ds.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
                label_ds_cropp = label_ds.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
                ## get rid of the big data set
                data_ds = None
                label_ds = None
            
            # Expand the topography and lsm using the time dimension in the input feature data set
            lowtopo_time = lowtopo.expand_dims(time=data_ds_cropp.time).sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3])) #.squeeze()
            hightopo_time = hightopo.expand_dims(time=data_ds_cropp.time).sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))#.squeeze()
            lowlsm_time = lowlsm.expand_dims(time=data_ds_cropp.time).sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))#.squeeze()
            highlsm_time = highlsm.expand_dims(time=data_ds_cropp.time).sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))#.squeeze()
           

            ## Merge input fields with batch_data
            ##first, get the names:
            parameterdata = list(data_ds_cropp.keys())[0]
            parameterltopo = list(lowtopo_time.keys())[0]
            parameterhtopo = list(hightopo_time.keys())[0]
            parameterllsm = list(lowlsm_time.keys())[0]           
            parameterhlsm = list(highlsm_time.keys())[0]
            
            

            # Merge input fields with batch_data
            data = np.concatenate([data_ds_cropp[parameterdata].values[..., np.newaxis], lowtopo_time[parameterltopo].values[..., np.newaxis],
                                    hightopo_time[parameterhtopo].values[..., np.newaxis], lowlsm_time[parameterllsm].values[..., np.newaxis],
                                    highlsm_time[parameterhlsm].values[..., np.newaxis]], axis=-1)            
            
            ##Now do the same for target data
            parametertarget = list(label_ds_cropp.keys())[0]
            if z_branch:
                label = np.concatenate([label_ds_cropp[parametertarget].values[..., np.newaxis], hightopo_time[parameterhtopo].values[..., np.newaxis]],
                                    axis=-1)
            else:       
                label = label_ds_cropp[parametertarget].values
            
            #print(data)
            #print(label)
            print(data.shape)
            print(label.shape)
            
            ##removing the time-based topography to avoid errors            
            hightopo_time = None
            lowtopo_time = None
            highlsm_time = None
            lowlsm_time = None            
            
            ##yield the data
            yield data, label

######################################GENERATOR SLIDING WINDOW#################################
def batch_generator_2(data_path, label_path, terrain_features, lsm_features,num_samples, batch_size, cropbox, window_size, stride, z_branch):
    '''
    21.06.2023, I. Schicker
   
    This is the generator for the UNET version (but can be used somewhere else, too).
   
    It expects:
      - data_path        = a pandas data frame consisting of rows with the path to the input feature file.
                           Here, these are the interpolated ERA5 files, monthly files.
      - label_path       = a pandas data frame consisting of rows with the path to the target file.
                           Here, these are the to latlon converted CERRA files.
      - terrain_features = low resolution ERA5 topography and high resolution CERRA topography [lowres, highres]
                           Xarray data sets.
      - lsm_features     = low resolution ERA5 lsm and high resolution CERRA lsm [lowres, highres]
                           Xarray data sets.
      - num_samples      = number of overall samples in the training/validation data set. 3hourly*days_per_year*years
      - batch_size       = the batch_size used in the training process.
      - cropbox          = the bounding box coordinates [lat_min, lat_max, lon_min, lon_max] to crop the data.
      - window_size      = size of the sub-windows to extract from the data.
      - stride           = size of overlapping pixels
      - z_branch         = if we use highres topo as additional target
 
    Input data in the paths are expected to be netcdf files.
    '''
   
    # Pre-load the terrain data as they do not have a time dimension
    lowtopo = terrain_features[0]
    hightopo = terrain_features[1]
    lowlsm = lsm_features[0]
    highlsm = lsm_features[1]    
 
    while True:
        for offset in range(0, num_samples, batch_size):
            print('Yielding data')           
            # Get the filenames corresponding to the offset range
            data_path_list = data_path[offset:offset + batch_size].values.flatten()
            label_path_list = label_path[offset:offset + batch_size].values.flatten()
            #print(data_path_list)
            #print(num_samples, batch_size, len(data_path))
            # Read the meteorological parameter input feature and the target
            data_ds = xr.open_mfdataset(data_path_list)
            label_ds = xr.open_mfdataset(label_path_list)
           
            # ATTENTION!!! We decided, as proposed by the Spanish team, to use a smaller domain for testing purposes
            # Cropping here to the smaller domain
            data_ds_cropp = data_ds.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
            label_ds_cropp = label_ds.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
            
            ## Get rid of the big data sets
            data_ds = None
            label_ds = None
           
            ## Get the dimensions of the data
            time_steps  = data_ds_cropp.sizes['time'] #num_samples_time, needed for version with residuals
            lat_steps = data_ds_cropp.sizes['latitude']
            lon_steps = data_ds_cropp.sizes['longitude']
           
            
            ## Calculate the number of sub-windows in the latitude and longitude directions
            lat_subwindows = (lat_steps - window_size[0]) // stride[0] + 1
            lon_subwindows = (lon_steps - window_size[1]) // stride[1] + 1
            #print(lat_subwindows, lon_subwindows)
            #print(lat_steps,lon_steps)
            #print(data_ds_cropp)

            sw = 0
            for i in range(0,lat_steps ,window_size[0]-stride[0]):  #- window_size[0] + stride[0]
                i1 = i+window_size[0] #+stride[0]
                #print(i1,data_ds_cropp['latitude'][i:i1])
                for j in range(0,lon_steps,window_size[1]-stride[1]): #- window_size[1] + stride[0]
                    j1 = j+window_size[0] #+stride[1]
                    #print(j1,data_ds_cropp['longitude'][j:j1])
                    
                    ## Create the smaller spatial batches for the small_windowing UNET
                    # Extract the sub-window from data and labels
                    sub_window_data = data_ds_cropp.isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                    sub_window_labels = label_ds_cropp.isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                    
                    # Expand the topography and lsm using the time dimension in the input feature data set
                    lowtopo_time = lowtopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                    hightopo_time = hightopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                    lowlsm_time = lowlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                    highlsm_time = highlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()

                    ##Depending on the overall domain size we need to adjust the subwindows to match our window_size:
                    if sub_window_labels.sizes['longitude'] < window_size[1]:
                        print('longitude dims too small, extending domain overlap to the West')
                        diffsize = window_size[1] - sub_window_labels.sizes['longitude']
                        j = j - diffsize
                        sub_window_data = data_ds_cropp.isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        sub_window_labels = label_ds_cropp.isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        lowtopo_time = lowtopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        hightopo_time = hightopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        lowlsm_time = lowlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        highlsm_time = highlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()

                    if sub_window_labels.sizes['latitude'] < window_size[0]:
                        print('latitude dims too small, extending domain overlap to the North')
                        diffsize = window_size[0] - sub_window_labels.sizes['latitude']
                        i = i - diffsize
                        sub_window_data = data_ds_cropp.isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        sub_window_labels = label_ds_cropp.isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        lowtopo_time = lowtopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        hightopo_time = hightopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        lowlsm_time = lowlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        highlsm_time = highlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()


                    ##for validation purposes, write the sub-window domains into a text file, append, so that we can later plot them
                    text = "subwindow " + str(sw) + " "+str(sub_window_data.latitude.min().values) + " "+str(sub_window_data.latitude.max().values) + " "+str(sub_window_data.longitude.min().values)  + " "+str(sub_window_data.longitude.max().values) + str(sub_window_data.time.min().values) +" " + sub_window_data.time.max().values+"\n"
                    with open("subwindows_train.txt", "a") as f:
                            f.write(text)


                    # Merge input fields with batch_data
                    data = np.concatenate([sub_window_data['t2m_normalized'].values[..., np.newaxis], lowtopo_time['orog_normalized'].values[..., np.newaxis],
                                            hightopo_time['orog_normalized'].values[..., np.newaxis], lowlsm_time['lsm'].values[..., np.newaxis],
                                            highlsm_time['lsm'].values[..., np.newaxis]], axis=-1)
                    
                    
                    
                    # Now do the same for target data
                    parametertarget = list(label_ds_cropp.keys())[0]
                    if z_branch:
                        label = np.concatenate([label_ds_cropp[parametertarget].values[..., np.newaxis], hightopo_time[parameterhtopo].values[..., np.newaxis]],
                                            axis=-1)
                    else:       
                        label = label_ds_cropp[parametertarget].values
                    

                    sw += 1
                    yield data, label


######################################GENERATOR TIME CHUNKS#################################
def batch_generator_3(data_path, label_path, terrain_features, lsm_features,num_samples, batch_size, cropbox, z_branch):
    '''
    07.07.2023, I. Schicker
   
    This is the generator for the UNET version (but can be used somewhere else, too).
   
    It expects:
      - data_path        = a pandas data frame consisting of rows with the path to the input feature file.
                           Here, these are the interpolated ERA5 files, monthly files.
      - label_path       = a pandas data frame consisting of rows with the path to the target file.
                           Here, these are the to latlon converted CERRA files.
      - terrain_features = low resolution ERA5 topography and high resolution CERRA topography [lowres, highres]
                           Xarray data sets.
      - lsm_features     = low resolution ERA5 lsm and high resolution CERRA lsm [lowres, highres]
                           Xarray data sets.
      - num_samples      = number of overall samples in the training/validation data set. 3hourly*days_per_year*years
      - batch_size       = the batch_size used in the training process.
      - cropbox          = the bounding box coordinates [lat_min, lat_max, lon_min, lon_max] to crop the data.
      - window_size      = size of the sub-windows to extract from the data.
      - stride           = size of overlapping pixels
      - z_branch         = if we use highres topo as additional target
 
    Input data in the paths are expected to be netcdf files.
   
    uses time chunks depending on the input size.
    '''
   
    # Pre-load the terrain data as they do not have a time dimension
    lowtopo = terrain_features[0]
    hightopo = terrain_features[1]
    lowlsm = lsm_features[0]
    highlsm = lsm_features[1]    
 
    while True:
        for offset in range(0, num_samples, batch_size):
            print('Yielding data')   
            print(offset)
                   
            # Get the filenames corresponding to the offset range
            data_path_list = data_path[offset:offset + batch_size] #.values.flatten()
            label_path_list = label_path[offset:offset + batch_size] #.values.flatten()
            print(data_path_list)
            print(num_samples, batch_size, len(data_path))

            # Read the meteorological parameter input feature and the target
            data_ds = xr.open_mfdataset(data_path_list)
            label_ds = xr.open_mfdataset(label_path_list)
                        
            # ATTENTION!!! We decided, as proposed by the Spanish team, to use a smaller domain for testing purposes
            # Cropping here to the smaller domain
            data_ds_cropp = data_ds.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
            label_ds_cropp = label_ds.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
            
            ## Get rid of the big data sets
            data_ds = None
            label_ds = None
           
            ## Get the dimensions of the data
            time_steps  = data_ds_cropp.sizes['time'] #num_samples_time, needed for version with residuals
            #lat_steps = data_ds_cropp.sizes['latitude']
            #lon_steps = data_ds_cropp.sizes['longitude']
           
            ##calculate window_size
            window_size_div = 8
            print(time_steps, window_size_div, time_steps/window_size_div )
            

            sw = 0
            for i in range(0,time_steps +1,int(time_steps/window_size_div)):  #- window_size[0] + stride[0]
                i1 = i+ int(time_steps/window_size_div)
             
                ## Create the smaller spatial batches for the small_windowing UNET
                # Extract the sub-window from data and labels
                sub_window_data = data_ds_cropp.isel(time=slice(i,i1)).squeeze()
                sub_window_labels = label_ds_cropp.isel(time=slice(i,i1)).squeeze()
                print('FEATURES')
                print(sub_window_data)
                print('Target')
                print(sub_window_labels)
                #exit()
                # Expand the topography and lsm using the time dimension in the input feature data set
                lowtopo_time = lowtopo.expand_dims(time=data_ds_cropp.time).isel(time=slice(i,i1)).squeeze()
                hightopo_time = hightopo.expand_dims(time=data_ds_cropp.time).isel(time=slice(i,i1)).squeeze()
                lowlsm_time = lowlsm.expand_dims(time=data_ds_cropp.time).isel(time=slice(i,i1)).squeeze()
                highlsm_time = highlsm.expand_dims(time=data_ds_cropp.time).isel(time=slice(i,i1)).squeeze()

                ##for validation purposes, write the sub-window domains into a text file, append, so that we can later plot them
                #text = "subwindow " + str(sw) + " "+str(sub_window_data.latitude.min().values) + " "+str(sub_window_data.latitude.max().values) + " "+str(sub_window_data.longitude.min().values)  + " "+str(sub_window_data.longitude.max().values) + str(sub_window_data.time.min().values) +" " + sub_window_data.time.max().values+"\n"
                #with open("subwindows_train.txt", "a") as f:
                #        f.write(text)


                # Merge input fields with batch_data
                data = np.concatenate([sub_window_data['t2m_normalized'].values[..., np.newaxis], lowtopo_time['orog_normalized'].values[..., np.newaxis],
                                        hightopo_time['orog_normalized'].values[..., np.newaxis], lowlsm_time['lsm'].values[..., np.newaxis],
                                        highlsm_time['lsm'].values[..., np.newaxis]], axis=-1)
                
                
                
                # Now do the same for target data
                parametertarget = list(sub_window_labels.keys())[0]
                if z_branch:
                    label = np.concatenate([labsub_window_labelsel_ds_cropp[parametertarget].values[..., np.newaxis], hightopo_time[parameterhtopo].values[..., np.newaxis]],
                                        axis=-1)
                else:       
                    label = sub_window_labels[parametertarget].values                

                sw += 1
                yield data, label

                

######################################################GENERATOR PREDICTION###############################################################
######################################GENERATOR#################################
def batch_generator_predict(data_path, terrain_features,lsm_features, num_samples, batch_size, cropbox, input_shape):
    '''
    21.06.2023, I. Schicker
    
    This is the generator for the UNET version (but can be used somewhere else, too).
    
    It expects:
      - data_path        = a pandas data frame consisting of rows with the path to the input feature file. 
                           Here, these are the interpolated ERA5 files, monthly files.
      - terrain_features = low resolution ERA5 topography and high resolution CERRA topography [lowres, highres]
                           Xarray data sets.
      - num_samples      = number of overall samples in the training/validation data set. 3hourly*days_per_year*years
      - batch_size       = the batch_size used in the training process.
      - window_size      = and another splitting of the data to fit it into memory, splitting time
      - input_shape      = size of the target domain we defined before.

    Input data in the paths are expected to be netcdf files.
    
    IF adding lsm as additional input feature, make some minor adaptations below.
    '''
    

    while True:
        ## Pre-load the terrain data as they do not have a time dimension

        lowtopo = terrain_features[0]
        hightopo = terrain_features[1]
        lowlsm = lsm_features[0]
        highlsm = lsm_features[1]        
 
        for offset in range(0, num_samples, batch_size):
            print(type(data_path))
            ##We check what type our data_path is
            ## - if it is an xarray data set we use the xarray batching directly
            ##   also, no need for cropping if it is xarray as then we use, for this C4E challenge, the residuals which are already cropped
            ## - if it is a list, we use first list subsetting, then open the subsetted list
            if type(data_path) != list:
                
                data_ds_cropp = data_path.isel(time=slice(offset,offset + batch_size))
                 
                if 'latitude' not in list(data_ds_cropp.coords.keys()):
                    #print('Renaming coordinates to longitude and latitude')
                    ##MIND: expects dimensions to be (time,lon,lat) in the coords list for data_ds and (lon,lat,time) for label_ds!!
                    data_ds_cropp = data_ds_cropp.rename({list(data_ds_cropp.coords.keys())[1]:'longitude',list(data_ds_cropp.coords.keys())[2]:'latitude'})
                
                ##check size of domain, if larger then we need to cropp using the cropbox
                if data_ds_cropp.sizes['latitude'] > input_shape[0] or data_ds_cropp.sizes['longitude'] > input_shape[1]:
                    data_ds_cropp = data_ds_cropp.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
                

            elif type(data_path) == list:
                ##get the filenames corresponding to the offset range
                data_path_list = data_path[offset:offset + batch_size] #.values.flatten()
                print(data_path_list)
                
                ##read the meteorological parameter input feature and the target
                data_ds = xr.open_mfdataset(data_path_list)
                 
                ##ATTENTION!!! We decided, as proposed by the spanish team, to use a smaller domain for testing purposes
                ## cropping here to the smaller domain
                data_ds_cropp = data_ds.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
                ## get rid of the big data set
                data_ds = None

            # Expand the topography and lsm using the time dimension in the input feature data set
            lowtopo_time = lowtopo.expand_dims(time=data_ds_cropp.time).sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3])) #.squeeze()
            hightopo_time = hightopo.expand_dims(time=data_ds_cropp.time).sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))#.squeeze()
            lowlsm_time = lowlsm.expand_dims(time=data_ds_cropp.time).sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))#.squeeze()
            highlsm_time = highlsm.expand_dims(time=data_ds_cropp.time).sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))#.squeeze()
           

            ## Merge input fields with batch_data
            ##first, get the names:
            parameterdata = list(data_ds_cropp.keys())[0]
            parameterltopo = list(lowtopo_time.keys())[0]
            parameterhtopo = list(hightopo_time.keys())[0]
            parameterllsm = list(lowlsm_time.keys())[0]           
            parameterhlsm = list(highlsm_time.keys())[0]
            
            

            # Merge input fields with batch_data
            data = np.concatenate([data_ds_cropp[parameterdata].values[..., np.newaxis], lowtopo_time[parameterltopo].values[..., np.newaxis],
                                    hightopo_time[parameterhtopo].values[..., np.newaxis], lowlsm_time[parameterllsm].values[..., np.newaxis],
                                    highlsm_time[parameterhlsm].values[..., np.newaxis]], axis=-1)            
            

            print(data.shape)
        
            ##removing the time-based topography to avoid errors            
            hightopo_time = None
            lowtopo_time = None
            highlsm_time = None
            lowlsm_time = None            
            
            ##yield the data
            yield data

###################################################################################################################################
def batch_generator_2_prediction(data_path, terrain_features,lsm_features, cropbox, window_size, stride):
    '''
    This is the batch generator for the prediction part using the sub-window approach.
    
    It expects:
      - data_path        : a pandas data frame consisting of rows with the path to the input feature file. 
                           Here, these are the interpolated ERA5 files, monthly files.
      - terrain_features : low-resolution ERA5 topography and high-resolution CERRA topography [lowres, highres]
                           xarray data sets.
      - lsm_features     = low resolution ERA5 lsm and high resolution CERRA lsm [lowres, highres]
                           Xarray data sets.

      - window_size      : size of the sub-windows to extract from the data.

    Input data in the paths is expected to be netCDF files.
    
    IF adding lsm as an additional input feature, make some minor adaptations below.
    '''
    
    # Pre-load the terrain data as they do not have a time dimension
    lowtopo = terrain_features[0]
    lowtopo = lowtopo.rename({'orog_normalized':'orogl_normalized'})
    hightopo = terrain_features[1]
    lowlsm = lsm_features[0]
    lowlsm = lowlsm.rename({'lsm':'llsm'})
    highlsm = lsm_features[1]    

    while True:
        for data_file in data_path:           
            # Read the meteorological parameter input feature
            #print(data_file)
            data_ds = xr.open_dataset(data_file)
            data_ds_cropp = data_ds.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
            data_ds.close()
            
            ## Get the dimensions of the data
            time_steps  = data_ds_cropp.sizes['time'] #num_samples_time --> might be needed in the future
            lat_steps = data_ds_cropp.sizes['latitude']
            lon_steps = data_ds_cropp.sizes['longitude']
           
            
            ## Calculate the number of sub-windows in the latitude and longitude directions
            #lat_subwindows = (lat_steps - window_size[0]) // stride[0] + 1
            #lon_subwindows = (lon_steps - window_size[1]) // stride[1] + 1
            #print(lat_subwindows, lon_subwindows)
            sw = 0
            
            for i in range(0,lat_steps, window_size[0]-stride[0]):  #- window_size[0] + stride[0]
                i1 = i+window_size[0] #+stride[0]
                
                for j in range(0,lon_steps,window_size[1]-stride[1]): #- window_size[1] + stride[0]
                    j1 = j+window_size[0] #+stride[1]
                    
                    
                    ## Create the smaller spatial batches for the small_windowing UNET
                    # Extract the sub-window from data and labels
                    sub_window_data = data_ds_cropp.isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                    
                    # Expand the topography and lsm using the time dimension in the input feature data set
                    lowtopo_time = lowtopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                    hightopo_time = hightopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                    lowlsm_time = lowlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                    highlsm_time = highlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()

                    ##Depending on the overall domain size we need to adjust the subwindows to match our window_size:
                    if sub_window_data.sizes['longitude'] < window_size[1]:
                        print('longitude dims too small, extending domain overlap to the West')
                        diffsize = window_size[1] - sub_window_data.sizes['longitude']
                        j = j - diffsize
                        sub_window_data = data_ds_cropp.isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        lowtopo_time = lowtopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        hightopo_time = hightopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        lowlsm_time = lowlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        highlsm_time = highlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()

                    if sub_window_data.sizes['latitude'] < window_size[0]:
                        print('latitude dims too small, extending domain overlap to the North')
                        diffsize = window_size[0] - sub_window_data.sizes['latitude']
                        i = i - diffsize
                        sub_window_data = data_ds_cropp.isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        lowtopo_time = lowtopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        hightopo_time = hightopo.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        lowlsm_time = lowlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()
                        highlsm_time = highlsm.expand_dims(time=data_ds_cropp.time).isel(latitude=slice(i,i1),longitude=slice(j,j1)).squeeze()


                    ##for validation purposes, write the sub-window domains into a text file, append, so that we can later plot them
                    text = "subwindow " + str(sw) + " "+str(sub_window_data.latitude.min().values) + " "+str(sub_window_data.latitude.max().values) + " "+str(sub_window_data.longitude.min().values)  + " "+str(sub_window_data.longitude.max().values) +"\n"
                    with open("subwindows_train.txt", "a") as f:
                            f.write(text)
                    
                                        
                    # Merge input fields with batch_data, using xr info here as we need it to add time, lat,lon info back to subwindow prediction:
                    data_xr = xr.merge([sub_window_data['t2m_normalized'], lowtopo_time['orogl_normalized'],
                                            hightopo_time['orog_normalized'], lowlsm_time['llsm'],highlsm_time['lsm']])

                    sw += 1

                    yield data_xr

            
            # Close the data file
            data_ds.close()

