'''
21.06.2023, I. Schicker

Utils for downscaling of ERA5 2m temperature to CERRA using a UNET for the Code4Earth Challenge


History:
- 21.06.2023, I. Schicker: splitting the test skript into a more readable code
- 23.06.2023, I. Schicker: added the calculation of weights for weighted loss functions
'''

import os
#from downscaling_utils import create_plots
import numpy as np
import pandas as pd
import xarray as xr

##plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

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

###############################################TOPOGRAPHY MODIFICATION##########################################
def topography_lsm(filepath, cropbox, crop=True):
    '''
    21.06.2023, I. Schicker
    
    Read in topography and lsm and return "stripped" and cropped data.
    
    Input dimension = (time, latitude, longitude)
    cropbox = lat0,lat1,lon0,lon1
    
    MIND: latitude is descending in both CERRA and ERA5, lat0 = latmax
    '''
    data = xr.open_dataset(filepath)
    parameter = list(data.keys())[0]
    
    data[parameter] = data[parameter][0,:,:]
    ##dropping time
    data = data.drop_vars('time')   
    
    if crop:
        data_crop = data.sel(latitude=slice(cropbox[0],cropbox[1]), longitude=slice(cropbox[2],cropbox[3]))
    
    return data_crop
    

##################################################CALCULATE (INVERSE) WEIGHTS######################################
def calc_weights_local_percentiles(data, method='inv'):
    
    def get_percentiles(data):
        percentiles = []
        for p in np.arange(50,100):
            percentiles.append(np.percentile(data,p))
        return np.array(percentiles)
    
    if method=='inv':
        weights = 50/np.arange(50,0,-1)
        weights/=weights[0]
    elif method=='lin':
        weights = np.arange(1,51,1)
    
    all_weights = np.zeros(data.shape)
    for i in range(64):
        print(i)
        for j in range(64):

            percentiles = get_percentiles(data[:,i,j])
            percentiles = np.append(percentiles, percentiles[-1]+100)

            cats = np.zeros(data[:,i,j].shape, dtype=int)
            for k in range(len(percentiles)-1):
                indices = np.where( (data[:,i,j]>=percentiles[k]) & (data[:,i,j]<percentiles[k+1]) )
                cats[indices] = k

            all_weights[:,i,j] = weights[cats]    
            
    return all_weights 
        
        
#################################################################################################################
def normalize(x,xmin,xmax):
    """
    Normalizes data by a min-max skaling method.

    The data is returned again in their panda data structures.

    Input parameters:
        - x data to normalize as a numpy matrix or panda data frame

    Output:
        - xnorm: normalized input
        - scaler: scaling object for later usage for denormalizing
    """
    
    data = (x - xmin) / (xmax - xmin)

    return data



#################################################################################################################
def denormalize(data, xmin,xmax):
    """
    denormalize the data by the scaler (typically obtained at a previous normalization)

    parameters:
        - x: normalized input
        - scaler: scaling object used in the normalizing process
    returns the denormalized data as pandas data frame
    """
    x = data * (xmax - xmin) + xmin
    
    return x

#################################################################################################################
def denormalize_predictions(data, flinktopo, flinktemp):
    """
    Denormalizing predictions. Need to get/add links to the data for scaling.
    """

    topominmax = xr.open_dataset(flinktopo)
    tempmin = xr.open_dataset(flinktemp[0])
    tempmax = xr.open_dataset(flinktemp[1])

    if len(data.keys()) == 1:
        temp_denorm = denormalize(data['prediction_t2m_normalized'], tempmin['t2m'][0,:,:],tempmax['t2m'][0,:,:]).to_dataset(name='predicted_t2m')
        data['predicted_t2m'] = temp_denorm['predicted_t2m']    
    elif len(data.keys()) > 1:
        
        topo_denorm = denormalize(data['prediction_topography_normalized'],topominmax['orog'][0,:,:].min(),topominmax['orog'][0,:,:].max()).to_dataset(name='predicted_topo')
        data['predicted_topo'] = topo_denorm['predicted_topo']

    return data

#################################################################################################################
def destandardise_predictions(data, flinktemp_mu,flinktemp_std):
    """
    Denormalizing predictions. Need to get/add links to the data for scaling.
    """
    
    tempmu = xr.open_dataset(flinktemp_mu)
    tempstd = xr.open_dataset(flinktemp_std)
    
    tempfc_denorm = data['prediction_topography_normalized'] * tempstd['sd_modeled'] + tempmu['mu_modeled']
    #print(temp_denorm)
    #temp_denorm['predicted_t2m'][48,:,:].plot()
    #plt.show()
    #print(topo_denorm)
    #topo_denorm['predicted_topo'][48,:,:].plot()
    #plt.show()
    
    data['predicted_t2m'] = tempfc_denorm['predicted_t2m']

    return data

#################################################TO_XARRAY#######################################################
def predict_2_xarray(data, metafile,z_branch, unet_name):
    """ 
    Convert the predicted array to xarray
    """
    if z_branch:
        data_vars = {'prediction_t2m_normalized':(['time', 'latitude','longitude'], data[:,:,:,0], 
                        {'units': 'Kelvin', 
                        'long_name':'normalized_t2m'}), 
                        'prediction_topography_normalized':(['time', 'latitude','longitude'], data[:,:,:,1], 
                        {'units': 'm', 
                        'long_name':'normalized_topography'})}
    else:
        data_vars = {'prediction_t2m_normalized':(['time', 'latitude','longitude'], data[:,:,:], 
                        {'units': 'Kelvin', 
                        'long_name':'normalized_t2m'})}
       
    # define coordinates
    coords = {'time': (['time'], metafile.time.data),
                'latitude':(['latitude'], metafile.latitude.data),
                'longitude':(['longitude'], metafile.longitude.data)}

    # define global attributes
    attrs = {'creation_date': datetime.now(), 
                'author':'Irene Schicker, within the frame of Code4Earth 2023, Issue 25', 
                'modelname': unet_name,
                'email':'irene.schicker@geosphere.at'}
    
    predictions_ds = xr.Dataset(data_vars=data_vars, 
                    coords=coords, 
                    attrs=attrs)
    
    return predictions_ds


#########################PLOTTING function
def plottingfield(plotdata, paramname,plotfilename, title, mini, maxi):
    '''
    Plot the desired field metric and add the score as a text field into the plot.
    '''
    #print(plotdata)
    plotdata = plotdata.rename({list(plotdata.keys())[0]:paramname}) 

    ##spatial plot levels
    clevs=range(mini,maxi,1)  
    #print(clevs)
    colors = ["#233777","#2D4289","#384E9C","#425BB1","#5069C3","#6578C8","#7A89CE","#8F9AD6","#A4ADDE","#BCC2E6","#D7DBF1","#FFFFFF",
              "#F3D5D6","#E9B8BA","#DF9FA2","#D5898C","#CB7478","#C16064","#B64C51","#A33F45","#90353A","#7D2B30","#6B2226"]
    cmap_name = 'my_templist'
    cmaptemp = LinearSegmentedColormap.from_list(cmap_name, colors)
    
    # Set up the projection that will be used for plotting
    mapcrs = ccrs.LambertConformal(central_longitude=12, central_latitude=47, standard_parallels=(30, 60))
    
    # Set up the projection of the data; if lat/lon then PlateCarree is what you want
    datacrs = ccrs.PlateCarree()
    fig, ax = plt.subplots()
    fig = plt.figure(1, figsize=(25, 20))
    # Add geopolitical boundaries for map reference
    ax = plt.subplot(111, projection=mapcrs)
    ax.set_extent([plotdata.x.min(), plotdata.x.max(), plotdata.y.min(), plotdata.y.max()], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    
    cs = plotdata[paramname].plot.pcolormesh(cmap=cmaptemp,vmin=mini, vmax=maxi, transform=datacrs, label=False, add_colorbar=False) #,alpha=0.7) 
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False,linewidth=0.5, color="k", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    
    
    ax.set_title(title, fontsize=12)
    plt.grid('on')
    #plt.tight_layout()
    
    plt.colorbar(cs, fraction=0.046, pad=0.15, orientation="horizontal").set_label(label='2m temperature',size=8) #,weight='bold'
    fig.set_size_inches(15.5, 7.5)
    plt.savefig(plotfilename, dpi=300, bbox_inches='tight')
    
    #plt.show()
    plt.close('all')
    
