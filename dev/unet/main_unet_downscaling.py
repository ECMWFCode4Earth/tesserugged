'''
15.06.2023, I. Schicker

Downscaling of ERA5 2m temperature to CERRA using a UNET for the Code4Earth Challenge

Main Code

History:
- 15.06.2023, I. Schicker: using an existing skript and modify for a first shot with a UNET. 
- 21.06.2023, I. Schicker: splitting the test skript into a more readable code


Uses args, training command:
python main_unet_downscaling.py --model USha --mode training --modelname MODEL/UNET_v0_hyper_MINMAXNORM_MSE_USha_nZ --lossfunction MSE --pathtotarget targets_unet_temp.csv --pathtofeatures features_unet_temp.csv  --datasource minmaxnorm --z_branch False --normalize True --leadtime 00

'''

import os, sys
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import numpy as np
import pandas as pd
#import rioxarray
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
import train
import predict
import utils as ut
import custom_loss_functions as clf

##testing something:
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


#fix_gpu()

def main():
    '''
    21.06.2023, I. Schicker
    
    This is the main code for both training and prediction for the C4E challenge issue 25
    
    It sets some basics and calls the respective functions
    '''
    ##Defining a parser, needs to be extended to other parameters once we are done:
    parser = argparse.ArgumentParser(prog='C4E Downscaling using a Unet',
                    description='Use a Unet to downscale ERA5 to CERRA for temperature',
                    epilog='Text at the bottom of help')
    ##chose which model
    parser.add_argument('-model',"--model", help="choose model architecture from: Ularge Usmall Uswindow USha", type=str, required=True)
    parser.add_argument('-mode',"--mode", help="choose mode from: training prediction", type=str, required=True)
    parser.add_argument('-modelname',"--modelname", help="the name to save the model to/load the model from", type=str, required=True)
    parser.add_argument('-loss',"--lossfunction", help="the loss function to use: MSE, MAE, IWMSE", type=str, default='MSE')
    parser.add_argument('-target',"--pathtotarget", help="the path to the target csv filelist", type=str)
    parser.add_argument('-features',"--pathtofeatures", help="the path to the features csv filelist", type=str)
    parser.add_argument('-datasource',"--datasource", help="What data source we are using: minmaxnorm, residuals", type=str)
    parser.add_argument('-z_branch',"--z_branch", help="Are we prediction high res topography? True, False", type=str)
    parser.add_argument('-normalize',"--normalize", help="Do we need to normalize the data in the batch generator? True, False",type=str)
    parser.add_argument('-ldtime',"--leadtime", help="Which leadtime we are working on",type=int)

    args = parser.parse_args()
    print(args)
    print("model choice is set to {}".format(args.model))
    print("we are {}".format(args.mode))
    print(args.normalize) 
    ##################################FILE PATH FEATURE/TARGET PAIRS etc. ################################
    ##feature and target path files have been created in another script. 
    ##They contain, right now, the training, validation, and prediction periods. Therefore,
    ##we need the datetimes below.
    ##next modification: provide these file paths as argument when calling the script
    feature_path = args.pathtofeatures #'features_unet_temp.csv'
    target_path = args.pathtotarget #'targets_unet_temp.csv'
    
    ##read the feature-target file
    features_list = pd.read_csv(feature_path, names=['Features'])
    targets_list = pd.read_csv(target_path, names=['Targets'])
    
    
    ##################################FILE PATH TOPOGRAPHY AND LSM ################################
    ###We are using topography as additional input information
    ###read topography and lsm, remove time dimension
    lowtfile = '../DATA/MM-STANDARDISED/orog_era5_005deg_standardised.nc'
    hightfile = '../DATA/MM-STANDARDISED/orog_cerra_005deg_standardised.nc'
    ##used:
    lowlsm ='../DATA/MM-STANDARDISED/lsm_era5_005deg_standardised.nc'
    highlsm ='../DATA/MM-STANDARDISED/lsm_cerra_005deg_standardised.nc'
    
    ##MIND: latitude is descending in both CERRA and ERA5, lat0 = latmax
    ##NEEDED FOR: 
    # ## - topography and land-sea mask
    # ## - minmaxnorm data
    cropbox = [44.96,37.0,-6.85,5.1]

    #################Additional definitions:
    input_shape = (160, 240, 5)

    if args.z_branch == 'True':
        z_branch = True
    else:
        z_branch = False

    ##############################MODEL ETC.#################################
    ##which unet
    ##for small domain so far only unet_model_small is prepared
    ##Ularge Usmall Uswindow
    if args.model == 'Uswindow':
        unet_name = 'unet_model_small_window'
    elif args.model == 'Usmall':
        unet_name = 'unet_model_small'
    elif args.model == 'Ularge':
        unet_name = 'unet_model_largedomain'
    elif args.model == 'USha':
        unet_name = 'build_unet_sha'

    modelpath = args.modelname

    loss_function_path = args.lossfunction

    if loss_function_path == 'MSE':
        ## Define loss function and optimizer
        if tf.__version__ == '1.12.0':
            loss_function = tf.contrib.losses.mean_squared_error
        else:
            loss_function = tf.keras.losses.MeanSquaredError()

    elif loss_function_path == 'MAE':
        ## Define loss function and optimizer
        if tf.__version__ == '1.12.0':
            loss_function = tf.contrib.losses.mean_absolute_error
        else:
            loss_function = tf.keras.losses.MeanAbsoluteError()

    elif loss_function_path == 'InvMSE':
        loss_function = clf.InverselyWeightedMSE()
    

     ##################################CALL TRAINING################################
    if args.mode == 'training':
        print('We are training')
        ################DEFINE TRAIN period & valid######################
        ##to determine the number of feature-target pairs make a nice datetime list
        trainperiod = pd.date_range('2010-01-01','2017-01-01',freq='3H')
        validperiod =  pd.date_range('2017-01-01','2017-12-31',freq='3H')
        

        ##data sources for training
        if args.datasource == 'minmaxnorm':
            train_feature = list(features_list['Features'][0:83])
            train_target =  list(targets_list['Targets'][0:83])
            valid_feature = list(features_list['Features'][83:95])
            valid_target =  list(targets_list['Targets'][83:95] )
            #inputlist =  list(features_list['Features'][95::])      #48::

        elif args.datasource == 'residuals':
            ##right now, residuals are all in one file per targettime
            train_feature = xr.open_dataset(features_list['Features'][0]).sel(time=slice('2010-01-01','2016-12-31'))
            train_target = xr.open_dataset(targets_list['Targets'][0]).sel(time=slice('2010-01-01','2016-12-31'))
            valid_feature = xr.open_dataset(features_list['Features'][0]).sel(time=slice(validperiod[0],validperiod[-1]))
            valid_target = xr.open_dataset(targets_list['Targets'][0]).sel(time=slice(validperiod[0],validperiod[-1]))
            #inputlist    = xr.open_dataset(features_list['Features'][0]).sel(time=slice('2018-01-01','2018-12-31'))
            print(train_feature)            
            ##residuals have the crs as variable, getting rid of them:
            train_feature = train_feature.drop_vars('crs')
            train_target = train_target.drop_vars('crs')
            valid_feature = valid_feature.drop_vars('crs')
            valid_target = valid_target.drop_vars('crs')
            #inputlist = inputlist.drop_vars('crs')        
                 
            ##residuals also have the coods named x and y, changing to longitude and latitude
            ##to be consistent with raw/original data
            if 'latitude' not in list(train_feature.coords.keys()):
                train_feature = train_feature.rename({'x':'longitude','y':'latitude'})
                train_target = train_target.rename({'x':'longitude','y':'latitude'})
                valid_feature = valid_feature.rename({'x':'longitude','y':'latitude'})
                valit_target = valid_target.rename({'x':'longitude','y':'latitude'})
                #inputlist = inputlist.rename({'x':'longitude','y':'latitude'})

                        
            ###we might need to interpolate the residuals to the target resolution
            resolution = train_feature['longitude'][1].values - train_feature['longitude'][0].values
            #print(resolution)
            if resolution > 0.0666666:
                ##interpolate:
                train_feature = train_feature.interp_like(train_target,method_non_numeric='nearest')
                valid_feature = valid_feature.interp_like(valid_target,method_non_numeric='nearest')
                #inputlist = inputlist.interp_like(train_target,method_non_numeric='nearest')
        
                

        ###############################Create/get data for normalization
        normalize = args.normalize    
        if normalize:
            if args.datasource == 'minmaxnorm':
                ##MIND, right now hardcoded, location is
                ERA5min = xr.open_dataset('DATA/CLIMS/t2m_era5_1985-2016_overall_min.nc').isel(time=0)
                ERA5max = xr.open_dataset('DATA/CLIMS/t2m_era5_1985-2016_overall_max.nc').isel(time=0)
                CERRAmin = xr.open_dataset('DATA/CLIMS/t2m_cerra_1985-2016_overall_min.nc').isel(time=0)
                CERRAmax = xr.open_dataset('DATA/CLIMS/t2m_cerra_1985-2016_overall_max.nc').isel(time=0)
            elif args.datasource == 'residuals':
                ##Calculating min and max on the fly depending on training source/length
                ERA5min =  train_feature.min(dim='time').compute()
                ERA5max =  train_feature.max(dim='time').compute()
                CERRAmin = train_target.min(dim='time').compute()
                CERRAmax = train_target.max(dim='time').compute()
                ##for prediction we going to read that from file so check if clim file exists:
                if not os.path.exists('DATA/CLIMS/ERA5_min_residuals_'+str(args.leadtime)+'.nc'):
                    ERA5min.to_netcdf('DATA/CLIMS/ERA5_min_residuals_'+str(args.leadtime)+'.nc')
                    ERA5max.to_netcdf('DATA/CLIMS/ERA5_max_residuals_'+str(args.leadtime)+'.nc')
                    CERRAmin.to_netcdf('DATA/CLIMS/CERRA_min_residuals_'+str(args.leadtime)+'.nc')
                    CERRAmax.to_netcdf('DATA/CLIMS/CERRA_max_residuals_'+str(args.leadtime)+'.nc')
        
        

            normsource = [ERA5min,ERA5max, CERRAmin, CERRAmax]

        #######################NOW the real stuff starts###########################################    
        ##Define batch_sikze and epochs
        if args.datasource == 'minmaxnorm':
          ## Create batch generators for training and validation
          ##also, resetting the batch_size:
          batch_size = 2
        
        if args.datasource == 'residuals':
            batch_size = 8
        num_epochs = 50

        
        optimizer = tf.keras.optimizers.Adam(learning_rate=5*10**(-4))

        
        train.train_unet(train_feature,train_target,valid_feature, valid_target,lowtfile, hightfile,lowlsm,highlsm,cropbox,unet_name, batch_size,
                             num_epochs,loss_function, optimizer, modelpath,z_branch,normalize,normsource, args.datasource)

 

                     
    ##################################CALL PREDICTING################################
    elif args.mode == 'prediction':
        ##Define batch_sikze and epochs
        if args.datasource == 'minmaxnorm':
            batch_size = 4
            ERA5min = xr.open_dataset('DATA/CLIMS/t2m_era5_1985-2016_overall_min.nc').isel(time=0)
            ERA5max = xr.open_dataset('DATA/CLIMS/t2m_era5_1985-2016_overall_max.nc').isel(time=0)
            CERRAmin = xr.open_dataset('DATA/CLIMS/t2m_cerra_1985-2016_overall_min.nc').isel(time=0)
            CERRAmax = xr.open_dataset('DATA/CLIMS/t2m_cerra_1985-2016_overall_max.nc').isel(time=0)

        if args.datasource == 'residuals':
            batch_size = 24
            ERA5min = xr.open_dataset('DATA/CLIMS/ERA5_min_residuals_'+str(args.leadtime)+'.nc')
            ERA5min = ERA5min.rename({list(ERA5min.keys())[0]:'t2mminera'})
            ERA5max = xr.open_dataset('DATA/CLIMS/ERA5_max_residuals_'+str(args.leadtime)+'.nc')
            ERA5max = ERA5max.rename({list(ERA5max.keys())[0]:'t2mmaxera'})
            CERRAmin = xr.open_dataset('DATA/CLIMS/CERRA_min_residuals_'+str(args.leadtime)+'.nc')
            CERRAmin = CERRAmin.rename({list(CERRAmin.keys())[0]:'t2mmincerra'})
            CERRAmax = xr.open_dataset('DATA/CLIMS/CERRA_max_residuals_'+str(args.leadtime)+'.nc')
            CERRAmax = CERRAmax.rename({list(CERRAmax.keys())[0]:'t2mmaxcerra'})

        normsource = [ERA5min,ERA5max, CERRAmin, CERRAmax]

        ##file paths for de-normalization for testing purposes only
        file_topo = './DATA/CLIMS/t2m_cerra_lsm_orog_201801_005deg.nc'
        
        ##input files, check again... features_input von oben nehmen
        if len(features_list['Features']) == 1:
            files_temperature = xr.open_dataset(features_list['Features'][0])
            files_temperature = files_temperature.drop_vars('crs')

        
        predict.predict(files_temperature, modelpath,lowtfile, hightfile, lowlsm,highlsm,cropbox,unet_name,args.datasource, file_topo,[CERRAmin, CERRAmax] ,input_shape, z_branch, batch_size,normsource,args.normalize,args.leadtime)
        

if __name__ == '__main__':
    
    fix_gpu()
    physical_devices = tf.config.list_physical_devices('GPU')
    print("physical_devices-------------", len(physical_devices))
    print("GPU--------------------------", tf.config.list_physical_devices('GPU'))
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
    main()

