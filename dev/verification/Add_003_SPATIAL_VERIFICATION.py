"""
V_011_ANOMALIES.py

09.02.2023, I. Schicker

Verification of weekly aggregates and anomalies for the control run
Deterministic only.

Uses config_AnomVerifDet.ini to get settings
"""

import os,sys, subprocess
import datetime, time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import tarfile
import shutil
from io import BytesIO
import joblib
import gc 
import configparser

##plotting
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

import xarray as xr
import xskillscore as xs
import h5py
import dask.dataframe as daskd
from dask.delayed import delayed
import dask.dataframe as dd
import dask.array as da

#########################################FUNCTIONS##################################################
#######################ANOMALIECALCULATION
def calc_anomaly(climfiles,data, parname,climname):
    '''
    The definition of the climatology has important implications in forecast verification. 
    The weekly anomalies used for verification are computed as deviations from 
    the climatology over the 20 years of hindcast (1996-2015). Likewise, the weekly anomalies
    of the observations are calculated as deviations from the observed climatology over the
    same period. 
    The definition of the climatology affects as well the climatological forecast
    benchmark used as reference for the skill scores and the
    adjustment of the biases, which is performed with respect to the observed climatology. To analyze the impact
    that the construction of the climatology has on the skills
    scores, three different approaches have been tested.
    They are described below and represented with the gray
    boxes in the schematic in Fig. 1. For each case, the same
    method has been employed to compute the climatology
    in the reanalysis and in the hindcast. In the case of the
    hindcast, the climatology is computed for each forecast
    time independently (weeks 1-4).https://www.7thverificationworkshop.de/Presentation/S2S_1_Vitart.pdf'''

    print ('Calculating weekly anomalies ...')
    anomalies = data[parname] - climfiles[climname]  #.groupby('time')
    return anomalies.to_dataset(name=parname)

##############################AGGREGATES CALCULATION
def calculate_weekly_averages(forecast, initialization_date):
    # Convert initialization date to datetime64 format
    initialization_date = np.datetime64(initialization_date)
    
    # Filter forecast to start from initialization date
    forecast = forecast.where(forecast.time >= initialization_date, drop=True)
    
    # Resample forecast to weekly intervals and calculate the mean
    weekly_averages = forecast.resample(time='1W').mean(dim='time')
    
    return weekly_averages

#################################ACC Function
def ACC(FC,OBS,CL):
    '''
    FC = Forecast
    OBS = Obs
    CL = Climatological value
    https://metclim.ucd.ie/wp-content/uploads/2017/07/DeterministicSkillScore.pdf
    Correlations between forecasts and observations may have too high correlations due to
    seasonal variations therefore the anomaly correlation coefficient (ACC) is used. It re-
    moves the climate average from both forecast and observations and verifies the anomalies.

    Increasing numerical values indicate increasing “success”. An ACC=60% corresponds to
    the range up to which there is synoptic skill for the largest weather patterns. An ACC=
    50% corresponds to forecasts for which the error is the same as for a forecast based on
    a climatological average.
    '''
    top = np.mean((FC-CL)*(OBS-CL))
    bottom = np.sqrt(np.mean((FC-CL)**2)*np.mean((OBS-CL)**2))
    ACC = top/bottom
    return ACC

#########################PLOTTING function
def plottingfieldbias(plotdata, score, paramname,plotfilename, title):
    '''
    Plot the desired field metric and add the score as a text field into the plot.
    '''
    print(plotdata)

    # Set up the projection that will be used for plotting
    mapcrs = ccrs.LambertConformal(central_longitude=12, central_latitude=47, standard_parallels=(30, 60))
    
    # Set up the projection of the data; if lat/lon then PlateCarree is what you want
    datacrs = ccrs.PlateCarree()
    fig, ax = plt.subplots()
    fig = plt.figure(1, figsize=(25, 20))
    # Add geopolitical boundaries for map reference
    ax = plt.subplot(111, projection=mapcrs)
    ax.set_extent([plotdata.longitude.min(), plotdata.longitude.max(), plotdata.latitude.min(), plotdata.latitude.max()], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    
    cs = plotdata[paramname].plot.pcolormesh(cmap='BrBG_r',vmin=-15., vmax=15.,transform=datacrs, label=False, add_colorbar=False) #, label=False, add_colorbar=False) 
    plt.text(2,44.0, 'Spatial BIAS = '+str(np.round(score,2)), fontsize=10, bbox = dict(facecolor='white', alpha=0.9),transform=datacrs)
    #+'\nRMSE = '+str(np.round(rmse, 2))+'\nMAE = '+ str(np.round(mae, 2))+'\nSS-MSE = ' +str(np.round(SS_MSE, 2)), fontsize=10, bbox = dict(facecolor='white', alpha=0.3))
        
    #(cmap="binary", vmin=0, vmax=1000, transform=datacrs, label=False, add_colorbar=False,alpha=0.7)#, binaryPuButransform=ccrs.PlateCarree())
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False,linewidth=0.5, color="k", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    
    
    ax.set_title(title, fontsize=12)
    plt.grid('on')
    #plt.tight_layout()
    
    plt.colorbar(cs, fraction=0.046, pad=0.15, orientation="horizontal").set_label(label='2m temperature BIAS',size=8) #,weight='bold'
    fig.set_size_inches(15.5, 7.5)
    plt.savefig(plotfilename, dpi=300, bbox_inches='tight')
    
    #plt.show()
    plt.close('all')
    
#########################PLOTTING function
def plottingfield(plotdata, paramname,plotfilename, title):
    '''
    Plot the desired field metric and add the score as a text field into the plot.
    '''
    #print(plotdata)
    plotdata[paramname] = plotdata[paramname] -273.15

    ##spatial plot levels
    clevs=range(-20,40,4)  
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
    ax.set_extent([plotdata.longitude.min(), plotdata.longitude.max(), plotdata.latitude.min(), plotdata.latitude.max()], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    
    cs = plotdata[paramname].plot.pcolormesh(cmap=cmaptemp,vmin=-20, vmax=40., transform=datacrs, label=False, add_colorbar=False) #,alpha=0.7) 
    
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
    

##############################SCORES and WRITE###################################
def calculate_scores(vmodel,datamodel,targetdata_weekly, weekly_scenario, incaclim_date_weekly, basefilescoresoverall, basefilescores,dateinit,index,plotfilename,verifparameter, averaging):
  '''
  Calculate the scores and write to a file
  Next step: call plotting function.
  '''

  param = list(datamodel.keys())[0]
  targetparam = list(targetdata_weekly.keys())[0]
  scenarioparam = list(weekly_scenario.keys())[0]
  #incaclimparam =  list(incaclim_date_weekly.keys())[0]
  
  ##Overall scores
  ##BIAS, RMSE, MSE, MAESS,ACC         
  biasmodel = (datamodel[param] - targetdata_weekly[targetparam]).to_dataset(name='temperature_bias_'+vmodel)
  #print(datamodel[param])
  #print(param)
  rmsemodel = np.sqrt(np.mean((datamodel[param] - targetdata_weekly[targetparam])**2)).to_dataset(name='temperature_rmse_'+vmodel)
  #rmseec = rmseec.values
  maemodel = (np.mean(np.abs(datamodel[param] - targetdata_weekly[targetparam]))).to_dataset(name='temperature_mae_'+vmodel) #.values
  #accmodel = ACC(datamodel[param],targetdata_weekly[targetparam],incaclim_date_weekly[incaclimparam])

  MSE_corr = np.mean(datamodel[param] - targetdata_weekly[targetparam])**2
  MSE_orig = np.mean(weekly_scenario[scenarioparam] - targetdata_weekly[targetparam])**2
  SS_MSE =  np.where(MSE_orig > MSE_corr, (MSE_corr - MSE_orig)/ (0. - MSE_orig), -1.*(MSE_orig - MSE_corr)/ (0. - MSE_corr))

  ###########################################DAILY/WEEKLY##################################
  ##weekly or daily scores
  ww = 1
  
  for week in datamodel.time:  
      print((pd.to_datetime(week.values)).strftime('%H'))
          
      
      plotfilenamebiasf = plotfilename.format(vmodel,'BIAS_'+(pd.to_datetime(week.values)).strftime('%Y%m%d'),(pd.to_datetime(week.values)).strftime('%H'),verifparameter)
      plottitlebiasf = 'BIAS for '+str(vmodel)+' at date '+(pd.to_datetime(week.values)).strftime('%Y%m%d')+' and hour '+(pd.to_datetime(week.values)).strftime('%H')
      plotfilenamef = plotfilename.format(vmodel,(pd.to_datetime(week.values)).strftime('%Y%m%d'),(pd.to_datetime(week.values)).strftime('%H'),verifparameter)
      plottitlef = 'Temperature for ' +str(vmodel)+ ' at date '+(pd.to_datetime(week.values)).strftime('%Y%m%d')+' and hour '+(pd.to_datetime(week.values)).strftime('%H')
      scorefilename = basefilescores.format('HOURLY',vmodel)
      header = 'INIT\tHOUR\tBIAS\tRMSE\tMAE\tACC\tMSE\tSS_MSE\n'

      ##Open weekly scores file: basefilescores
      if ww == 0:
        with open(scorefilename, 'w') as t:
              t.write(header)

      biasmodelw = (datamodel[param].sel(time=week) - targetdata_weekly[targetparam].sel(time=week)).to_dataset(name='temperature_bias_'+vmodel)
      rmsemodelw = np.sqrt(np.mean((datamodel[param].sel(time=week) - targetdata_weekly[targetparam].sel(time=week))**2))
      maemodelw = (np.mean(np.abs(datamodel[param].sel(time=week) - targetdata_weekly[targetparam].sel(time=week)))) #.values
      #accmodelw = ACC(datamodel[param].sel(time=week),targetdata_weekly[targetparam].sel(time=week),incaclim_date_weekly[incaclimparam].sel(time=week))
      #print(accmodelw.compute())
      
      MSE_corrw = np.mean(datamodel[param].sel(time=week) - targetdata_weekly[targetparam].sel(time=week))**2
      MSE_origw = np.mean(weekly_scenario[scenarioparam].sel(time=week) - targetdata_weekly[targetparam].sel(time=week))**2
      SS_MSEw =  np.where(MSE_origw > MSE_corrw, (MSE_corrw - MSE_origw)/ (0. - MSE_origw), -1.*(MSE_origw - MSE_corrw)/ (0. - MSE_corrw))
      
      with open(scorefilename, 'a+') as t:
        t.write(str((pd.to_datetime(week.values)).strftime('%Y%m%d'))+'\t'+(pd.to_datetime(week.values)).strftime('%H'))
        t.write('\t')
        t.write(str(np.round(biasmodelw['temperature_bias_'+vmodel].mean().values,2)))
        t.write('\t'+str(np.round(rmsemodelw.values.mean(), 2)))
        t.write('\t'+str(np.round(maemodelw.values.mean(), 2)))
        #t.write('\t'+str(np.round(accmodelw.values.mean(),2)))
        t.write('\t'+str(np.round(MSE_origw.values.mean(),2)))
        t.write('\t'+str(np.round(SS_MSEw.mean(),2)))
        t.write('\n')
      
      ###PLOTTING weekly
      ##Model_Metric_Parameter_atSAMOSgrid_date.png
      #print(biasmodel['temperature_bias_'+vmodel].mean(dim='time').compute())
      plottingfieldbias(biasmodelw['temperature_bias_'+vmodel].compute().to_dataset(name='temperature_bias_'+vmodel), biasmodelw['temperature_bias_'+vmodel].mean().values,
                    'temperature_bias_'+vmodel, plotfilenamebiasf, plottitlebiasf)
      plottingfield(datamodel[param].sel(time=week).compute().to_dataset(name=param), param, plotfilenamef,plottitlef)

      ww += 1
      #exit()
 
##########################MAIN FUNCTION##########################################    
def main():
  """
  For verification purposes calculate the daily and weekly anomalies.
  Also, perform verification on the INCA grid 
  For every date in period read:
      + ECLR2HR forecasts for that date
      + select corresponding observation
      + combine ECLR2HR_raw + modified + targetdate and perform aggregation to weekly values, than caluclate anomalies
      
  """
  ###############Settings via config_AnomVerifDet.ini
  config = configparser.ConfigParser()
  config.read('config_Verification.ini')

  ###verification parameter:
  verifparameter = config.get('BASICS','verifparameter')
 
  ###post-processing models we want to verify:
  postpromodels = config['FORECASTMODELS'].get('postpromodels').split(',')
  postpromodelpaths = config['FORECASTMODELS'].get('postpromodelpaths').split(',')

  for pmodel in postpromodels:
      if pmodel == 'SAMOS':
          downscaledf =  [x for x in postpromodelpaths if 'samos' in x][0]
          print(downscaledf)
          downscaledfull = xr.open_mfdataset(downscaledf.format(str('*')), engine='netcdf4', combine='nested')
          
           
  ###raw model path --> we need the raw model for some scores
  rawmodel = config['FORECASTMODELS'].get('rawmodel') #.split(',')
  rawmodelpath = config['FORECASTMODELS'].get('rawmodelpath') #.split(',')

  ###targetfiles are located here
  targetpath = config['TARGET'].get('targetpath')
  
  ###climatefiles are here --> we need the climatology for some scores
  climf = config['CLIMATOLOGY'].get('climpath')

  ###generating the verification dates
  start_trainday,end_trainday = config.get('BASICS','verificationperiod').split(',')
  daterange_loop = pd.date_range(start=start_trainday, end=end_trainday, freq="M")
  
  ###scorespath
  basefilescoresoverall = config['SCORES'].get('basefilescoresoverall')
  basefilescores = config['SCORES'].get('basefilescores')
  plotfilename = config['SCORES'].get('plotfilename')

  ##member to verify, mind we do deterministic here thus member is the control member
  MEMBER = config['BASICS'].getint('member')

  ##member to verify, mind we do deterministic here thus member is the control member
  averaging = config['BASICS'].get('averaging')
  
  #########################################CROPPING DOMAIN
  ### Before we start with calculations we need to reduce the domain size, SAMOS/GMOS have smaler domains so we verif

  minlat=config['CROPPING'].getfloat('minlat')
  maxlat=config['CROPPING'].getfloat('maxlat')
  minlon=config['CROPPING'].getfloat('minlon')
  maxlon=config['CROPPING'].getfloat('maxlon')
   
  #######################################INCA climate, data are in degree Celcius
  ###MISSING RIGHT NOW!
  '''
  clim_ = xr.open_dataset(climf)  
  clim = xr.concat([clim_['mu_modeled'],clim_['mu_modeled'],clim_['mu_modeled_2'],clim_['mu_modeled_3'],clim_['mu_modeled_4'],\
                      clim_['mu_modeled_5'],clim_['mu_modeled_6'],clim_['mu_modeled_7'],clim_['mu_modeled_8']], dim = 'time').to_dataset(name='mu_modeled')

  clim = clim.assign_coords(time=(clim.time + timedelta(days=365)))# + relativedelta(years=1)))
  print(clim)
  '''
  
  ## get parameter name in inca clim
  #climparametername = list(clim.keys())[0]
  ## to be on the safe side convert to float 64 and to Kelvin
  #clim[climparametername] = clim[climparametername].astype('float64')
  
  ######################################LOOP OVER INIT DATES################################################
  ### calculate weekly averages for model, raw model and climatology
  index = 0
  for dateinit in daterange_loop:
      print(dateinit)
      #print(daterange_loop)
      ######################################DETERMINISTIC VERIFICATION MODEL LOOP
      mm = 0 
      for model in postpromodels:
          ##path to post-processing model
          downscaledf = postpromodelpaths[mm]
          print(model, downscaledf)       

          ###############TARGET FILES
          ##We need to load the corresponding target files and the downscaled files.
          ##create two lists, the targetfiles and the day of year index for climatology
          with xr.open_dataset(targetpath.format(str(dateinit.strftime('%Y%m'))),lock=False) as target:
              target.load()
          
          ##parameter name might change, better use this:
          targetparametername = list(target.keys())[0]
          target = target.sel(latitude=slice(maxlat,minlat), longitude=slice(minlon,maxlon)) #.squeeze()
          print(target)

          ###############RAWMODEL files:
          with xr.open_dataset(rawmodelpath.format(str(dateinit.strftime('%Y%m'))),lock=False) as scenario:
              scenario.load()
          
          ##parameter name might change, better use this:
          rawparametername = list(scenario.keys())[0]
         
          ###cropping, interpolation is done later
          #scenario = scenario.sel(latitude=slice(maxlat,minlat), longitude=slice(minlon,maxlon)) #.squeeze()
          scenario = scenario.interp(longitude=target.longitude, latitude=target.latitude)
 
          #print(scenario)
          #exit()
          #print(pd.to_datetime(scenario.time.data[0])) #.strftime('%Y%m%d %H:%M:%S'))
          ###############DOWNSCALED FILES
          ##First, downscaled files (because, easier) --> check for file format!
          if model == 'SAMOS':
              downscaled = downscaledfull.sel(time=slice(pd.to_datetime(scenario.time.data[0]),pd.to_datetime(scenario.time.data[-1])))

               
          elif model == 'UNET':
              downscaled = xr.open_mfdataset(downscaledf.format(str(dateinit.strftime('%Y%m%d')), str(MEMBER).zfill(2)), engine='netcdf4')
          elif model == 'GAN':
              downscaled =  xr.open_mfdataset(downscaledf.format(str(dateinit.strftime('%Y%m%d')), str(MEMBER).zfill(2)), engine='netcdf4')

          ###get parameter name
          downscaledparametername = list(downscaled.keys())[0]                       
  
          ############CLIMATOLOGY################################
          ###climatologies for anomalies --> also need this for daily TAWES values!
          ##incaclim: we subset incaclim for the dates we have forecasted (as day of year) and add correct times
          ##it is used for both original and post-processed forecasts      
          '''
          clim_date["time"] = ("time", time.values.flatten())  
          clim_date = downscalede.assign_coords(time=downscalede.time - pd.Timedelta(hours=12))
          downscaled = downscaled.assign_coords(lead_time=downscaled.lead_time + fcsttime)

          clim_date = clim.sel(time=slice(pd.to_datetime(scenario.time.data[0]),pd.to_datetime(scenario.time.data[-1])))
          clim_date["time"] = ("time", time.values.flatten())  
          print(clim_date)
          '''
          
          ###############TARGET FILES
          ##We need to load the corresponding target files and the downscaled files.
          ##create two lists, the targetfiles and the day of year index for climatology
          with xr.open_dataset(targetpath.format(str(dateinit.strftime('%Y%m'))),lock=False) as targetdata:
              targetdata.load()

          ############WEEKLY AVERAGES YES/NO################################
          ###added this to also use this code for daily verification
          if averaging == 'True':
              ##calculate weekly averages scenario
              average_scenario = calculate_weekly_averages(scenario, dateinit)
              print('###########------RAW ECMWF AVERAGING------###########')
              ##getting rid of realization and forecast time, might need in future but not now
              average_scenario[rawparametername] = average_scenario[rawparametername] #[:,0,0,:,:] ##--> check if using other files                  
              
              ##calculate weekly averages downscaled       
              average_downscaled = calculate_weekly_averages(downscaled, dateinit)
              print('###########------PostPRO ECMWF AVERAGING------###########')
              
              ##calculate weekly averages target
              targetdata_average = calculate_weekly_averages(targetdata, dateinit)
              print('###########------TARGET AVERAGING------###########')
              
              '''
              ##calculate weekly averages climatology
              clim_date_average = calculate_weekly_averages(clim_date, dateinit)
              print('###########------CLIMATOLOGY------###########')
              #print(incaclim_date_weekly)
              '''
              ##release some memory and get rid of daily values
              scenario = None 
              targetdata = None              
              downscaled = None  
              #clim_date = None        
          else:
            average_scenario = scenario
            #clim_date_average = clim_date
            targetdata_average = targetdata
            average_downscaled = downscaled

            ##release some memory and get rid of daily values
            scenario = None 
            targetdata = None              
            downscaled = None  
            #incaclim_date = None  

          
          ##################################VERIFICATION and METRICS###################################################
          ### Now start with the calculation of indices, first only deterministic like RMSE etc.
          ## to keep things simple and easy readable we will use a loop --> convert to function!
          ## 
          verifmodels = [rawmodel, model] #'CLIM', 
          datamodel = [average_scenario, average_downscaled] #incaclim_date_average, 
          incaclim_date_average = None

          i = 0
          for vmodel in verifmodels:
              calculate_scores(vmodel,datamodel[i],targetdata_average, average_scenario, incaclim_date_average, basefilescoresoverall, basefilescores,dateinit,index, plotfilename,verifparameter,averaging)
              i += 1 
          
          mm += 1 
      index += 1
  



if __name__ == '__main__':
    main()
