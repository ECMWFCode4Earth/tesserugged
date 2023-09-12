#!/usr/bin/env bash

##################################################################
#Description    : wrangle tesserugged output to be used by the 
#                 DeepR valdation procedure
#Creation Date  : 2023-09-12
#Author         : Konrad Mayer
##################################################################

validation_dir='/scratch/klifol/kmayer/tmp/code4earth'
projectroot=$(git rev-parse --show-toplevel)

# create folders
mkdir -R $validation_dir/datasets/{baseline,model,cerra}
mkdir $validation_dir/validation

# merge output for individual lead times to common dataset
cdo mergetime $projectroot/dat/TESTING/SAMOS/postprocessed/*.nc $validation_dir/datasets/model/samos.nc
cdo mergetime $projectroot/dat/TESTING/PREPROCESSED/ERA5_regridded/*.nc $validation_dir/datasets/baseline/era5_regridded.nc
cdo mergetime $projectroot/dat/TESTING/PREPROCESSED/CERRA/*.nc $validation_dir/datasets/cerra/cerra.nc

# rename variables for prediction datasets
cdo chname,mu_samos,prediction $validation_dir/datasets/model/samos.nc $validation_dir/datasets/model/tmp.nc; mv $validation_dir/datasets/model/tmp.nc $validation_dir/datasets/model/samos.nc
cdo chname,t2m,prediction $validation_dir/datasets/baseline/era5_regridded.nc $validation_dir/datasets/baseline/tmp.nc; mv $validation_dir/datasets/baseline/tmp.nc $validation_dir/datasets/baseline/era5_regridded.nc

# removing of the grid_mapping attribute in the samos data and selection of the prediction variable
ncatted -a grid_mapping,prediction,d,, $validation_dir/datasets/model/samos.nc
cdo select,name=prediction, $validation_dir/datasets/model/samos.nc $validation_dir/datasets/model/tmp.nc; mv $validation_dir/datasets/model/tmp.nc $validation_dir/datasets/model/samos.nc

# rename dimensions in baseline and model data to match the cerra dimension names
ncrename -d x,longitude $validation_dir/datasets/baseline/era5_regridded.nc
ncrename -d y,latitude $validation_dir/datasets/baseline/era5_regridded.nc
cdo chname,x,longitude,y,latitude $validation_dir/datasets/baseline/era5_regridded.nc $validation_dir/datasets/baseline/tmp.nc; mv $validation_dir/datasets/baseline/tmp.nc $validation_dir/datasets/baseline/era5_regridded.nc

ncrename -d x,longitude $validation_dir/datasets/model/samos.nc
ncrename -d y,latitude $validation_dir/datasets/model/samos.nc
cdo chname,x,longitude,y,latitude $validation_dir/datasets/model/samos.nc $validation_dir/datasets/model/tmp.nc; mv $validation_dir/datasets/model/tmp.nc $validation_dir/datasets/model/samos.nc

# the validation script cannot handle na values - there are some present on the edges due to the cdo bilinear interpolation in the baseline as well as samos output. clip all data on the edges
ncks -d latitude,5,159 -d longitude,2,237 $validation_dir/datasets/model/samos.nc -O $validation_dir/datasets/model/tmp.nc; mv $validation_dir/datasets/model/tmp.nc $validation_dir/datasets/model/samos.nc
ncks -d latitude,5,159 -d longitude,2,237 $validation_dir/datasets/baseline/era5_regridded.nc -O $validation_dir/datasets/baseline/tmp.nc; mv $validation_dir/datasets/baseline/tmp.nc $validation_dir/datasets/baseline/era5_regridded.nc
ncks -d latitude,5,159 -d longitude,2,237 $validation_dir/datasets/cerra/cerra.nc -O $validation_dir/datasets/cerra/tmp.nc; mv $validation_dir/datasets/cerra/tmp.nc $validation_dir/datasets/cerra/cerra.nc
