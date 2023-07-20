
#!/usr/bin/env bash

##################################################################
#Description    : bilinear interpolation of ERA5 data onto the 
#                 CERRA grid
#Creation Date  : 2023-06-29
#Author         : Konrad Mayer
##################################################################

# find project root (by finding git root) to be able to start the script from
# anywhere within the project, but still avoid absolute paths
projectroot=$(git rev-parse --show-toplevel)
dataset=${1:-"TRAINING"}

# Declare a string array with type
declare -a leadtime=( "00" "03" "06" "09" "12" "15" "18" "21" )

for lt in "${leadtime[@]}"; do
    echo "lead time ${lt}"
    rm /tmp/targetgrid.txt
    # create descriptor file for target grid
    stdbuf -oL cdo griddes dat/TRAINING/RESIDUALS/CERRA/t2m_cerra_${lt}_residuals.nc > /tmp/targetgrid.txt
    # bilinear interpolation
    cdo remapbil,/tmp/targetgrid.txt ${projectroot}/dat/${dataset}/RESIDUALS/ERA5/t2m_era5_${lt}_residuals.nc ${projectroot}/dat/${dataset}/RESIDUALS/ERA5_regridded/t2m_era5_${lt}_residuals.nc
done 
