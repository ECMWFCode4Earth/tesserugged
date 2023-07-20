#!/usr/bin/env bash

##################################################################
#Description    : Rechunk data from montly files to partitions per
#                 hour of the day
#Creation Date  : 2023-06-20
#Author         : Konrad Mayer
##################################################################

# find project root (by finding git root) to be able to start the script from
# anywhere within the project, but still avoid absolute paths
projectroot=$(git rev-parse --show-toplevel)

# # TRAINING DATA
# # split ERA5 data by hour
# cdo splithour -sellonlatbox,-6.86,5.11,44.96,36.99 -select,name=t2m ${projectroot}/dat/TRAINING/RAW/ERA5/t2m_era5_[!lsm]*_025deg.nc ${projectroot}/dat/TRAINING/PREPROCESSED/ERA5/t2m_era5_

# # split CERRA data by hour
# cdo splithour -sellonlatbox,-6.86,5.11,44.96,36.99 -select,name=t2m ${projectroot}/dat/TRAINING/RAW/CERRA/t2m_cerra_[!lsm]*_005deg.nc ${projectroot}/dat/TRAINING/PREPROCESSED/CERRA/t2m_cerra_


# TEST DATA
# split ERA5 data by hour
cdo splithour -sellonlatbox,-6.86,5.11,44.96,36.99 -select,name=t2m ${projectroot}/dat/TESTING/RAW/ERA5/t2m_era5_[!lsm]*_025deg.nc ${projectroot}/dat/TESTING/PREPROCESSED/ERA5/t2m_era5_

# split CERRA data by hour
cdo splithour -sellonlatbox,-6.86,5.11,44.96,36.99 -select,name=t2m ${projectroot}/dat/TESTING/RAW/CERRA/t2m_cerra_[!lsm]*_005deg.nc ${projectroot}/dat/TESTING/PREPROCESSED/CERRA/t2m_cerra_