#!/usr/bin/env bash

##################################################################
#Description    : crop CERRA land-sea-mask to subdomain
#Creation Date  : 2023-07-03
#Author         : Konrad Mayer
##################################################################

# find project root (by finding git root) to be able to start the script from
# anywhere within the project, but still avoid absolute paths
projectroot=$(git rev-parse --show-toplevel)

# crop
cdo sellonlatbox,-6.86,5.11,44.96,36.99 ${projectroot}/dat/TRAINING/RAW/CERRA/t2m_cerra_lsm_orog_201801_005deg.nc ${projectroot}/dat/TRAINING/PREPROCESSED/cerra_lsm.nc
