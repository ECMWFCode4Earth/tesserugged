#!/usr/bin/env Rscript

##################################################################
#Description    : Fit SAMOS model per lead time
#Creation Date  : 2023-07-03
#Authors        : Konrad Mayer and Markus Dabernig
##################################################################

set.seed(42) # set seed for reproducibility

library(tidyverse)
library(here)
library(lubridate)
library(stars)
library(glue)
library(crch)
library(logger)

# land sea mask 
lsm <- read_stars(here( "dat/PREPROCESSED/cerra_lsm.nc"), sub = "lsm")
lsm <- as_tibble(lsm) %>%
    mutate(lsm = as.logical(lsm)) %>% # encode as logical, TRUE when land
    select(-time) %>%
    mutate(x = round(x, 2), y = round(y, 2)) # needed, as otherwide floating point inprecsion prevents join


# helpers

fix_attributes <- function(x) {
    # workaround as for some reason blocksizes attribute hinders combining objects to a common stars object
    attributes(st_dimensions(x))$raster$blocksizes <- NULL;
    x
}

striptease <- function(fit) {
    # reduce object size of stored models
    attr(fit$terms$location, '.Environment') <- attr(fit$terms$scale, '.Environment') <- attr(fit$terms$full, '.Environment') <- NULL
    fit$residuals <- fit$fitted.values <- fit$model <- fit$link$scale$dmu.deta <- fit$control$start <- fit$formula <- NULL
    fit
}

# load residuals for training data (ERA5 already regridded)
dothis <- function(lead_time, sample_frac = 0.1) {
    lead_time <- str_pad(lead_time, 2, pad = "0")

    log_info("load data for lead time {lead_time}")
    era5 <- fix_attributes(read_stars(here(glue("dat/RESIDUALS/ERA5_regridded/t2m_era5_{lead_time}_residuals.nc")), proxy = FALSE))
    cerra <- fix_attributes(read_stars(here(glue("dat/RESIDUALS/CERRA/t2m_cerra_{lead_time}_residuals.nc")), proxy = FALSE))

    st_crs(era5) <- st_crs(cerra)

    # composite dataset
    dat <- c(era5, cerra) %>%
        set_names("era5", "cerra")

    log_info("start data wrangling. use only a fraction of {sample_frac} of the data to fit model.")
    modeldat <- dat %>%
        as_tibble() %>%
        sample_frac(sample_frac) %>% # use only a part of the complete data
        mutate(x = round(x, 2), y = round(y, 2)) %>% # needed, as otherwide floating point inprecsion prevents join
        left_join(lsm, by = join_by(x, y))
    
    # release memory
    rm(dat, era5, cerra); gc()

    # fit model
    log_info("fit model on {nrow(modeldat)} observations.")
    fit <- striptease(crch(cerra ~ era5 * lsm | era5 * lsm, data = modeldat)) # does not work for full data due to memory issues
    log_info("save model to disk.")
    write_rds(fit, here(glue("dat/SAMOS/models/samos-model_{lead_time}.rds")))

}

log_info("START iterations over lead times to fit SAMOS models")
lead_times <- seq(0, 21, by = 3)
walk(lead_times, dothis)
log_info("END iterations over lead times to fit SAMOS models")