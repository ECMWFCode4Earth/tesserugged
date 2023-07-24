#!/usr/bin/env Rscript

##################################################################
#Description    : SAMOS predictions for the test dataset
#Creation Date  : 2023-07-20
#Author         : Konrad Mayer
##################################################################

library(stars)
library(glue)
library(logger)
library(stringr)
library(here)
library(readr)
library(dplyr)
library(crch)
library(zamg.ncdf)
library(purrr)

# land sea mask 
lsm <- read_stars(here( "dat/TRAINING/PREPROCESSED/cerra_lsm.nc"), sub = "lsm")
lsm <- as_tibble(lsm) %>%
    mutate(lsm = as.logical(lsm)) %>% # encode as logical, TRUE when land
    select(-time) %>%
    mutate(x = round(x, 2), y = round(y, 2)) # needed, as otherwide floating point inprecsion prevents join

# helpers
reshape_results <- function(x, dat) {
    array(unlist(x), dim = dim(dat))
}

# main function
dothis <- function(lead_time) {

        log_info("Start calculation for lead time {lead_time}.")
        lead_time <- str_pad(lead_time, 2, pad = "0")

        # load data
        dat <- read_stars(here(glue("dat/TESTING/RESIDUALS/ERA5_regridded/t2m_era5_{lead_time}_residuals.nc")), proxy = FALSE) %>%
            setNames("era5")
        log_info("Data loaded.")

        # load SAMOS model
        mdl <- read_rds(here(glue("dat/TRAINING/SAMOS/models/samos-model_{lead_time}.rds")))
        log_info("SAMOS model loaded.")

        # data wrangling
        newdat <- dat %>%
            as_tibble() %>%
            mutate(x = round(x, 2), y = round(y, 2)) %>% # needed, as otherwide floating point inprecsion prevents join
            left_join(lsm, by = join_by(x, y))

        out <- dat[0]
        st_crs(out) <- 4326
        out$mu_samos <- reshape_results(predict(mdl, newdata = newdat, type = "location"), out)
        out$sd_samos <- reshape_results(predict(mdl, newdata = newdat, type = "scale"), out)

        write_stars_ncdf(out, here(glue("dat/TESTING/SAMOS/predictions/samos-predictions_{lead_time}.nc")))
        log_info("Predictions for {lead_time} saved to disk.")
}

log_info("START SAMOS predictions for test data")
lead_times <- seq(0, 21, by = 3)
walk(lead_times, dothis)
log_info("END SAMOS predictions for test data")
