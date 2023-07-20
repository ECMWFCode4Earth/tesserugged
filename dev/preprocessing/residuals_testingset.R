
#!/usr/bin/env Rscript

##################################################################
#Description    : SAMOS residuals for the test dataset
#Creation Date  : 2023-07-20
#Author         : Konrad Mayer
##################################################################

datasets <- "era5" # for the test dataset residuals are likely only needed for the ERA5 data, but the script below is parametrized on the dataset in case we need CERRA residuals as well


library(stars)
library(lubridate)
library(crch)
library(logger)
library(stringr)
library(readr)
library(here)
library(glue)
library(dplyr)
library(purrr)
library(zamg.ncdf)

reshape_results <- function(x, dat) {
    aperm(array(unlist(x), dim = dim(dat)[c(3, 2, 1)]), c(3, 2, 1))
}

dothis <- function(lead_time, dataset) {
    tryCatch(
    {
        log_info("Start calculation for lead time {lead_time}.")
        lead_time <- str_pad(lead_time, 2, pad = "0")

        # load data
        dat <- read_stars(here(glue("dat/TESTING/PREPROCESSED/{toupper(dataset)}/t2m_{tolower(dataset)}_{lead_time}.nc")), proxy = FALSE) %>%
            units::drop_units()
        log_info("Data loaded.")

        # extract time coordinate and derive objects
        timestamps <- st_get_dimension_values(dat, "time")
        year <- year(timestamps)
        yday <- yday(timestamps)

        # model components
        predictors <- tibble(
            sin1 = sin(yday * 2 * pi / 365),
            cos1 = cos(yday * 2 * pi / 365),
            sin2 = sin(yday * 4 * pi / 365),
            cos2 = cos(yday * 4 * pi / 365),
            trend = year - min(year) + yday / max(yday)
        )

        mdls <- read_rds(here(glue("dat/TRAINING/CLIMATOLOGY/{toupper(dataset)}/models/t2m_{tolower(dataset)}_{lead_time}_climatology-models.rds")))
        log_info("Models loaded.")

        # fill predicted mu and sd to stars object
        log_info("Predict mu and sd from climatology models.")
        prediction <- dat[0] #initialize empty stars object with same coordinates as dat
        prediction$mu_modeled <- reshape_results(map(mdls$mdl, ~predict(.x, newdata = predictors, type = "location"), .progress = interactive()), dat)
        prediction$sd_modeled <- reshape_results(map(mdls$mdl, ~predict(.x, newdata = predictors, type = "scale"), .progress = interactive()), dat)

        # st_crs(prediction) <- 4326
        log_info("Predicted mu and sd, based on climatology model")

        
        # calculate residuals
        residuals <- (dat - prediction["mu_modeled"]) / prediction["sd_modeled"]
        st_crs(residuals) <- 4326
        write_stars_ncdf(residuals, here(glue("dat/TESTING/RESIDUALS/{toupper(dataset)}/t2m_{tolower(dataset)}_{lead_time}_residuals.nc")))
        log_info("Residuals saved to disk.")

        log_info("Calculation for lead time {lead_time} was successful.")
    },
        error = function(e) {log_error("Calculation of residuals for lead time {lead_time} failed: "); print(e)}
    )}

doall_dataset <- function(dataset) {
    log_info("START iterations to calculate residuals from {toupper(dataset)} data.")
    lead_times <- seq(0, 21, by = 3)
    walk(lead_times, dothis, dataset = dataset)
    log_info("END iterations to calculate residuals from {toupper(dataset)} data.")
}

walk(datasets, doall_dataset)