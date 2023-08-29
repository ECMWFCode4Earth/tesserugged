#!/usr/bin/env Rscript

##################################################################
#Description    : add climatology to standardized anomalies
#Creation Date  : 2023-07-20
#Author         : Konrad Mayer
##################################################################

# use from the commandline as follows: first argument is a path template (path with lead time parametrized as "{lead_time}") for modelled residuals, second argument a path template for the output (again, parametrize lead time), third argument ist the variable name as contained within the modelled residuals to be used
# for samos the call would look like this:
# ./dev/postprocessing/reapply_climatology.R "dat/TESTING/SAMOS/predictions/samos-predictions_{lead_time}.nc" "dat/TESTING/SAMOS/postprocessed/samos-postprocessed_{lead_time}.nc" "mu_samos"

library(stars)
library(here)
library(glue)
library(readr)
library(lubridate)
library(tibble)
library(purrr)
library(crch)
library(zamg.ncdf)
library(logger)
library(stringr)

# commandline input
args <- commandArgs(trailingOnly=TRUE)
modeled_residuals_template <- args[[1]] 
log_info("Use template for model residuals: {modeled_residuals_template}")
varname <- args[[3]]
if(!is.null(varname)) {log_info("Variable {varname} selected.")}

out_template <- args[[2]]
log_info("Write output to: {out_template}")

mdl_template <- "dat/TRAINING/CLIMATOLOGY/CERRA/models/t2m_cerra_{lead_time}_climatology-models.rds"
log_info("Models taken from {mdl_template}")

# helpers
reshape_results <- function(x, dat) {
    aperm(array(unlist(x), dim = dim(dat)[c(3, 2, 1)]), c(3, 2, 1))
}

# main functions
dothis_perleadtime <- function(lead_time, modeled_residuals_template, out_template, varname = NULL) {

    lead_time <- str_pad(lead_time, 2, pad = "0")
    log_info("start for lead time {lead_time}")

    modeled_residuals <- read_stars(here(glue(modeled_residuals_template)))
    if(!is.null(varname)) {
        # subset variable if given
        modeled_residuals <- modeled_residuals[varname]
    }

    mdls <- read_rds(here(glue(mdl_template)))

    # extract new time steps
    timestamps <- st_get_dimension_values(modeled_residuals, "time")
    year <- year(timestamps)
    yday <- yday(timestamps)

    # predictors for new timesteps
    predictors <- tibble(
        sin1 = sin(yday * 2 * pi / 365),
        cos1 = cos(yday * 2 * pi / 365),
        sin2 = sin(yday * 4 * pi / 365),
        cos2 = cos(yday * 4 * pi / 365),
        trend = year - min(year) + yday / max(yday)
    )

    # dataframe with row and column indizes
    climatology <- modeled_residuals[0] #initialize empty stars object with same coordinates as dat
    climatology$mu_modeled <- reshape_results(map(mdls$mdl, ~predict(.x, newdata = predictors, type = "location"), .progress = interactive()), climatology)
    climatology$sd_modeled <- reshape_results(map(mdls$mdl, ~predict(.x, newdata = predictors, type = "scale"), .progress = interactive()), climatology)

    # reapply to standardized anomalies
    out <- modeled_residuals * climatology$sd_modeled + climatology$mu_modeled

    # save to disk
    write_stars_ncdf(out, here(glue(out_template)))

    log_info("finished for lead time {lead_time}.")
}

doall <- function(modeled_residuals_template, out_template) {
    log_info("START iterating over lead times")
    lead_times <- seq(0, 21, by = 3)
    walk(lead_times, ~dothis_perleadtime(.x, modeled_residuals_template, out_template, varname))
    log_info("END iterating over lead times")
}

doall(modeled_residuals_template, out_template)