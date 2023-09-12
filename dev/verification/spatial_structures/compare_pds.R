#!/usr/bin/env Rscript

##################################################################
#Description    : power spectral density analysis
#Author         : Konrad Mayer
##################################################################

library(stars)
library(purrr)
library(here)
library(glue)
library(stringr)
library(tidyverse)
library(radialpsd)
library(ggfan)

# helpers ----------------------------------------------------------------------
# pad_to_square <- function(m, pad = 0) {
#     dim_m <- dim(m)
#     maxdim <- max(dim_m)
#     out <- matrix(pad, nrow = maxdim, ncol = maxdim)
#     start <- floor((maxdim - dim_m) / 2)
#     end <- start + dim_m
#     start <- start + 1
#     out[start[1]:end[1], start[2]:end[2]] <- m
#     out
# }

truncate_to_square <- function(m) {
    dim_m <- dim(m)
    mindim <- min(dim_m)
    start <- floor((dim_m - mindim) / 2)
    end <- start + mindim
    start <- start + 1
    m[start[1]:end[1], start[2]:end[2]]
}

replace_na <- function(m, replacement = 0) {
    m[is.na(m)] <- replacement
    m
}

do_psd <- function(x) {
    tmp <- replace_na(truncate_to_square(as.matrix(x[[1]])))
    psd <- radial.psd(tmp, plot = FALSE, scaled = FALSE, normalized = FALSE)
    max_dist <- sqrt(sum((ceiling(dim(tmp)/2)-1)^2))
    mutate(psd, wavenumber = wavenumber / max_dist)
}

drop_units <- function(x) {
    tryCatch(
        units::drop_units(x),
        error = function(e) {
            return(x)
        }
    )
}
# main -------------------------------------------------------------------------

dothis <- function(lead_time) {
    lead_time <- str_pad(lead_time, 2, pad = "0")

    # load datasets (crop a bit as SAMOS baseline has some values NA around)
    datasets <- c(
        cerra = "dat/TESTING/PREPROCESSED/CERRA/t2m_cerra_{lead_time}.nc",
        downscaled = "dat/TESTING/SAMOS/postprocessed/samos-postprocessed_{lead_time}.nc",
        era5 = "dat/TESTING/PREPROCESSED/ERA5_regridded/t2m_era5_{lead_time}.nc"
    )
    dat <- map2(datasets, names(datasets), 
        ~setNames(drop_units(read_stars(here(glue(.x)))), .y)[1, 3:238,6:160]) # data gets truncated to get rid of NA values on two of the edges caused by CDO bilinear interpolation

    timesteps <- seq_len(dim(dat[[1]])["time"])
    
    #overall

    psd_overall <- map(dat, 
        ~do_psd(st_apply(.x, c("x", "y"), mean))) %>%
        bind_rows(.id = "model")

    psd_overall %>%
        ggplot(aes(wavenumber, r_spectrum, lty = model)) +
        geom_line() +
        theme_minimal(20) +
        scale_y_log10() + scale_x_log10(sec.axis = sec_axis(trans = ~ (.^-1), name = "wavelength")) + annotation_logticks() +
        labs(y = "Power")

    ggsave(here(glue("plt/PSD/radialPSD_samos_leadtime{lead_time}_overall.pdf")), width = 10, height = 6)


    # loop over timesteps and calculate mean psd
    psd_timesteps <- map(dat, function(ds) {
        map(timesteps, function(ts) do_psd(ds[,,,ts, drop = TRUE]))
    })
    # common dataframe
    plotdat <- psd_timesteps %>%
        map(~bind_rows(setNames(.x, timesteps), .id = "timesteps")) %>%
        bind_rows(.id = "dataset")
        
    # plot
    plotdat %>%
        ggplot(aes(x = wavenumber, y = r_spectrum, color = dataset, group = dataset)) + 
        geom_interval() +
        theme_minimal(20) +
        scale_y_log10() + scale_x_log10(sec.axis = sec_axis(trans = ~ (.^-1), name = "wavelength")) + annotation_logticks() +
        labs(y = "Power")

    ggsave(here(glue("plt/PSD/radialPSD_samos_leadtime{lead_time}_distribution.pdf")), width = 10, height = 6)

    # aggregate to seasons and calculate PSD per season
    seasons <- list(DJF = c(12, 1, 2), MAM = 3:5, JJA = 6:8, SON = 9:11) 

    per_season <- function(months) {
        map(dat, ~do_psd(st_apply(.x %>% filter(lubridate::month(time) %in% months), c("x", "y"), mean))) %>%
           bind_rows(.id = "model")
    }

    psd_season <- map(seasons, per_season)

    psd_season %>%
        bind_rows(.id =  "season") %>%
        ggplot(aes(wavenumber, r_spectrum, color = season, lty = model)) +
        geom_line() +
        theme_minimal(20) +
        scale_y_log10() + scale_x_log10(sec.axis = sec_axis(trans = ~ (.^-1), name = "wavelength")) + annotation_logticks() +
        labs(y = "Power")

    ggsave(here(glue("plt/PSD/radialPSD_samos_leadtime{lead_time}_season.pdf")), width = 10, height = 6)
}

lead_times <- seq(0, 21, by = 3)
walk(lead_times, dothis)

# TODO: PSD needs to be done on projected data, otherwise distance is not valid