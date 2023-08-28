library(stars)
library(purrr)
library(here)
library(glue)
library(stringr)
library(tidyverse)
library(radialpsd)


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

# main -------------------------------------------------------------------------

dothis <- function(lead_time) {
    lead_time <- str_pad(lead_time, 2, pad = "0")

    # load datasets (crop a bit as SAMOS baseline has some values NA around)
    cerra <- read_stars(here(glue("dat/TESTING/PREPROCESSED/CERRA/t2m_cerra_{lead_time}.nc"))) %>%
        setNames("cerra") %>%
        .[1, 3:238,6:160]
    downscaled <- read_stars(here(glue("dat/TESTING/SAMOS/postprocessed/samos-postprocessed_{lead_time}.nc"))) %>%
        setNames("samos") %>%
        .[1, 3:238,6:160]

    timesteps <- seq_len(dim(cerra)["time"])

    # loop over timesteps and calculate psd
    psd_cerra <- list()
    psd_ds <- list()

    for (i in seq_along(timesteps)){
        psd_cerra[[i]] <- radial.psd(replace_na(truncate_to_square(as.matrix(units::drop_units(cerra[[1]][,,timesteps[i]])))), plot = FALSE)#, scaled = FALSE, normalized = FALSE)
        psd_ds[[i]] <- radial.psd(replace_na(truncate_to_square(as.matrix(downscaled[[1]][,,timesteps[i]]))), plot = FALSE)#, scaled = FALSE, normalized = FALSE)
    }

    # common dataframe
    plotdat <- bind_rows(list(
        cerra = psd_cerra |>
            setNames(timesteps) |>
            bind_rows(.id = "timesteps"),
        samos = psd_ds|>
            setNames(timesteps) |>
            bind_rows(.id = "timesteps")
    ), .id = "dataset")

    # plot
    plotdat %>%
        ggplot(aes(x = log(wavenumber), y = log(r_spectrum), color = dataset)) + 
        stat_smooth(
            geom="ribbon",
            aes(ymax = after_stat(y) + 1,
                ymin = after_stat(y) - 1,
                fill = dataset),
            alpha = 0.1, linetype = 0
        ) +
        geom_smooth(se = FALSE)+
        theme_minimal()

    ggsave(here(glue("plt/PSD/radialPSD_samos_leadtime{lead_time}.pdf")))
}

lead_times <- seq(0, 21, by = 3)
walk(lead_times, dothis)
