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
    era5 <- read_stars(here(glue("dat/TESTING/PREPROCESSED/ERA5_regridded/t2m_era5_{lead_time}.nc"))) %>%
        setNames("era5") %>%
        .[1, 3:238,6:160]

    timesteps <- seq_len(dim(cerra)["time"])
    
    #overall

    psd_cerra_overall <- radial.psd(replace_na(truncate_to_square(as.matrix((st_apply(cerra, c("x", "y"), mean)[[1]])))), plot = FALSE, scaled = FALSE, normalized = FALSE)
    psd_downscaled_overall <- radial.psd(replace_na(truncate_to_square(as.matrix((st_apply(downscaled, c("x", "y"), mean)[[1]])))), plot = FALSE, scaled = FALSE, normalized = FALSE)
    psd_era5_overall <- radial.psd(replace_na(truncate_to_square(as.matrix((st_apply(era5, c("x", "y"), mean)[[1]])))), plot = FALSE, scaled = FALSE, normalized = FALSE)
    psd_overall <- bind_rows(list(cerra = psd_cerra_overall, downscaled = psd_downscaled_overall, era5 = psd_era5_overall), .id =  "model")


    psd_overall %>%
        ggplot(aes(wavenumber, r_spectrum, lty = model)) +
        geom_line() +
        theme_minimal() +
        scale_y_log10() + scale_x_log10() + annotation_logticks()
ggsave(here(glue("plt/PSD/radialPSD_samos_leadtime{lead_time}_overall.pdf")))


    # loop over timesteps and calculate mean psd
    psd_cerra <- list()
    psd_ds <- list()
    psd_era5 <- list()

    for (i in seq_along(timesteps)){
        psd_cerra[[i]] <- radial.psd(replace_na(truncate_to_square(as.matrix(units::drop_units(cerra[[1]][,,timesteps[i]])))), plot = FALSE)#, scaled = FALSE, normalized = FALSE)
        psd_ds[[i]] <- radial.psd(replace_na(truncate_to_square(as.matrix(downscaled[[1]][,,timesteps[i]]))), plot = FALSE)#, scaled = FALSE, normalized = FALSE)
        psd_era5[[i]] <- radial.psd(replace_na(truncate_to_square(as.matrix(era5[[1]][,,timesteps[i]]))), plot = FALSE)#, scaled = FALSE, normalized = FALSE)
    }

    # common dataframe
    plotdat <- bind_rows(list(
        cerra = psd_cerra |>
            setNames(timesteps) |>
            bind_rows(.id = "timesteps"),
        samos = psd_ds |>
            setNames(timesteps) |>
            bind_rows(.id = "timesteps"),
        era5 = psd_era5 |>
            setNames(timesteps) |>
            bind_rows(.id = "timesteps")
    ), .id = "dataset")

    # plot
    plotdat %>%
        ggplot(aes(x = wavenumber, y = r_spectrum, color = dataset, group = dataset)) + 
        # stat_smooth(
        #     geom="ribbon",
        #     aes(ymax = after_stat(y) + 1,
        #         ymin = after_stat(y) - 1,
        #         fill = dataset),
        #     alpha = 0.1, linetype = 0
        # ) +
        # geom_smooth(se = FALSE)+
        geom_interval() +
        theme_minimal() +
          scale_y_log10() + scale_x_log10() + annotation_logticks()

    ggsave(here(glue("plt/PSD/radialPSD_samos_leadtime{lead_time}_distribution.pdf")))

    # aggregate to seasons and calculate PSD per season
    seasons <- list(DJF = c(12, 1, 2), MAM = 3:5, JJA = 6:8, SON = 9:11) 

    per_season <- function(months) {
        cerra_season <- st_apply(cerra %>% filter(lubridate::month(time) %in% months), c("x", "y"), mean) 
        downscaled_season <- st_apply(downscaled %>% filter(lubridate::month(time) %in% months), c("x", "y"), mean)
        era5_season <- st_apply(era5 %>% filter(lubridate::month(time) %in% months), c("x", "y"), mean)
        psd_cerra_season <- radial.psd(replace_na(truncate_to_square(as.matrix((cerra_season[[1]])))), plot = FALSE)#, scaled = FALSE, normalized = FALSE)
        psd_downscaled_season <- radial.psd(replace_na(truncate_to_square(as.matrix((downscaled_season[[1]])))), plot = FALSE)#, scaled = FALSE, normalized = FALSE)
        psd_era5_season <- radial.psd(replace_na(truncate_to_square(as.matrix((era5_season[[1]])))), plot = FALSE)#, scaled = FALSE, normalized = FALSE)
        bind_rows(list(cerra = psd_cerra_season, downscaled = psd_downscaled_season, era5 = psd_era5_season), .id =  "model")
    }
    psd_season <- map(seasons, per_season)

    psd_season %>%
        bind_rows(.id =  "season") %>%
        ggplot(aes(wavenumber, r_spectrum, color = season, lty = model)) +
        geom_line() +
        theme_minimal() +
        scale_y_log10() + scale_x_log10() + annotation_logticks()
    ggsave(here(glue("plt/PSD/radialPSD_samos_leadtime{lead_time}_season.pdf")))
}

lead_times <- seq(0, 21, by = 3)
walk(lead_times, dothis)

# TODO: PSD needs to be done on projected data, otherwise distance is not valid