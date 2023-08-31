library(here)
library(lubridate)
library(stars)
library(ggplot2)
library(purrr)
library(glue)
library(dplyr)
library(patchwork)
library(metR)
library(stringr)
library(radialpsd)

# helpers
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
# main

per_lead_time <- function(lead_time) {
        
    lead_time <- str_pad(lead_time, 2, pad = "0")

    era5 <- read_stars(here(glue("dat/TESTING/PREPROCESSED/ERA5/t2m_era5_{lead_time}.nc"))) %>%
        setNames("era5")
    cerra <- read_stars(here(glue("dat/TESTING/PREPROCESSED/CERRA/t2m_cerra_{lead_time}.nc"))) %>%
        setNames("cerra")
    downscaled <- read_stars(here(glue("dat/TESTING/SAMOS/postprocessed/samos-postprocessed_{lead_time}.nc"))) %>%
        setNames("samos")

    era5_regridded <- era5 <- read_stars(here(glue("dat/TESTING/PREPROCESSED/ERA5_regridded/t2m_era5_{lead_time}.nc"))) %>% # only needed for PSD
        setNames("era5")

    diff_field <- downscaled[0]
    diff_field[["diff_samos_cerra"]] <- downscaled[[1]] - units::drop_units(cerra[[1]])

    dothis <- function(eval_datetime) {
        plts <- c(map2(
            list(era5, cerra, downscaled, diff_field) %>% map(~filter(.x, time == eval_datetime)),
            list(scale_fill_viridis_c, scale_fill_viridis_c, scale_fill_viridis_c, scale_fill_divergent),
            ~ggplot() +
                geom_stars(data = .x) +
                ggplot2::borders() +
                coord_sf(
                    xlim = range(st_get_dimension_values(era5, "x")),
                    ylim = range(st_get_dimension_values(era5, "y")),
                    clip = "on", expand = FALSE
                    ) +
                .y() +
                theme_minimal()
        ),
        list(map(
            list(era5_regridded, cerra, downscaled) %>% map(~filter(.x %>%.[1, 3:238,6:160], time == eval_datetime)),
            ~{radial.psd(replace_na(truncate_to_square(as.matrix((st_apply(.x, c("x", "y"), mean)[[1]])))), plot = FALSE)}
        ) %>%
            setNames(c("era5", "cerra", "samos")) %>%
            bind_rows(.id = "model") %>%
            ggplot(aes(wavenumber, r_spectrum, lty = model)) +
                geom_line() +
                theme_minimal() +
                scale_y_log10() + annotation_logticks() + scale_x_log10() 
        ))


        wrap_plots(plts[1:4], ncol = 2) / plts[[5]] + plot_annotation(title = eval_datetime)
        ggsave(here(glue("plt/SAMOS/{format(eval_datetime, '%Y%m%dT%H%M%S')}_testset.pdf")), height = 14, width = 10)

    }

    walk(st_get_dimension_values(downscaled, "time"), dothis)
}

lead_times <- seq(0, 21, by = 3)
walk(lead_times, per_lead_time)
