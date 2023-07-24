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

per_lead_time <- function(lead_time) {
        
    lead_time <- str_pad(lead_time, 2, pad = "0")

    era5 <- read_stars(here(glue("dat/TESTING/PREPROCESSED/ERA5/t2m_era5_{lead_time}.nc"))) %>%
        setNames("era5")
    cerra <- read_stars(here(glue("dat/TESTING/PREPROCESSED/CERRA/t2m_cerra_{lead_time}.nc"))) %>%
        setNames("cerra")
    downscaled <- read_stars(here(glue("dat/TESTING/SAMOS/postprocessed/samos-postprocessed_{lead_time}.nc"))) %>%
        setNames("samos")

    diff_field <- downscaled[0]
    diff_field[["diff_samos_cerra"]] <- downscaled[[1]] - units::drop_units(cerra[[1]])

    dothis <- function(eval_datetime) {
        plts <- map2(
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
        )

        wrap_plots(plts, ncol = 2) + plot_annotation(title = eval_datetime)
        ggsave(here(glue("plt/SAMOS/{format(eval_datetime, '%Y%m%dT%H%M%S')}_testset.pdf")), height = 7, width = 10)

    }

    walk(st_get_dimension_values(downscaled, "time"), dothis)
}

lead_times <- seq(0, 21, by = 3)
walk(lead_times, per_lead_time)
