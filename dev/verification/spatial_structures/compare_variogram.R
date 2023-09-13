library(tidyverse)
library(stringr)
library(here)
library(stars)
library(glue)
library(geoR)

dothis <- function(lead_time) {
    lead_time = 12# for testing
    lead_time <- str_pad(lead_time, 2, pad = "0")

    # load datasets (crop a bit as SAMOS baseline has some values NA around)
    cerra <- read_stars(here(glue("dat/TESTING/PREPROCESSED/CERRA/t2m_cerra_{lead_time}.nc"))) %>%
        setNames("cerra") %>%
        .[1, 3:238,6:160]
    downscaled <- read_stars(here(glue("dat/TESTING/SAMOS/postprocessed/samos-postprocessed_{lead_time}.nc"))) %>%
        setNames("samos") %>%
        .[1, 3:238,6:160]
 


#  cerra_tibble <- as_tibble(cerra[,,,10]) # how to deal with timesteps? including more results in too big ram use
#  downscaled_tibble <- as_tibble(downscaled[,,,10])
#  variog_cerra = variog(data = cerra_tibble[,4], coords = as.matrix(cerra_tibble[1:2]))
#  variog_downscaled = variog(data = downscaled_tibble[,4], coords = as.matrix(downscaled_tibble[1:2]))

#  plot(variog_cerra)
#  points(variog_downscaled$u, variog_downscaled$v, col = "red")

#  plot(variog_cerra)
#  points(variog_downscaled$u, variog_downscaled$v, col = "red")

 seasons <- list(DJF = c(12, 1, 2), MAM = 3:5, JJA = 6:8, SON = 9:11) 


 # TODO: variogram needs to be done on projected data, otherwise distance is not valid
 per_season <- function(months) {
    cerra_tibble <- as_tibble(st_apply(cerra %>% filter(lubridate::month(time) %in% months), c("x", "y"), mean)) 
    downscaled_tibble <- as_tibble(st_apply(downscaled %>% filter(lubridate::month(time) %in% months), c("x", "y"), mean))
    variog_cerra = variog(data = cerra_tibble[,3], coords = as.matrix(cerra_tibble[1:2]))
    variog_downscaled = variog(data = downscaled_tibble[,3], coords = as.matrix(downscaled_tibble[1:2]))
    tibble(distance = variog_cerra$u, semivar_cerra = variog_cerra$v, semivar_downscaled = variog_downscaled$v)
 }

 variog_seasons <- purrr::map(seasons, per_season) %>% setNames(names(seasons))

 variog_seasons %>%
    bind_rows(.id = "season") %>%
    pivot_longer(-c(1:2), names_to = "model") %>%
    ggplot(aes(distance, value, lty = model, color = season)) +
        geom_line() + 
        theme_minimal()
 ggsave(here(glue("plt/seasonal_variogram_samos_leadtime{lead_time}.pdf")))
}

