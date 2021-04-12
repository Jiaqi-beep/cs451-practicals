library(tidyverse)
library(sf)
library(lubridate)
library(furrr)


fishing_vessels <- read_csv("F:/gfw_data/fishing-vessels-v2.csv")
fishing_vessels <- fishing_vessels %>% filter(!is.na(mmsi))

eez <- read_sf("F:/gfw_data/All_eez/all_eez_s.shp")

west_africa_eez <- eez %>%
    filter(Name %in% c("Angola", "République du Congo", "Gabon",
                       "Sao Tome and Principe", "Togo", "Liberia",
                       "Sierra Leone", "Senegal", "Gambia", "Mauritania",
                       "Cape Verde", "Equatorial Guinea", "Benin",
                       "Nigeria / Sao Tome and Principe",
                       "Cameroon", "Ghana", "Ivory Coast",
                       "Nigeria", "Guinea Bissau", "Guinea") )

west_africa_eez <- west_africa_eez[,-c(3:11)]
eez_sf <- st_as_sf(west_africa_eez)

plot(west_africa_eez)

data_dir <- 'F:/gfw_data/fishing_effort_bymmsi/'

# Create dataframe of filenames dates and filter to date range of interest
effort_files <- tibble(
    file = list.files(paste0(data_dir, 'mmsi-daily-csvs-10-v2-2017'), 
                      pattern = '.csv', recursive = T, full.names = T),
    date = ymd(str_extract(file, 
                           pattern = '[[:digit:]]{4}-[[:digit:]]{2}-[[:digit:]]{2}')))

plan(multisession)
effort_df <- furrr::future_map_dfr(effort_files$file, .f = read_csv)

effort_df <- effort_df %>%
    filter(!fishing_hours == 0) %>%
    filter(cell_ll_lon > -29.7 & cell_ll_lon < 15) %>%
    filter(cell_ll_lat > -19 & cell_ll_lat < 22)

# convert effort dataset into sf
effort_sf <- st_as_sf(effort_df, coords = c("cell_ll_lon", "cell_ll_lat"), 
                      crs = 4326) %>%
    st_cast("POINT")

effort_trunc <- effort_sf[west_africa_eez,]

effort_joined <- effort_trunc %>%
    left_join(fishing_vessels, by = "mmsi")

write_csv(effort_joined,"F:/Machine Learning/course project/Data/2017_West_Africa.csv")


count<-effort_joined %>%
    group_by(flag_gfw) %>%
    summarize(count = n()) %>%
    arrange(-count)

count2<-effort_joined %>%
    group_by(flag_gfw) %>%
    summarize(count = sum(fishing_hours)) %>%
    arrange(-count)

ggplot() +
    geom_sf(data = effort_joined)

ggplot()+
    geom_sf(data = west_africa_eez)

library(raster)


dst_shore <- raster("F:/gfw_data/distance-from-shore-v1/distance-from-shore.tif")
v <- extract(dst_shore, effort_joined)

joined <- cbind(effort_joined, v, make.row.names = TRUE)



small <- joined %>%
    filter(v < 11.1)


count3<-small %>%
    group_by(flag_gfw) %>%
    summarize(count = sum(fishing_hours)) %>%
    arrange(-count)

ggplot()+
    geom_sf(data = small,
            aes(color = flag_gfw))

