library(ggplot2)
library(tidyverse)
library(nanoparquet)
library(ggdensity)

data = read_parquet("data/shot_data_2425.parquet")

data %>%
  mutate(
    x_error = x_adj_coord - shot_x,
    y_error = y_adj_coord - shot_y
  ) %>%
  ggplot(aes(x=x_error, y=y_error)) +
  geom_hdr() +
  geom_hdr_lines(size=0.75, show.legend=FALSE) +
  geom_hdr_rug(length = unit(.2, "cm")) +
  theme_bw(base_size=12) +
  coord_fixed() +
  # scale_fill_viridis_d(option="magma", begin = .8, end = 0) +
  labs(x="Estimated X Coordinate Error (ft)", y="Estimated Y Coordinate Error (ft)",
        alpha="Probability")

# ggsave("plots/shot_detection/location_error_distribution.png", dpi=500)
