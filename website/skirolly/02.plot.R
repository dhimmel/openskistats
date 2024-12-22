library(arrow)
library(dplyr)
library(ggplot2)
library(yaml)
library(patchwork)
library(cowplot)

img_data_dir <- "../images/data"
img_dir <- "../images"
colors <- c("#f07178", "#004B59", "#FFC857", "#36B37E", "#FF8C42", "#F4F1E9", "#8A9393", "#2A2D34")
my_font <- "Helvetica"

## Plot functions ------------------------------------------------
bg_transparent <- function() {
  theme_minimal(base_family = my_font) +
    theme(
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      axis.title = element_blank(),
      axis.text = element_blank(),
      legend.position = "none",
    )
}
plot_rose <- function(dat, ski_area_name, size_title = 24, size_x = 20, highlight = FALSE, labels = NULL, type = NULL, hemi = NULL, ngr = 32, vert = 0) {
  y_break <- max(dat$bin_count)
  p <- dat |>
    ggplot() +
    aes(x = bin_center, y = bin_count) +
    coord_radial(start = -pi / ngr, expand = FALSE) +
    # coord_polar(start = -pi / 32) +
    scale_x_continuous(
      breaks = seq(0, 270, 90),
      labels = labels
    ) +
    scale_y_sqrt( # scaled by area
      breaks = y_break
    ) +
    labs(title = ski_area_name) +
    bg_transparent() +
    theme(
      panel.grid = element_line(color = "grey60"),
      axis.text.x = element_text(size = size_x, color = "#EBEBEB"),
      plot.title = element_text(hjust = 0.5, size = size_title, color = "#EBEBEB")
    ) +
    NULL
  if (!is.null(type)) {
    p <- p + 
      theme(
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(linewidth = vert)
      )
    if (type == "all"){
      p <- p +
        geom_col(fill = "#f07178", width = 360 / ngr)
    }
    if (type == "us") {
      p <- p +
        geom_col(fill = "#f07178", width = 360 / ngr, color = "#EBEBEB", linewidth = 0.2) +
        annotate(
          geom = "label",
          label = ski_area_name,
          x = 180, y = max(dat$bin_count),
          hjust = 0.5,
          alpha = 0.5,
          fill = "#002B36",
          size = size_title,
          family = my_font,
          color = "#EBEBEB"
        ) +
        theme(
          # plot.margin=grid::unit(c(-35,-35, -35, -35), "mm"),
          panel.background = element_rect(fill = "#002B36"),
          plot.title = element_blank()
        )
    }
  } else if (!is.null(hemi)) {
    p <- p +
      geom_col(color = "#EBEBEB", width = 360 / ngr, aes(fill = color)) +
      scale_fill_manual(values = hemi)
  } else if (!highlight) {
    p <- p + geom_col(color = "#EBEBEB", width = 360 / ngr, fill = "#f07178")
  } else {
    p <- p +
      geom_col(color = "#EBEBEB", width = 360 / ngr, aes(fill = color)) +
      scale_fill_identity()
  }
  p
}

rose_empty <- function(a, size_label) {
  data.frame(x = c(0, 360), y = c(1, 1)) |>
    ggplot() +
    geom_col(fill = "#f07178", alpha = 0) +
    aes(x = x, y = y) +
    coord_radial() +
    scale_y_sqrt( # scaled by area
      breaks = 1
    ) +
    annotate(
      geom = "label",
      label = a,
      x = 180,
      y = 1,
      hjust = 0.5,
      alpha = 0.5,
      fill = "#002B36",
      size = size_label,
      family = my_font,
      color = "#EBEBEB"
    ) +
    bg_transparent() +
    theme(
      panel.background = element_rect(fill = "#002B36"),
      plot.background = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
    )
}

## Data ------------------------------------------------
x1 <- -72.095
y1 <- 43.7775
x2 <- -72.101
y2 <- 43.7900
m <- (y2 - y1) / (x2 - x1)
b <- y1 - m * x1

dart_url <- "https://github.com/user-attachments/assets/1a02ca26-7034-4d87-bc0c-01f6bed997f7"
# download.file(dart_url, destfile = file.path(img_dir, "dartmouth.png"))
# dartmouth_img <- png::readPNG(file.path(img_dir, "dartmouth.png"), native = TRUE)
bearings_ls <- readRDS(file.path(img_data_dir, "bearings_48_ls.rds"))
dartmouth_segs <- read_parquet(file.path(img_data_dir, "dartmouth_segs.parquet"))
hemi <- read_parquet(file.path(img_data_dir, "hemisphere_roses.parquet")) |>
  tidyr::unnest(bearings) 

dart <- read_parquet(file.path(img_data_dir, "dartmouth_runs.parquet")) |>
  group_by(run_id) |>
  mutate(winslow = (m * longitude) + b < latitude) |>
  arrange(index)

dartmouth <- bearings_ls[["Dartmouth Skiway"]]
# whaleback <- bearings_ls[["Whaleback Mountain"]]
# killington <- bearings_ls[["Killington Resort"]]


n_groups <- 32 # number of spokes
x_range <- c(-72.1065, -72.08635)
y_range <- c(43.77828, 43.7899)
length_x <- diff(x_range) # Length in x-direction
length_y <- diff(y_range) # Length in y-direction

## Plots ----
fig_height <- 3.75
fig_width <- fig_height*703/503#865/452

# desired_yx_ratio <- dim(dartmouth_img)[1] / dim(dartmouth_img)[2]
# desired_yx_ratio <- 1399/2356
# desired_yx_ratio <- 1429 / 2768
desired_yx_ratio <- 1420/1897
ratio <- (length_x / length_y) * desired_yx_ratio
coord_dartmouth <- coord_fixed(
  xlim = x_range,
  ylim = y_range,
  ratio = ratio
)

dots_only <- dart |>
  ggplot() +
  aes(x = longitude, y = latitude, color = winslow) +
  scale_color_manual(values = c("#FF8C42", "#36B37E"), guide = "none") +
  geom_point(size = 0.5) +
  coord_dartmouth +
  bg_transparent() +
  theme(
    panel.border = element_blank(),
  )

dots_overlay <- ggimage::ggbackground(
  dots_only + theme(panel.grid = element_blank()),
  dart_url
)

ggsave(
  file.path(img_dir, "dots_overlay.png"), 
  dots_overlay,
  width = fig_width,
  height = fig_height,
  dpi = 300
)

align_dots <- function(p){
  ggplot(data.frame(x = 0:1, y = 0:1), aes(x = x, y = y)) + 
  ggimage::geom_subview(subview = p, width = Inf, height = Inf, x = 0.5, y = 0.5) +
  ggimage::theme_nothing() 
}
ggsave(
  file.path(img_dir, "dots_only.png"), 
  align_dots(dots_only),
  width = fig_width, height = fig_height, dpi = 300
)

segments_plot <- ggplot(dartmouth_segs) +
  geom_segment(
    color = "#f07178",
    aes(x = x, y = y, xend = xend, yend = yend),
    arrow = arrow(type = "closed", length = unit(0.1, "cm"))
  ) +
  coord_dartmouth +
  bg_transparent()

ggsave(
  file.path(img_dir, "segments_plot.png"), 
  align_dots(segments_plot),
  width = fig_width, height = fig_height, dpi = 300
)

dartmouth_rose <- plot_rose(dartmouth, "", labels = c("N", "E", "S", "W"))
ggsave(
  file.path(img_dir, "dartmouth_rose.png"), 
  dartmouth_rose,
  width = 5, height = 5, dpi = 300
)

rose_nwbw <- dartmouth |>
  mutate(color = if_else((row_number()) != 28, "#004B59", "#f07178")) |> 
  plot_rose("", labels = c("N", "E", "S", "W"), highlight = TRUE) +
  geom_text(x = 298, y = 26.3, label = "NWbW", color = "#f07178", size = 6)
rose_nwbw
ggsave(
  file.path(img_dir, "rose_nwbw.png"), 
  rose_nwbw,
  width = 5, height = 5, dpi = 300
)

segments_highlight_nwbw <- dartmouth_segs |>
  mutate(nwbw = group == 28) |>
  ggplot() +
  aes(x = x, y = y, xend = xend, yend = yend, color = nwbw) +
  geom_segment(arrow = arrow(type = "open", length = unit(0.1, "cm"))) +
  scale_color_manual(values = c("#004B59", "#f07178"), guide = "none") +
  coord_dartmouth +
  bg_transparent() +
  theme(
    panel.grid = element_blank(),
  ) 

ggsave(
  file.path(img_dir, "segments_highlight_nwbw.png"), 
  segments_highlight_nwbw,
  width = fig_width, height = fig_height, dpi = 300
)

segments_highlight_nne <- dartmouth_segs |>
  mutate(nne = group == 3) |>
  ggplot() +
  aes(x = x, y = y, xend = xend, yend = yend, color = nne) +
  geom_segment(arrow = arrow(type = "open", length = unit(0.1, "cm"))) +
  scale_color_manual(values = c("#004B59", "#f07178"), guide = "none") +
  coord_dartmouth +
  bg_transparent() +
  theme(
    panel.grid = element_blank(),
  )

ggsave(
  file.path(img_dir, "segments_highlight_nne.png"), 
  segments_highlight_nne,
  width = fig_width, height = fig_height, dpi = 300
)


rose_nne <- dartmouth |>
  mutate(color = if_else((row_number()) != 3, "#004B59", "#f07178")) |>
  plot_rose("", labels = c("N", "E", "S", "W"), highlight = TRUE) +
  geom_text(x = 22, y = 29, label = "NNE", color = "#f07178", size = 6)

ggsave(
  file.path(img_dir, "rose_nne.png"), 
  rose_nne,
  width = 5, height = 5, dpi = 300
)

all_roses <- cowplot::plot_grid(
  plotlist = purrr::map2(
    bearings_ls,
    names(bearings_ls),
    plot_rose,
    size_x = 7,
    size_title = 9,
    vert = 0.1,
    type = "all"
  )
)

ggsave(
  file.path(img_dir, "all_roses.png"), 
  all_roses,
  width = 8, height = 6, dpi = 600
)

north <- hemi |> 
  filter(num_bins == n_groups, hemisphere == "north") |> 
  plot_rose("Northern hemisphere", labels = c("N", "E", "S", "W"))

south <- hemi |> 
  filter(num_bins == n_groups, hemisphere == "south") |> 
  plot_rose("Southern hemisphere", labels = c("N", "E", "S", "W"))

ggsave(
  file.path(img_dir, "hemisphere.png"), 
  cowplot::plot_grid(north, south, ncol = 2),
  width = 12, height = 6, dpi = 300
)

## Region -----------------------------------------------------------
ngroups_us <- 16

max_thickness <- 6
min_thickness <- 0.2
size_label <- 12
water_mark <- "OpenSkiStats.org\nLicense: CC BY 4.0"
region_raw <- arrow::read_parquet(file.path(img_data_dir, "region_roses.parquet"))
verts <- region_raw |>
  select(region, combined_vertical) |>
  mutate(thickness = sqrt(combined_vertical / max(combined_vertical)) * max_thickness + min_thickness, .keep = "unused") |>
  tibble::deframe()
n_ski_areas <- sum(region_raw$ski_areas_count)

region <- region_raw |>
  dplyr::select(region, bearings) |>
  tibble::deframe() |>
  lapply(
    \(x) dplyr::filter(x, num_bins == ngroups_us) |>
      dplyr::mutate(
        color = dplyr::if_else(bin_index == 2, "#f07178", "#004B59"),
      )
  )

layout <- c(
  area(1, 1),
  area(1, 11),
  area(2, 10, 2, 10), area(2, 11),
  area(3, 1), area(3, 2), area(3, 3), area(3, 4), area(3, 5),
  area(3, 7), area(3, 9), area(3, 10), area(3, 11),
  area(4, 1), area(4, 2), area(4, 3), area(4, 4), area(4, 5),
  area(4, 6), area(4, 7), area(4, 8), area(4, 9), area(4, 10),
  area(5, 1), area(5, 2), area(5, 3), area(5, 4), area(5, 5),
  area(5, 6), area(5, 7), area(5, 8), area(5, 9), area(5, 10),
  area(6, 2), area(6, 3), area(6, 4), area(6, 5), area(6, 6),
  area(6, 7), area(6, 8), area(6, 9), area(6, 10),
  area(7, 3), area(7, 4), area(7, 5), area(7, 6), area(7, 7), area(7, 8),
  area(8, 1), area(8, 3), area(8, 9)
)

state_names <- c(
  "AK",
  "ME", "VT", "NH", "WA", "ID", "MT", "ND", "MN", "MI", "NY", "MA", "RI",
  "OR", "UT", "WY", "SD", "IA", "WI", "OH", "PA", "NJ", "CT",
  "CA", "NV", "CO", "NE", "IL", "IN", "WV", "VA", "MD", "DE",
  "AZ", "NM", "KS", "MO", "KY", "TN", "SC", "NC", "DC",
  "OK", "LA", "AR", "MS", "AL", "GA",
  "HI", "TX", "FL"
)

state_map <- setNames(state.abb, state.name)
all_regions <- list(region, state_map[names(region)], vert = verts) |> 
  purrr::pmap(
    plot_rose,
    size_title = size_label,
    ngr = ngroups_us,
    type = "us"
  )
all_regions[[1]]
names(all_regions) <- state_map[names(region)]

state_plots <- lapply(state_names, rose_empty, size_label = size_label) |>
  setNames(state_names)

skiable_states <- names(all_regions)
for (i in skiable_states) {
  state_plots[[i]] <- all_regions[[i]]
}
state_plots <- state_plots[state_names]
# state_plots[[1]]
map_plots <- wrap_plots(state_plots, design = layout) &
  plot_annotation(theme = theme(
    panel.background = element_rect(fill = "#004B59", colour = NA),
    plot.background = element_rect(fill = "#004B59", colour = NA)
  ))

# sysfonts::font_add_google("Nunito Sans")
# showtext::showtext_auto()

title <-
  ggplot() +
  theme_void() +
  coord_cartesian(ylim = c(-2, 0.7)) +
  annotate(
    "text",
    x = 0, y = 0,
    label = "Which way do you ski?", size = 32,
    color = "#EBEBEB",
    family = my_font
  ) +
  annotate(
    "text",
    x = 0, y = -0.9,
    label = sprintf("Orientations of %s US ski areas", n_ski_areas),
    size = 18,
    color = "#EBEBEB",
    family = my_font
  )

layout_title <- c(area(1, 3, 2, 9), layout)
map_plots <- c(list(title = title), state_plots) |>
  wrap_plots(design = layout_title) &
  plot_annotation(theme = theme(
    panel.background = element_rect(fill = "#004B59", colour = NA),
    plot.background = element_rect(fill = "#004B59", colour = NA)
  ))

to_publish <- ggdraw(map_plots) +
  draw_label(
    x = 0.985,
    y = 0.043,
    size = 28,
    hjust = 1,
    color = "grey60",
    fontfamily = my_font,
    lineheight = 1.6,
    label = water_mark
  ) +
  draw_label(
    x = 0.165,
    y = 0.78,
    size = 28,
    hjust = 1,
    color = "grey60",
    fontfamily = my_font,
    label = "border proportional to\ncombined vertical drop"
  ) +
  draw_line(
    x = c(0.185, 0.21),
    y = c(0.768, 0.726),
    color = "grey60", size = 2
  ) +
  draw_line(
    x = c(0.17, 0.185),
    y = c(0.768, 0.768),
    color = "grey60", size = 2
  ) +
  NULL

ggsave(
  file.path(img_dir, "us_roses.png"),
  to_publish,
  width = 11 * 3, height = 8 * 3, units = "in",
  dpi = 300
)
