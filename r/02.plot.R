library(ggplot2)
# so that we don't have to do patchwork::area a zillion times
area <- patchwork::area

img_data_dir <- "../data/images/data"
img_dir <- "../data/images"
colors <- c("#f07178", "#004B59", "#FFC857", "#36B37E", "#FF8C42", "#F4F1E9", "#8A9393", "#2A2D34")

rose_pink <- "#f07178"
dark_pink <- "#d33c44"
text_teal <- "#004B59"
text_light <- "#EBEBEB"
dark_green <- "#002B36"

my_font <- "Alegreya Sans"
sysfonts::font_add_google(my_font)
showtext::showtext_auto()
showtext::showtext_opts(dpi = 300)

# for US map:
ngroups_us <- 16
max_thickness <- 6
min_thickness <- 0.2
size_label <- 12

dark_pal <- list(
  title_color = text_light,
  anno_color = "grey80",
  circle_color = "grey60",
  canvas = text_teal,
  circle_fill = dark_green,
  size_label = size_label,
  my_font = my_font
)

light_pal <- list(
  title_color = "grey10",
  anno_color = "grey30",
  circle_color = "grey50",
  canvas = "white",
  circle_fill = "grey90",
  size_label = size_label,
  my_font = my_font
)

my_aesthetics <- list(light = light_pal, dark = dark_pal)


## Plot functions ------------------------------------------------
#' Make transparent background
#'
#' @return ggplot2 theme with transparent background
#' @export
#'
#' @examples
#' ggplot(mtcars) +
#'   aes(mpg, wt) +
#'   geom_point() +
#'   bg_transparent()
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
#' Plot a ski rose
#'
#' @param dat Data frame of bearings with columns `bin_center` and `bin_count`.
#' @param ski_area_name Character. Name of the ski area.
#' @param size_title Numeric. Size of the title.
#' @param size_x Numeric. Size of the x-axis labels.
#' @param labels Character vector. Labels for the x-axis (around the circle).
#' Defaults to `c("N", "E", "S", "W")`.
#' @param highlight Logical. If `TRUE`, highlight one petal of the rose.
#' @param type Character. Type of rose plot.
#'  - `NULL`: a standard rose
#'  - `"all"`: a minimal rose for a randomly selected ski area
#'  - `"us"`: a rose for a US state
#' @param ngr Number of bearing groups. Defaults to 32.
#' @param vert Numeric. Width of the spokes. Defaults to 0.
#' @param aesthetics List. Aesthetics for the plot. Including `title_color`,
#' `anno_color`, `circle_color`, `canvas`, `circle_fill`, `size_label`, and
#' `my_font`.
#' @param empty_rose Logical. If `TRUE`, plot an empty rose.
#'
#' @return ggplot2 object of a ski rose.
plot_rose <- function(
    dat, ski_area_name = "", size_title = 18, size_x = 24,
    labels = c("N", "E", "S", "W"), highlight = FALSE, type = NULL,
    ngr = 32, vert = 0, aesthetics = NULL, empty_rose = FALSE) {
  y_break <- max(dat$bin_count)
  p <- dat |>
    ggplot() +
    aes(x = bin_center, y = bin_count) +
    coord_radial(start = -pi / ngr, expand = FALSE) +
    # coord_polar(start = -pi / 32) +
    scale_x_continuous(breaks = seq(0, 270, 90), labels = labels) +
    scale_y_sqrt(breaks = y_break) + # scaled by area
    labs(title = ski_area_name) +
    theme(
      panel.grid = element_line(color = "grey60"),
      axis.text.x = element_text(size = size_x, color = text_light),
      plot.title = element_text(hjust = 0.5, size = size_title, color = text_light)
    )

  if (highlight) {
    return(
      p +
        geom_col(color = text_light, width = 360 / ngr, aes(fill = color)) +
        scale_fill_identity()
    )
  }

  if (is.null(type)) { # standard rose
    return(p + geom_col(color = text_light, width = 360 / ngr, fill = rose_pink))
  }

  p <- p +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(linewidth = vert, color = aesthetics$circle_color)
    )
  if (type == "all") {
    return(p + geom_col(fill = rose_pink))
  }

  # type == "us"
  if (!empty_rose) {
    p <- p + geom_col(fill = rose_pink, width = 360 / ngr, color = text_light, linewidth = 0.2)
  } else {
    p <- p + geom_col(fill = NA)
  }
  p <- p +
    annotate(
      geom = "label",
      label = ski_area_name,
      x = 180, y = max(dat$bin_count),
      hjust = 0.5,
      alpha = 0.8,
      fill = aesthetics$circle_fill,
      size = aesthetics$size_label,
      family = aesthetics$my_font,
      color = aesthetics$title_color
    ) +
    theme(
      # plot.margin=grid::unit(c(-35,-35, -35, -35), "mm"),
      panel.background = element_rect(fill = aesthetics$circle_fill),
      plot.title = element_blank()
    )
  p
}

#' Plot an empty rose.
#'
#' @inheritParams plot_rose
#'
#' @return ggplot2 object of an empty rose
rose_empty <- function(ski_area_name, aesthetics) {
  dat <- data.frame(bin_center = c(0, 360), bin_count = c(1, 1))
  plot_rose(
    dat, ski_area_name,
    type = "us", aesthetics = aesthetics,
    empty_rose = TRUE, labels = NULL
  ) +
    theme(
      plot.background = element_blank(),
      panel.grid.major.y = element_blank(),
    )
}

#' Align dots
#'
#' @param p ggplot2 object
#'
#' @return ggplot2 object of dots aligned earlier images (dots_overlay)
align_dots <- function(p) {
  ggplot(data.frame(x = 0:1, y = 0:1), aes(x = x, y = y)) +
    ggimage::geom_subview(subview = p, width = Inf, height = Inf, x = 0.5, y = 0.5) +
    ggimage::theme_nothing()
}

#' Plot segments, highlight specific directions
#'
#' @param dat Data frame of segments with columns `x`, `y`, `xend`, `yend`,
#' and `highlight`
#' @param color_vals Character vector of colors for `FALSE` and `TRUE`.
#'
#' @return ggplot2 object of segments with potential highlights
plot_segments <- function(
    dat, linewidth = 1, arrow_length = 0.2,
    color_vals = c(`FALSE` = text_teal, `TRUE` = rose_pink)) {
  dat |>
    ggplot() +
    aes(x = x, y = y, xend = xend, yend = yend, color = highlight) +
    geom_segment(
      arrow = arrow(type = "open", length = grid::unit(arrow_length, "cm")),
      linewidth = linewidth
    ) +
    scale_color_manual(values = color_vals, guide = "none") +
    coord_dartmouth +
    theme(
      panel.grid = element_blank(),
    )
}

## Data ------------------------------------------------
theme_set(bg_transparent())
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
dartmouth_segs <- arrow::read_parquet(file.path(img_data_dir, "dartmouth_segs.parquet"))
hemi <- arrow::read_parquet(file.path(img_data_dir, "hemisphere_roses.parquet")) |>
  tidyr::unnest(bearings)

dart <- arrow::read_parquet(file.path(img_data_dir, "dartmouth_runs.parquet")) |>
  dplyr::group_by(run_id) |>
  dplyr::mutate(winslow = (m * longitude) + b < latitude) |>
  dplyr::arrange(index)

dartmouth <- bearings_ls[["Dartmouth Skiway"]]
# whaleback <- bearings_ls[["Whaleback Mountain"]]
# killington <- bearings_ls[["Killington Resort"]]

n_groups <- 32 # number of spokes
x_range <- c(-72.1065, -72.08635)
y_range <- c(43.77828, 43.7899)
length_x <- diff(x_range)
length_y <- diff(y_range)

fig_height <- 3.75
fig_width <- fig_height * 703 / 503 # 865/452

# desired_yx_ratio <- dim(dartmouth_img)[1] / dim(dartmouth_img)[2]
# desired_yx_ratio <- 1399/2356
# desired_yx_ratio <- 1429 / 2768
desired_yx_ratio <- 1420 / 1897
ratio <- (length_x / length_y) * desired_yx_ratio
coord_dartmouth <- coord_fixed(
  xlim = x_range,
  ylim = y_range,
  ratio = ratio
)

## Dots and segments ----
dots_only <- dart |>
  ggplot() +
  aes(x = longitude, y = latitude, color = winslow) +
  scale_color_manual(values = c("#FF8C42", "#36B37E"), guide = "none") +
  geom_point(size = 0.5) +
  coord_dartmouth +
  theme(
    panel.grid.major = element_line(linewidth = 0.2, color = "grey80"),
    panel.grid.minor = element_line(linewidth = 0.2, color = "grey80"),
  )
dots_overlay <- ggimage::ggbackground(
  dots_only +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
    ),
  dart_url
)

ggsave(
  file.path(img_dir, "dots_overlay.png"),
  dots_overlay,
  width = fig_width,
  height = fig_height,
  dpi = 300
)
ggsave(
  file.path(img_dir, "dots_only.png"),
  align_dots(dots_only),
  width = fig_width, height = fig_height, dpi = 300
)

(
  dartmouth_segs |>
    dplyr::mutate(highlight = TRUE) |>
    plot_segments(linewidth = 0.5, arrow_length = 0.1) +
    theme(
      panel.grid.major = element_line(linewidth = 0.2, color = "grey80"),
      panel.grid.minor = element_line(linewidth = 0.2, color = "grey80"),
    )
) |>
  align_dots() |>
  ggsave(
    file.path(img_dir, "segments_plot.png"),
    plot = _,
    width = fig_width, height = fig_height
  )

dartmouth_segs |>
  dplyr::mutate(highlight = group == 28) |>
  plot_segments() |>
  ggsave(
    file.path(img_dir, "segments_nwbw.svg"),
    plot = _,
    width = fig_width * 2, height = fig_height * 2
  )

# dartmouth_segs |>
#   mutate(highlight = group == 28) |>
#   plot_segments(color_vals = c("grey80", dark_pink)) |>
#   ggsave(
#     file.path(img_dir, "segments_nwbw_light.svg"),
#     plot = _,
#     width = fig_width * 2, height = fig_height * 2
#   )

segs_nne <- dartmouth_segs |>
  dplyr::mutate(highlight = group == 3) |>
  plot_segments()
ggsave(
  file.path(img_dir, "segments_nne.svg"),
  plot = segs_nne,
  width = fig_width * 2, height = fig_height * 2
)

## Roses -----------------------------------------------------------
rose_dartmouth <- plot_rose(dartmouth)
ggsave(
  file.path(img_dir, "rose_dartmouth.svg"),
  rose_dartmouth,
  width = 6, height = 6
)

rose_nwbw <- dartmouth |>
  dplyr::mutate(color = dplyr::if_else((dplyr::row_number()) != 28, text_teal, rose_pink)) |>
  plot_rose(highlight = TRUE) +
  geom_text(x = 298, y = 26.3, label = "NWbW", color = rose_pink, family = my_font, size = 8)

ggsave(
  file.path(img_dir, "rose_nwbw.svg"),
  rose_nwbw,
  width = 6, height = 6
)

rose_nne <- dartmouth |>
  dplyr::mutate(color = dplyr::if_else((dplyr::row_number()) != 3, text_teal, rose_pink)) |>
  plot_rose(highlight = TRUE) +
  geom_text(x = 22, y = 29, label = "NNE", color = rose_pink, family = my_font, size = 8)

ggsave(
  file.path(img_dir, "rose_nne.svg"),
  rose_nne,
  width = 6, height = 6
)

segs_nne_light <- dartmouth_segs |>
  dplyr::mutate(highlight = group == 3) |>
  plot_segments(color_vals = c("grey80", dark_pink))

rose_nne_light <- dartmouth |>
  dplyr::mutate(color = dplyr::if_else((dplyr::row_number()) != 3, "grey80", dark_pink)) |>
  plot_rose(highlight = TRUE) +
  theme(axis.text.x = element_text(color = light_pal$circle_color, size = 12)) +
  geom_text(x = 22, y = 30, label = "NNE", color = dark_pink, family = my_font, size = 7)

ggsave(
  file.path(img_dir, "dartmouth_nne_light.svg"),
  height = 6, width = 8,
  cowplot::ggdraw(segs_nne_light) +
    cowplot::draw_plot(rose_nne_light, .25, 0, .6, .6)
)

all_roses <- cowplot::plot_grid(
  plotlist = purrr::map2(
    bearings_ls,
    names(bearings_ls),
    plot_rose,
    labels = NULL,
    vert = 0.1,
    type = "all"
  )
)
ggsave(
  file.path(img_dir, "all_roses.svg"),
  all_roses,
  width = 16, height = 12
)

north <- hemi |>
  dplyr::filter(num_bins == n_groups, hemisphere == "north") |>
  plot_rose("Northern hemisphere", size_title = 24, size_x = 20)
south <- hemi |>
  dplyr::filter(num_bins == n_groups, hemisphere == "south") |>
  plot_rose("Southern hemisphere", size_title = 24, size_x = 20)
ggsave(
  file.path(img_dir, "hemisphere.svg"),
  cowplot::plot_grid(north, south, ncol = 2),
  width = 12, height = 6
)

## Region -----------------------------------------------------------

water_mark <- "OpenSkiStats.org\nLicense: CC BY 4.0"
region_raw <- arrow::read_parquet(file.path(img_data_dir, "region_roses.parquet"))
n_ski_areas <- sum(region_raw$ski_areas_count)
verts <- region_raw |>
  dplyr::select(region, combined_vertical) |>
  dplyr::mutate(
    thickness = sqrt(combined_vertical / max(combined_vertical)) * max_thickness + min_thickness,
    .keep = "unused"
  ) |>
  tibble::deframe()

region <- region_raw |>
  dplyr::select(region, bearings) |>
  tibble::deframe() |>
  lapply(
    \(x) dplyr::filter(x, num_bins == ngroups_us)
  )

# US state layout slightly modified from NPR: move HI and AK in, switch MA and RI.
# Four corners are preserved. Redditors should be satisfied.
# https://blog.apps.npr.org/2015/05/11/hex-tile-maps.html
# This blog post has a good analysis but ignores the US perimeter shape:
# https://kristw.medium.com/whose-grid-map-is-better-quality-metrics-for-grid-map-layouts-e3d6075d9e80


layout <- c(
  area(1, 1),
  area(1, 11),
  area(2, 10, 2, 10), area(2, 11),
  area(3, 1), area(3, 2), area(3, 3), area(3, 4), area(3, 5), 
  area(3, 6), area(3, 7), area(3, 8), area(3, 9), area(3, 10), area(3, 11),
  area(4, 1), area(4, 2), area(4, 3), area(4, 4), area(4, 5),
  area(4, 6), area(4, 7), area(4, 8), area(4, 9), area(4, 10),
  area(5, 1), area(5, 2), area(5, 3), area(5, 4), area(5, 5),
  area(5, 6), area(5, 7), area(5, 8), area(5, 9), area(5, 10),
  area(6, 2), area(6, 3), area(6, 4), area(6, 5), area(6, 6),
  area(6, 7), area(6, 8), area(6, 9), 
  area(7, 4), area(7, 5), area(7, 6), area(7, 7), area(7, 8),
  area(8, 1), area(8, 4), area(8, 9)
)

layout_title <- c(area(1, 3, 2, 9), layout)

state_names <- c(
  "AK",
  "ME", "VT", "NH", "WA", "ID", "MT", "ND", "MN", 
  "IL", "WI", "MI", "NY", "MA", "RI",
  "OR", "NV", "WY", "SD", "IA", "IN", "OH", "PA", "NJ", "CT",
  "CA", "UT", "CO", "NE", "MO", "KY", "WV", "VA", "MD", "DE",
  "AZ", "NM", "KS", "AR", "TN", "NC", "SC", "DC",
  "OK", "LA", "MS", "AL", "GA",
  "HI", "TX", "FL"
)

state_map <- setNames(state.abb, state.name)

for (pal in names(my_aesthetics)) {
  my_aesthetic <- my_aesthetics[[pal]]
  all_regions <- list(region, state_map[names(region)], vert = verts) |>
    purrr::pmap(
      plot_rose,
      labels = NULL,
      size_title = my_aesthetic$size_label,
      ngr = ngroups_us,
      aesthetics = my_aesthetic,
      type = "us"
    )

  names(all_regions) <- state_map[names(region)]

  state_plots <- lapply(state_names, rose_empty, aesthetics = my_aesthetic) |>
    setNames(state_names)

  skiable_states <- names(all_regions)
  for (i in skiable_states) {
    state_plots[[i]] <- all_regions[[i]]
  }
  state_plots <- state_plots[state_names]

  title <-
    ggplot() +
    theme_void() +
    coord_cartesian(ylim = c(-2, 0.7)) +
    annotate(
      "text",
      x = 0, y = 0,
      label = "Which way do you ski?", size = 32,
      color = my_aesthetic$title_color,
      family = my_aesthetic$my_font
    ) +
    annotate(
      "text",
      x = 0, y = -0.9,
      label = sprintf("Orientations of %s US ski areas", n_ski_areas),
      size = 18,
      color = my_aesthetic$title_color,
      family = my_aesthetic$my_font
    )

  map_plots <- c(list(title = title), state_plots) |>
    patchwork::wrap_plots(design = layout_title) &
    patchwork::plot_annotation(theme = theme(
      panel.background = element_rect(fill = my_aesthetic$canvas, colour = NA),
      plot.background = element_rect(fill = my_aesthetic$canvas, colour = NA)
    ))

  to_publish <- cowplot::ggdraw(map_plots) +
    cowplot::draw_label(
      x = 0.985,
      y = 0.043,
      size = 28,
      hjust = 1,
      color = my_aesthetic$anno_color,
      fontfamily = my_font,
      lineheight = 1.6,
      label = water_mark
    ) +
    cowplot::draw_label(
      x = 0.165,
      y = 0.78,
      size = 28,
      hjust = 1,
      color = my_aesthetic$anno_color,
      fontfamily = my_font,
      label = "border proportional to\ncombined vertical drop"
    ) +
    cowplot::draw_line(
      x = c(0.185, 0.21),
      y = c(0.768, 0.726),
      color = my_aesthetic$circle_color, size = 2
    ) +
    cowplot::draw_line(
      x = c(0.17, 0.185),
      y = c(0.768, 0.768),
      color = my_aesthetic$circle_color, size = 2
    )

  ggsave(
    file.path(img_dir, sprintf("us_roses_%s.svg", pal)),
    to_publish,
    width = 11 * 3, height = 8 * 3, units = "in",
    dpi = 300
  )
  ggsave(
    file.path(img_dir, sprintf("us_roses_%s.png", pal)),
    to_publish,
    width = 11 * 3, height = 8 * 3, units = "in",
    device = "png",
    dpi = 300
  )
}
