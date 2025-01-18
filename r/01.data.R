set.seed(1618)
data_dir <- "../data"
img_data_dir <- "../data/images/data"

## dartmouth_runs -----------------------------------------------------------
dart <- arrow::read_parquet(file.path(img_data_dir, "skiway_run_coordinates.parquet"))

## ----dartmouth_segs---------------------------------------------------------
n_groups <- 32 # number of spokes
step_deg <- 360 / n_groups # Step in degrees
boundaries <- seq(-step_deg / 2, 360, by = step_deg)

dartmouth_segs <- dart |>
  dplyr::rename(path = run_id, x = longitude, y = latitude) |>
  dplyr::group_by(path) |>
  dplyr::arrange(index) |>
  dplyr::mutate(
    index0 = index,
    xend = dplyr::lead(x), # Next x-coordinate
    yend = dplyr::lead(y), # Next y-coordinate
    dx = xend - x,
    dy = yend - y,
    scaled_dx = dplyr::lead(distance_vertical) * dx / 4,
    scaled_dy = dplyr::lead(distance_vertical) * dy / 4,
    tg = (atan2(dx, dy) * 180 / pi) %% 360,
    group = findInterval(tg, boundaries) - 1,
    group = group %% n_groups + 1,
    state = "a",
  ) |>
  dplyr::ungroup() |>
  dplyr::mutate(index = dplyr::row_number()) |>
  dplyr::group_by(group) |>
  dplyr::arrange(tg) |>
  dplyr::ungroup() |>
  dplyr::relocate(xend, yend, state, scaled_dx, scaled_dy, dx, dy, tg, group) |>
  dplyr::filter(!is.na(xend)) # Remove rows where there is no "next" point

arrow::write_parquet(dartmouth_segs, file.path(img_data_dir, "dartmouth_segs.parquet"))

## ----bearings of 48 random ski areas + Dartmouth Skiway--------------------
ski_area_metrics <- arrow::read_parquet(file.path(data_dir, "ski_area_metrics.parquet"))
bearings_ls <- ski_area_metrics |>
  dplyr::filter(
    run_count >= 3, combined_vertical >= 50, ski_area_name != "",
    country == "United States", nchar(ski_area_name) < 20
  ) |>
  dplyr::sample_n(48) |>
  dplyr::bind_rows(dplyr::filter(ski_area_metrics, ski_area_name == "Dartmouth Skiway")) |>
  dplyr::arrange(ski_area_name) |>
  dplyr::select(ski_area_name, bearings) |>
  tibble::deframe() |>
  lapply(
    function(x) {
      dplyr::filter(x, num_bins == 32) |>
        dplyr::mutate(
          color = dplyr::if_else(bin_index == 2, "#f07178", "#004B59")
        )
    }
  )
saveRDS(bearings_ls, file.path(img_data_dir, "bearings_48_ls.rds"))
