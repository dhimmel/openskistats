import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from patito.exceptions import DataFrameValidationError
from rich.progress import Progress

from openskistats.bearing import (
    add_spatial_metric_columns,
    get_bearing_histograms,
    get_bearing_summary_stats,
    get_difficulty_color_to_bearing_bin_counts,
)
from openskistats.models import (
    BearingStatsModel,
    OpenSkiMapStatus,
    RunCoordinateSegmentModel,
    RunModel,
    SkiAreaModel,
)
from openskistats.openskimap_utils import (
    load_downhill_runs_from_download_pl,
    load_downhill_ski_areas_from_download_pl,
    load_lifts_from_download_pl,
    set_openskimap_download_info_in_variables,
)
from openskistats.plot import (
    _generate_margin_text,
    _plot_mean_bearing_as_snowflake,
    _plot_solar_location_band,
    plot_orientation,
    subplot_orientations,
)
from openskistats.sunlight import (
    add_solar_irradiation_columns,
)
from openskistats.utils import (
    get_data_directory,
    get_images_data_directory,
    get_images_directory,
    pl_condense_run_difficulty,
    pl_hemisphere,
    pl_weighted_mean,
)
from openskistats.variables import set_variables


def get_ski_area_metrics_path(testing: bool = False) -> Path:
    return get_data_directory(testing=testing).joinpath("ski_area_metrics.parquet")


def get_runs_parquet_path(testing: bool = False) -> Path:
    return get_data_directory(testing=testing).joinpath("runs.parquet")


def get_lifts_parquet_path(testing: bool = False) -> Path:
    return get_data_directory(testing=testing).joinpath("lifts.parquet")


_aggregate_run_coordinates_exprs = {
    "coordinate_count": pl.len(),
    "segment_count": pl.count("segment_hash"),
    "combined_vertical": pl.col("distance_vertical_drop").sum(),
    "combined_distance": pl.col("distance_3d").sum(),
    "latitude": pl.col("latitude").mean(),
    "longitude": pl.col("longitude").mean(),
    "min_elevation": pl.col("elevation").min(),
    "max_elevation": pl.col("elevation").max(),
    "vertical_drop": pl.max("elevation") - pl.min("elevation"),
}

_aggregate_run_segment_sunlight_exprs = {
    "solar_irradiation_season": pl_weighted_mean(
        value_col="solar_irradiation_season",
        weight_col="distance_vertical_drop",
    ),
    "solar_irradiation_solstice": pl_weighted_mean(
        value_col="solar_irradiation_solstice",
        weight_col="distance_vertical_drop",
    ),
}


def process_and_export_lifts() -> None:
    """
    Process and export lifts from OpenSkiMap.
    """
    lifts = load_lifts_from_download_pl()
    coords_df = (
        lifts.lazy()
        .select("lift_id", "lift_coordinates")
        .explode("lift_coordinates")
        .unnest("lift_coordinates")
        .filter(pl.col("index").is_not_null())
        .pipe(add_spatial_metric_columns, partition_by="lift_id")
        .select(
            "lift_id",
            *[
                x
                for x in RunCoordinateSegmentModel.model_fields
                if not x.startswith("solar_")
            ],
        )
        .group_by("lift_id")
        .agg(
            **_aggregate_run_coordinates_exprs,
            lift_coordinates=pl.struct(pl.exclude("lift_id")),
        )
        .collect()
    )
    lifts = (
        lifts.drop("lift_coordinates")
        .join(coords_df, on="lift_id", how="left")
        .with_columns(
            pl.col("coordinate_count").fill_null(0),
            pl.col("segment_count").fill_null(0),
        )
        .sort("lift_id")
    )
    lifts_path = get_lifts_parquet_path()
    logging.info(f"Writing {len(lifts):,} lifts to {lifts_path}")
    lifts.write_parquet(lifts_path)


def process_and_export_runs() -> None:
    """
    Process and export runs from OpenSkiMap.
    """
    runs_lazy = load_downhill_runs_from_download_pl().lazy()
    coords_df = (
        runs_lazy.select("run_id", "run_coordinates_clean")
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .filter(pl.col("index").is_not_null())
        .pipe(add_spatial_metric_columns, partition_by="run_id")
        .collect()
        .pipe(add_solar_irradiation_columns)
        .lazy()
        .select(
            "run_id",
            *RunCoordinateSegmentModel.model_fields,
        )
        .group_by("run_id")
        .agg(
            **_aggregate_run_coordinates_exprs,
            **_aggregate_run_segment_sunlight_exprs,
            run_coordinates_clean=pl.struct(pl.exclude("run_id")),
        )
    )
    runs_df = (
        runs_lazy.drop("run_coordinates_clean")
        .join(coords_df, on="run_id", how="left")
        .with_columns(
            pl.col("coordinate_count").fill_null(0),
            pl.col("segment_count").fill_null(0),
        )
        .collect()
    )
    try:
        RunModel.validate(runs_df, allow_superfluous_columns=True)
    except DataFrameValidationError as exc:
        logging.error(f"RunModel.validate failed with {exc}")
    set_variables(
        openskimap__runs__segments__counts__04_downhill_clean=runs_df[
            "segment_count"
        ].sum(),
        openskimap__runs__distance__vertical__04_downhill_clean=runs_df[
            "combined_vertical"
        ].sum(),
        openskimap__runs__distance__3d__04_downhill_clean=runs_df[
            "combined_distance"
        ].sum(),
    )
    runs_path = get_runs_parquet_path()
    logging.info(f"Writing {len(runs_df):,} runs to {runs_path}")
    runs_df.write_parquet(runs_path)


def analyze_all_ski_areas_polars(skip_runs: bool = False) -> None:
    """
    Analyze ski areas to create a table of ski areas and their metrics
    including bearing distributions.
    Keyed on ski_area_id.
    Write data as parquet.
    """
    from openskistats.counts import get_ski_area_comparable_counts

    set_openskimap_download_info_in_variables()
    process_and_export_lifts()
    if not skip_runs:
        process_and_export_runs()
    logging.info("Creating ski area metrics dataframe with bearing distributions.")
    ski_area_df = load_downhill_ski_areas_from_download_pl().lazy()
    ski_area_run_metrics_df = (
        load_runs_pl()
        .lazy()
        .explode("ski_area_ids")
        .rename({"ski_area_ids": "ski_area_id"})
        .filter(pl.col("ski_area_id").is_not_null())
        .select(
            "ski_area_id",
            "run_id",
            pl_condense_run_difficulty(),
            "run_coordinates_clean",
        )
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .with_columns(
            hemisphere=pl_hemisphere("latitude"),
        )
        .group_by("ski_area_id")
        .agg(
            run_count=pl.col("run_id").n_unique(),
            **_aggregate_run_coordinates_exprs,
            **_aggregate_run_segment_sunlight_exprs,
            hemisphere=pl.first("hemisphere"),
            _bearing_stats=pl.struct(
                "bearing",
                pl.col("distance_vertical_drop").alias("bearing_magnitude_cum"),
                "hemisphere",
            ).map_batches(_get_bearing_summary_stats_pl, returns_scalar=True),
            # do we need to filter nulls?
            bearings=pl.struct(
                "bearing", "distance_vertical_drop", "run_difficulty_condensed"
            ).map_batches(
                lambda x: get_bearing_histograms(
                    bearings=x.struct.field("bearing").to_numpy(),
                    weights=x.struct.field("distance_vertical_drop").to_numpy(),
                    difficulties=x.struct.field("run_difficulty_condensed").to_numpy(),
                ).to_struct(),
                returns_scalar=True,
            ),
        )
        .unnest("_bearing_stats")
    )
    ski_area_metrics_df = (
        ski_area_df.join(ski_area_run_metrics_df, on="ski_area_id", how="left")
        .with_columns(
            *[
                pl.col(col).fill_null(pl.lit(fill_value))
                for col, fill_value in SkiAreaModel.defaults.items()
            ]
        )
        .collect()
    )
    try:
        SkiAreaModel.validate(ski_area_metrics_df, allow_superfluous_columns=True)
    except DataFrameValidationError as exc:
        logging.error(f"SkiAreaModel.validate failed with {exc}")
    ski_area_metrics_path = get_ski_area_metrics_path()
    logging.info(f"Writing {ski_area_metrics_path}")
    ski_area_metrics_df.write_parquet(ski_area_metrics_path)
    set_variables(**get_ski_area_comparable_counts())
    set_latitude_metrics()
    _create_data_inputs_for_r()


def load_runs_pl(
    run_filters: list[pl.Expr] | None = None,
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.LazyFrame:
    """
    When ski_area_filters is provided, ski area metrics are loaded and ski_area_ids
    are exploded in the runs table, such that the runs table is filtered to only selected ski areas.
    A new ski_area_id column is added to the runs table that becomes part of the primary key.
    """
    path = get_runs_parquet_path()
    logging.info(f"Loading runs metrics from {path}")
    df = pl.scan_parquet(source=path)
    df = df.filter(*_prepare_pl_filters(run_filters))
    if ski_area_filters:
        ski_areas = (
            load_ski_areas_pl(ski_area_filters=ski_area_filters)
            .select("ski_area_id")
            .lazy()
        )
        df = (
            df.explode("ski_area_ids")
            .rename({"ski_area_ids": "ski_area_id"})
            .join(ski_areas, on="ski_area_id", how="semi", maintain_order="left")
        )
    return df


def load_run_coordinates_pl(
    run_filters: list[pl.Expr] | None = None,
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.LazyFrame:
    """Load run_id with coordinates exploded and unnested for a flat table of run-coordinate pairs."""
    return (
        load_runs_pl(run_filters=run_filters, ski_area_filters=ski_area_filters)
        .select("run_id", "run_coordinates_clean")
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
    )


def load_run_segments_pl(
    run_filters: list[pl.Expr] | None = None,
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.LazyFrame:
    """
    Load run_id with coordinates exploded, unnested, and filtered to segments,
    for a flat table of run-segment pairs.
    """
    return load_run_coordinates_pl(
        run_filters=run_filters, ski_area_filters=ski_area_filters
    ).filter(pl.col("segment_hash").is_not_null())


def load_lifts_pl() -> pl.DataFrame:
    path = get_lifts_parquet_path()
    logging.info(f"Loading lifts metrics from {path}")
    return pl.read_parquet(source=path)


def load_ski_areas_pl(ski_area_filters: list[pl.Expr] | None = None) -> pl.DataFrame:
    path = get_ski_area_metrics_path()
    logging.info(f"Loading ski area metrics from {path}")
    return pl.read_parquet(source=path).filter(*_prepare_pl_filters(ski_area_filters))


def load_bearing_distribution_pl(
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    """
    Table of ski area bearing distributions.
    Keyed on ski_area_id, num_bins, bin_center.
    """
    return (
        load_ski_areas_pl(ski_area_filters=ski_area_filters)
        .select("ski_area_id", "bearings")
        .explode("bearings")
        .filter(pl.col("bearings").is_not_null())
        .unnest("bearings")
    )


def _get_bearing_summary_stats_pl(struct_series: pl.Series) -> BearingStatsModel:
    df = (
        struct_series.alias("input_struct")
        .to_frame()
        .unnest("input_struct")
        .drop_nulls()
    )
    try:
        (hemisphere,) = df.get_column("hemisphere").unique()
    except ValueError:
        hemisphere = None
    return get_bearing_summary_stats(
        bearings=df.get_column("bearing").to_numpy(),
        cum_magnitudes=df.get_column("bearing_magnitude_cum").to_numpy(),
        alignments=df.get_column("bearing_alignment", default=None),
        hemisphere=hemisphere,
    )


def _prepare_pl_filters(
    filters: list[pl.Expr] | None = None,
) -> list[pl.Expr | bool]:
    if not filters:
        # pl.lit(True) had issues that True did not.
        # https://github.com/pola-rs/polars/issues/19771
        return [True]
    return filters


def aggregate_ski_areas_pl(
    group_by: list[str],
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    assert len(group_by) > 0
    bearings_pl = (
        aggregate_ski_area_bearing_dists_pl(
            group_by=group_by, ski_area_filters=ski_area_filters
        )
        .group_by(*group_by)
        .agg(bearings=pl.struct(pl.exclude(group_by)))
    )
    return (
        load_ski_areas_pl(ski_area_filters=ski_area_filters)
        .group_by(*group_by)
        .agg(
            ski_areas_count=pl.n_unique("ski_area_id"),
            country_count=pl.n_unique("country"),
            run_count=pl.sum("run_count"),
            coordinate_count=pl.sum("coordinate_count"),
            segment_count=pl.sum("segment_count"),
            lift_count=pl.sum("lift_count"),
            combined_vertical=pl.sum("combined_vertical"),
            combined_distance=pl.sum("combined_distance"),
            min_elevation=pl.min("min_elevation"),
            max_elevation=pl.max("max_elevation"),
            vertical_drop=pl.max("max_elevation") - pl.min("min_elevation"),
            latitude=pl.mean("latitude"),
            longitude=pl.mean("longitude"),
            osm_run_convention=pl.col("osm_run_convention").unique().drop_nulls(),
            _bearing_stats=pl.struct(
                pl.col("bearing_mean").alias("bearing"),
                "bearing_magnitude_cum",
                "bearing_alignment",
                "hemisphere",
            ).map_batches(_get_bearing_summary_stats_pl, returns_scalar=True),
        )
        .with_columns(
            osm_run_convention=pl.when(pl.col("osm_run_convention").list.len() == 1)
            .then(pl.col("osm_run_convention").list.first())
            .otherwise(None)
        )
        .unnest("_bearing_stats")
        .join(bearings_pl, on=group_by)
        .sort(*group_by)
    )


def aggregate_ski_area_bearing_dists_pl(
    group_by: list[str],
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    return (
        load_ski_areas_pl(ski_area_filters=ski_area_filters)
        .explode("bearings")
        .filter(pl.col("bearings").is_not_null())
        .unnest("bearings")
        .group_by(*group_by, "num_bins", "bin_index")
        .agg(
            pl.first("bin_center").alias("bin_center"),
            pl.selectors.starts_with("bin_count").sum(),
            pl.first("bin_label").alias("bin_label"),
        )
        .with_columns(
            bin_count_total=pl.sum("bin_count").over(*group_by, "num_bins"),
        )
        # nan values are not helpful here when bin_count_total is 0
        # Instead, set bin_proportion to 0, although setting to null could also make sense
        .with_columns(
            bin_proportion=pl.when(pl.col("bin_count_total") > 0)
            .then(pl.col("bin_count").truediv("bin_count_total"))
            .otherwise(0.0)
        )
        .drop("bin_count_total")
        .sort(*group_by, "num_bins", "bin_center")
    )


def bearing_dists_by_us_state() -> pl.DataFrame:
    return aggregate_ski_areas_pl(
        group_by=["region"],
        ski_area_filters=[
            pl.col("country") == "United States",
            pl.col("region").is_not_null(),
            pl.col("osm_status") == OpenSkiMapStatus.operating,
            pl.col("ski_area_name").is_not_null(),
        ],
    )


def bearing_dists_by_hemisphere() -> pl.DataFrame:
    dists = aggregate_ski_areas_pl(
        group_by=["hemisphere"],
        ski_area_filters=[
            pl.col("hemisphere").is_not_null(),
            pl.col("osm_status") == OpenSkiMapStatus.operating,
            pl.col("ski_area_name").is_not_null(),
        ],
    )
    # compare key metrics in the northern and southern hemispheres
    compare_cols = pl.selectors.by_name(
        "ski_areas_count",
        "run_count",
        "coordinate_count",
        "segment_count",
        "lift_count",
        "combined_vertical",
        "combined_distance",
    )
    set_variables(
        **dists.sort("hemisphere")
        .select(
            compare_cols.first()
            .truediv(compare_cols.last())
            .round(4)
            .name.map(lambda x: f"hemisphere__north_over_south__{x}")
        )
        .to_dicts()[0]
    )
    return dists


def set_latitude_metrics() -> None:
    from openskistats.plot_runs import RunLatitudeBearingHistogram

    rlbh = RunLatitudeBearingHistogram()
    set_variables(
        latitude__between_35_50__combined_vertical_prop=rlbh.get_combined_vertical_prop_in_latitude_range(
            35, 50
        ),
        latitude__between_40_50__combined_vertical_prop=rlbh.get_combined_vertical_prop_in_latitude_range(
            40, 50
        ),
        latitude__between_45_58__combined_vertical_prop=rlbh.get_combined_vertical_prop_in_latitude_range(
            45, 48
        ),
    )


def bearing_dists_by_status() -> pl.DataFrame:
    """
    Bearing distributions by ski area operating status.
    Only includes ski areas in the northern hemisphere because there were no abandoned ski areas in the southern hemisphere at ToW.
    """
    return aggregate_ski_areas_pl(
        group_by=["hemisphere", "osm_status"],
        ski_area_filters=[
            pl.col("hemisphere") == "north",
            pl.col("osm_status").is_in(
                [OpenSkiMapStatus.abandoned, OpenSkiMapStatus.operating]
            ),
        ],
    ).with_columns(
        hemisphere_status=pl.format(
            "{} in {} Hem.",
            pl.col("osm_status").str.to_titlecase(),
            pl.col("hemisphere").str.to_titlecase(),
        )
    )


def bearing_dists_by_country() -> pl.DataFrame:
    return aggregate_ski_areas_pl(
        group_by=["country"],
        ski_area_filters=[
            pl.col("country").is_not_null(),
            pl.col("osm_status") == OpenSkiMapStatus.operating,
            pl.col("ski_area_name").is_not_null(),
        ],
    )


def ski_rose_the_world(min_combined_vertical: int = 10_000) -> None:
    image_directory = get_images_directory()
    path = image_directory.joinpath("ski-roses.pdf")
    pdf_pages = PdfPages(
        filename=path,
        metadata={
            "Title": "Ski Roses of the World: Downhill Ski Trail Orientations",
            "Author": "https://github.com/dhimmel/openskistats",
        },
    )
    grouping_col_to_stats = {
        "hemisphere": bearing_dists_by_hemisphere(),
        "hemisphere_status": bearing_dists_by_status(),
        "country": bearing_dists_by_country(),
        "region": bearing_dists_by_us_state(),
    }
    figures = {}
    for grouping_col, groups_pl in grouping_col_to_stats.items():
        name = f"{grouping_col}_roses"
        logging.info(f"{name}: plotting ski roses by {grouping_col}")
        groups_pl.write_parquet(
            file=get_images_data_directory().joinpath(f"{name}.parquet")
        )
        groups_pl = groups_pl.filter(
            pl.col("combined_vertical") >= min_combined_vertical
        )
        if groups_pl.is_empty():
            logging.info(
                f"Skipping {grouping_col} plot which returns no groups with combined_vertical >= {min_combined_vertical:,} m."
            )
            continue
        fig = subplot_orientations(
            groups_pl=groups_pl,
            grouping_col=grouping_col,
            n_cols=min(4, len(groups_pl)),
            free_y=True,
        )
        figures[name] = fig
    from openskistats.plot_runs import (
        RunLatitudeBearingHistogram,
        plot_bearing_by_latitude_bin,
        plot_run_difficulty_histograms_by_slope,
    )
    from openskistats.plot_ski_areas import (
        SkiAreaSubsetPlot,
        plot_ski_area_metric_ecdfs,
    )

    figures["select_ski_area_roses"] = SkiAreaSubsetPlot.plot_rose_grid()
    figures["bearing_by_latitude_eye"] = plot_bearing_by_latitude_bin()
    figures["latitude_histogram"] = (
        RunLatitudeBearingHistogram().plot_latitude_histogram().draw()
    )
    figures["slope_by_difficulty_distributions_full"] = (
        plot_run_difficulty_histograms_by_slope(condense_difficulty=False).draw()
    )
    figures["slope_by_difficulty_distributions_condensed"] = (
        plot_run_difficulty_histograms_by_slope(condense_difficulty=True).draw()
    )
    lorenz_plot, gini_plot = plot_ski_area_metric_ecdfs()
    figures["ski_area_lorenz_curves"] = lorenz_plot.draw()
    figures["ski_area_gini_bars"] = gini_plot.draw()
    # save SVGs
    for name, fig in figures.items():
        fig.savefig(
            image_directory.joinpath(f"{name}.svg"),
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
            metadata={
                "Creator": "https://github.com/dhimmel/openskistats",
            },
        )
    # save multi-page PDF
    logging.info(f"Writing ski rose the world to {path}")
    with pdf_pages:
        for fig in figures.values():
            pdf_pages.savefig(fig, facecolor="#FFFFFF", bbox_inches="tight")
            matplotlib.pyplot.close(fig)


def get_story_ski_area_names() -> list[str]:
    """
    Subset of ski areas by name for the closeread scrollytelling story
    https://github.com/dhimmel/openskistats/issues/17
    """
    return [
        "Killington Resort",
        "Pico Mountain Resort",
        "Whaleback Mountain",
        "Storrs Hill Ski Area",
        "Dartmouth Skiway",
    ]


def get_display_ski_area_filters(story: bool = False) -> list[pl.Expr]:
    """Ski area filters to produce a subset of ski areas for display."""
    filters = [
        pl.col("run_count") >= 3,
        pl.col("combined_vertical") >= 50,
        pl.col("ski_area_name").is_not_null(),
    ]
    if story:
        filters.append(pl.col("ski_area_name").is_in(get_story_ski_area_names()))
    return filters


def create_ski_area_roses(overwrite: bool = False) -> None:
    """
    Export ski area roses to SVG for display.
    """
    directory = get_data_directory().joinpath("webapp", "ski-areas")
    directory_preview = directory.joinpath("roses-preview")
    directory_full = directory.joinpath("roses-full")
    for _directory in directory_preview, directory_full:
        _directory.mkdir(exist_ok=True, parents=True)
    ski_areas_pl = load_ski_areas_pl(
        ski_area_filters=get_display_ski_area_filters()
    ).drop("bearings")
    bearings_pl = load_bearing_distribution_pl(
        ski_area_filters=get_display_ski_area_filters()
    )
    logging.info(
        f"Filtered to {len(ski_areas_pl):,} ski areas. Rose plotting {overwrite=}."
    )
    tasks = []
    for info in ski_areas_pl.rows(named=True):
        ski_area_id = info["ski_area_id"]
        preview_path = directory_preview.joinpath(f"{ski_area_id}.svg")
        full_path = directory_full.joinpath(f"{ski_area_id}.svg")
        if not overwrite and full_path.exists():
            continue
        tasks.append(
            {
                "info": info,
                "bearing_pl": bearings_pl.filter(pl.col("ski_area_id") == ski_area_id),
                "preview_path": preview_path,
                "full_path": full_path,
            }
        )
    logging.info(f"Creating roses for {len(tasks):,} ski areas concurrently...")

    # use spawn instead of fork until Python 3.14 as per https://docs.pola.rs/user-guide/misc/multiprocessing/
    mp_context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(mp_context=mp_context) as executor, Progress() as progress:
        task_progress = progress.add_task("[cyan]Creating roses...", total=len(tasks))
        futures = [executor.submit(_create_ski_area_rose, **task) for task in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Task failed: {e}")
                raise
            finally:
                progress.update(task_progress, advance=1)


def _create_ski_area_rose(
    info: dict[str, Any], bearing_pl: pl.DataFrame, preview_path: Path, full_path: Path
) -> None:
    """Create a preview and a full rose for a ski area."""
    ski_area_id = info["ski_area_id"]
    ski_area_name = info["ski_area_name"]
    color_convention = info["osm_run_convention"]

    # plot and save preview rose
    bearing_preview_pl = bearing_pl.filter(pl.col("num_bins") == 8)
    fig, ax = plot_orientation(
        bin_counts=bearing_preview_pl.get_column("bin_count").to_numpy(),
        bin_centers=bearing_preview_pl.get_column("bin_center").to_numpy(),
        margin_text={},
        figsize=(1, 1),
        alpha=1.0,
        edgecolor="#6b6b6b",
        linewidth=0.4,
        disable_xticks=True,
    )
    # make the polar frame less prominent
    ax.spines["polar"].set_linewidth(0.4)
    ax.spines["polar"].set_color("#6b6b6b")
    logging.info(f"Writing {preview_path}")
    fig.savefig(
        preview_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
        metadata={
            "Title": f"Preview Ski Rose for {ski_area_name}",
            "Description": f"An 8-bin histogram of downhill ski trail orientations generated from <https://openskimap.org/?obj={ski_area_id}>.",
            "Creator": "https://github.com/dhimmel/openskistats",
        },
    )
    matplotlib.pyplot.close(fig)

    # plot and save full rose
    bearing_full_pl = bearing_pl.filter(pl.col("num_bins") == 32)
    fig, ax = plot_orientation(
        bin_counts=bearing_full_pl.get_column("bin_count").to_numpy(),
        bin_centers=bearing_full_pl.get_column("bin_center").to_numpy(),
        color_to_bin_counts=get_difficulty_color_to_bearing_bin_counts(
            bearing_full_pl, convention=color_convention
        ),
        title=ski_area_name,
        title_font_size=16,
        margin_text=_generate_margin_text(info),
        figsize=(4, 4),
        alpha=1.0,
    )
    _plot_mean_bearing_as_snowflake(
        ax=ax, bearing=info["bearing_mean"], alignment=info["bearing_alignment"]
    )
    _plot_solar_location_band(
        ax=ax,
        latitude=info["latitude"],
        longitude=info["longitude"],
        elevation=info["min_elevation"],
    )
    logging.info(f"Writing {full_path}")
    fig.savefig(
        full_path,
        format="svg",
        bbox_inches="tight",
        # pad_inches=0.02,
        facecolor="#FFFFFF",
        transparent=False,
        metadata={
            "Title": f"Ski Rose for {ski_area_name}",
            "Description": f"A 32-bin histogram of downhill ski trail orientations generated from <https://openskimap.org/?obj={ski_area_id}>.",
            "Creator": "https://github.com/dhimmel/openskistats",
        },
    )
    matplotlib.pyplot.close(fig)


def _create_data_inputs_for_r() -> None:
    """
    Create data inputs for R analysis.
    """
    logging.info("Creating data inputs for R analysis")
    path = get_images_data_directory().joinpath("skiway_run_coordinates.parquet")
    logging.info(f"Writing {path}")
    (
        load_runs_pl(ski_area_filters=[pl.col("ski_area_name").eq("Dartmouth Skiway")])
        .select("run_id", "run_name", "run_difficulty", "run_coordinates_clean")
        .collect()
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .drop("segment_hash")
        .write_parquet(path)
    )
