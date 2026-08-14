import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot
import polars as pl
from rich.progress import Progress

from openskistats.analyze import load_bearing_distribution_pl, load_ski_areas_pl
from openskistats.bearing import get_difficulty_color_to_bearing_bin_counts
from openskistats.plot import (
    _generate_margin_text,
    _plot_mean_bearing_as_snowflake,
    _plot_solar_location_band,
    plot_orientation,
)
from openskistats.utils import get_data_directory


def get_display_ski_area_filters() -> list[pl.Expr]:
    """Ski area filters to produce a subset of ski areas for display."""
    return [
        pl.col("run_count") >= 3,
        pl.col("combined_vertical") >= 50,
        pl.col("ski_area_name").is_not_null(),
    ]


def create_ski_area_roses(overwrite: bool = False) -> None:
    """
    Export ski area roses to SVG for display.
    """
    directory = get_data_directory().joinpath("webapp", "ski-areas")
    directory_preview = directory.joinpath("roses-preview")
    directory_full = directory.joinpath("roses-full")
    directory_openskimap = directory.joinpath("roses-openskimap")
    for _directory in directory_preview, directory_full, directory_openskimap:
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
        openskimap_path = directory_openskimap.joinpath(f"{ski_area_id}.svg")
        if not overwrite and full_path.exists():
            continue
        tasks.append(
            {
                "info": info,
                "bearing_pl": bearings_pl.filter(pl.col("ski_area_id") == ski_area_id),
                "preview_path": preview_path,
                "full_path": full_path,
                "openskimap_path": openskimap_path,
            }
        )
    logging.info(f"Creating roses for {len(tasks):,} ski areas concurrently...")

    with ProcessPoolExecutor() as executor, Progress() as progress:
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
    info: dict[str, Any],
    bearing_pl: pl.DataFrame,
    preview_path: Path,
    full_path: Path,
    openskimap_path: Path,
) -> None:
    """Create a preview and a full rose for a ski area."""
    ski_area_id = info["ski_area_id"]
    ski_area_name = info["ski_area_name"]
    color_convention = info["osm_run_convention"]

    # supported metadata keys listed at
    # https://matplotlib.org/stable/api/backend_svg_api.html#matplotlib.backends.backend_svg.FigureCanvasSVG.print_svg
    common_metadata = {
        "Title": f"Ski Rose for {ski_area_name}",
        "Creator": "https://github.com/dhimmel/openskistats",
        "Source": f"https://openskimap.org/?obj={ski_area_id}",
        "Rights": "https://creativecommons.org/licenses/by/4.0/",
    }

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
            **common_metadata,
            "Description": "An 8-bin histogram of downhill ski run orientations.",
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
    full_rose_metadata = {
        **common_metadata,
        "Description": "A 32-bin stacked histogram of downhill ski run orientations colored by difficulty.",
    }
    fig.savefig(
        full_path,
        format="svg",
        bbox_inches="tight",
        # pad_inches=0.02,
        facecolor="#FFFFFF",
        transparent=False,
        metadata=full_rose_metadata,
    )

    # create a reduced text version for OpenSkiMap embeds
    # https://github.com/dhimmel/openskistats/issues/49
    ax.set_title("")
    for text in ax.texts:
        text.remove()
    logging.info(f"Writing {openskimap_path}")
    fig.savefig(
        openskimap_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.005,
        facecolor="#FFFFFF",
        transparent=False,
        metadata=full_rose_metadata,
    )

    matplotlib.pyplot.close(fig)
