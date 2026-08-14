import hashlib
import json
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

ROSE_RENDER_VERSION = 1
"""Increment when a renderer change should invalidate local rose fingerprints."""

ROSE_FINGERPRINT_DECIMAL_PLACES = 8
"""Float precision preserved in rose fingerprints."""

ROSE_INFO_FIELDS = (
    "ski_area_id",
    "ski_area_name",
    "osm_run_convention",
    "run_count",
    "lift_count",
    "combined_vertical",
    "poleward_affinity",
    "eastward_affinity",
    "min_elevation",
    "max_elevation",
    "bearing_mean",
    "bearing_alignment",
    "latitude",
    "longitude",
)

ROSE_BEARING_FIELDS = (
    "num_bins",
    "bin_center",
    "bin_count",
    "bin_count_other",
    "bin_count_easy",
    "bin_count_intermediate",
    "bin_count_advanced",
)


def _canonicalize_rose_fingerprint_value(value: Any) -> Any:
    """Remove float noise below the precision relevant to rendered roses."""
    if not isinstance(value, float):
        return value
    rounded = round(value, ROSE_FINGERPRINT_DECIMAL_PLACES)
    return 0.0 if rounded == 0 else rounded


def get_display_ski_area_filters() -> list[pl.Expr]:
    """Ski area filters to produce a subset of ski areas for display."""
    return [
        pl.col("run_count") >= 3,
        pl.col("combined_vertical") >= 50,
        pl.col("ski_area_name").is_not_null(),
    ]


def _get_ski_area_rose_fingerprint(
    info: dict[str, Any], bearing_pl: pl.DataFrame
) -> str:
    """Fingerprint the inputs that determine all rose variants for a ski area."""
    payload = {
        "render_version": ROSE_RENDER_VERSION,
        "info": {
            field: _canonicalize_rose_fingerprint_value(info[field])
            for field in ROSE_INFO_FIELDS
        },
        "bearings": [
            {
                field: _canonicalize_rose_fingerprint_value(value)
                for field, value in row.items()
            }
            for row in bearing_pl.to_dicts()
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def create_ski_area_roses(overwrite: bool = False) -> None:
    """
    Export ski area roses to SVG for display.
    """
    data_directory = get_data_directory()
    directory = data_directory.joinpath("webapp", "ski-areas")
    directory_preview = directory.joinpath("roses-preview")
    directory_full = directory.joinpath("roses-full")
    directory_openskimap = directory.joinpath("roses-openskimap")
    fingerprint_path = data_directory.joinpath(
        "openskistats", "ski-area-rose-fingerprints.json"
    )
    for _directory in directory_preview, directory_full, directory_openskimap:
        _directory.mkdir(exist_ok=True, parents=True)
    fingerprint_path.parent.mkdir(exist_ok=True, parents=True)
    previous_fingerprints: dict[str, str] = {}
    if fingerprint_path.exists():
        previous_fingerprints = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    ski_areas_pl = load_ski_areas_pl(
        ski_area_filters=get_display_ski_area_filters()
    ).drop("bearings")
    bearings_pl = (
        load_bearing_distribution_pl(ski_area_filters=get_display_ski_area_filters())
        .filter(pl.col("num_bins").is_in([8, 32]))
        .sort("ski_area_id", "num_bins", "bin_center")
        .select("ski_area_id", *ROSE_BEARING_FIELDS)
    )
    bearings_by_ski_area = {
        key[0]: value
        for key, value in bearings_pl.partition_by(
            "ski_area_id", as_dict=True, include_key=False
        ).items()
    }
    logging.info(
        f"Filtered to {len(ski_areas_pl):,} ski areas. Rose plotting {overwrite=}."
    )
    tasks = []
    fingerprints: dict[str, str] = {}
    for info in ski_areas_pl.rows(named=True):
        ski_area_id = info["ski_area_id"]
        bearing_pl = bearings_by_ski_area[ski_area_id]
        fingerprint = _get_ski_area_rose_fingerprint(info, bearing_pl)
        fingerprints[ski_area_id] = fingerprint
        preview_path = directory_preview.joinpath(f"{ski_area_id}.svg")
        full_path = directory_full.joinpath(f"{ski_area_id}.svg")
        openskimap_path = directory_openskimap.joinpath(f"{ski_area_id}.svg")
        output_paths = preview_path, full_path, openskimap_path
        if (
            not overwrite
            and previous_fingerprints.get(ski_area_id) == fingerprint
            and all(path.exists() for path in output_paths)
        ):
            continue
        tasks.append(
            {
                "info": info,
                "bearing_pl": bearing_pl,
                "preview_path": preview_path,
                "full_path": full_path,
                "openskimap_path": openskimap_path,
            }
        )
    _create_ski_area_roses_concurrently(tasks)

    for rose_directory in directory_preview, directory_full, directory_openskimap:
        for path in rose_directory.glob("*.svg"):
            if path.stem not in fingerprints:
                path.unlink()
    fingerprint_path.write_text(
        json.dumps(fingerprints, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _create_ski_area_roses_concurrently(tasks: list[dict[str, Any]]) -> None:
    """Create pending ski-area roses in worker processes."""
    logging.info(f"Creating roses for {len(tasks):,} ski areas concurrently...")
    if not tasks:
        return
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
