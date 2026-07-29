"""
Dartmouth Skiway manuscript figure:
run segments with one compass direction highlighted and an inset ski rose.
Ported from the retired R implementation at
https://github.com/dhimmel/openskistats/blob/b01f47defbdb0119d76aed25235fa6d0cb887e92/r/02.plot.R.
"""

import math
from dataclasses import dataclass
from typing import Any

import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from openskistats.bearing import cut_bearings_pl
from openskistats.dartmouth import load_dartmouth_skiway_context
from openskistats.plot import plot_orientation

HIGHLIGHT_COLOR = "#d33c44"
MUTED_COLOR = "#cccccc"
LIFT_COLOR = "#dceaf2"
LODGE_COLOR = "#f2d49b"
ROAD_COLOR = LODGE_COLOR
ROAD_LABEL_COLOR = ROAD_COLOR
MAP_LABEL_COLOR = "#4d5961"

SKIWAY_FIGURE_NAME = "dartmouth_nne_light"
SKIWAY_NAME = "Dartmouth Skiway"
"""Ski area for the example figure, chosen for its two distinctly oriented ledges."""


@dataclass(frozen=True)
class GeographicBounds:
    """Fixed longitude and latitude bounds for a map canvas."""

    west: float
    east: float
    south: float
    north: float
    crs: str = "EPSG:4326"

    def local_data_aspect(self) -> float:
        """Return the latitude-to-longitude display scale at the map midpoint."""
        midpoint_latitude = (self.south + self.north) / 2
        return 1 / math.cos(math.radians(midpoint_latitude))

    def height_for_width(self, width: float) -> float:
        """Return the canvas height that preserves local geographic proportions."""
        longitude_span = self.east - self.west
        latitude_span = self.north - self.south
        geographic_width_to_height = longitude_span / (
            self.local_data_aspect() * latitude_span
        )
        return width / geographic_width_to_height

    def metadata_description(self) -> str:
        """Describe the coordinate reference system and bounding box."""
        return (
            f"{self.crs} bounds: "
            f"west={self.west}, east={self.east}, "
            f"south={self.south}, north={self.north}."
        )


@dataclass(frozen=True)
class GeographicLabel:
    """A map label positioned by the center of its text."""

    text: str
    longitude: float
    latitude: float
    color: str = MAP_LABEL_COLOR
    rotation: float = 0


SKIWAY_MAP_BOUNDS = GeographicBounds(
    west=-72.1072,
    east=-72.0859,
    south=43.7776,
    north=43.7903,
)
"""Editable fixed map extent, stored in WGS 84 longitude and latitude."""

SKIWAY_FIGURE_WIDTH = 8.0
"""Fixed figure width in inches; height adapts to `SKIWAY_MAP_BOUNDS`."""

SKIWAY_MAP_LABELS = (
    GeographicLabel(
        text="Grafton Turnpike",
        longitude=-72.0977,
        latitude=43.7847,
        color=ROAD_LABEL_COLOR,
        rotation=-69,
    ),
    GeographicLabel(
        text="Holt's Ledge",
        longitude=-72.1047,
        latitude=43.7778,
    ),
    GeographicLabel(
        text="Winslow Mountain",
        longitude=-72.0879,
        latitude=43.7833,
    ),
)
"""Editable map labels whose coordinates specify each text bounding-box center."""


def load_skiway_segments() -> pl.DataFrame:
    """
    Segments between consecutive coordinates of Dartmouth Skiway runs.
    Segment attributes like bearing are stored on the segment's ending
    coordinate, hence the backwards shift to combine them with the
    starting coordinate.
    """
    from openskistats.analyze import load_runs_pl

    return (
        load_runs_pl(ski_area_filters=[pl.col("ski_area_name") == SKIWAY_NAME])
        .select("run_id", "run_coordinates_clean")
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .sort("run_id", "index")
        .select(
            "run_id",
            "longitude",
            "latitude",
            longitude_end=pl.col("longitude").shift(-1).over("run_id"),
            latitude_end=pl.col("latitude").shift(-1).over("run_id"),
            bearing=pl.col("bearing").shift(-1).over("run_id"),
        )
        .drop_nulls()
        .collect()
    )


def load_skiway_bearings(num_bins: int = 32) -> pl.DataFrame:
    from openskistats.analyze import load_bearing_distribution_pl

    return load_bearing_distribution_pl(
        ski_area_filters=[pl.col("ski_area_name") == SKIWAY_NAME]
    ).filter(pl.col("num_bins") == num_bins)


def load_skiway_lift_coordinates() -> pl.DataFrame:
    """Coordinates for lifts belonging to Dartmouth Skiway."""
    from openskistats.analyze import load_lifts_pl, load_ski_areas_pl

    ski_area_ids = (
        load_ski_areas_pl(ski_area_filters=[pl.col("ski_area_name") == SKIWAY_NAME])
        .select("ski_area_id")
        .lazy()
    )
    return (
        load_lifts_pl()
        .lazy()
        .explode("ski_area_ids", empty_as_null=False, keep_nulls=False)
        .rename({"ski_area_ids": "ski_area_id"})
        .join(ski_area_ids, on="ski_area_id", how="semi", maintain_order="left")
        .select("lift_id", "lift_name", "lift_type", "lift_coordinates")
        .explode("lift_coordinates", empty_as_null=False, keep_nulls=False)
        .unnest("lift_coordinates")
        .select(
            "lift_id",
            "lift_name",
            "lift_type",
            "index",
            "longitude",
            "latitude",
            "elevation",
        )
        .drop_nulls(["lift_id", "index", "longitude", "latitude"])
        .sort("lift_id", "index")
        .collect()
    )


def _plot_skiway_map_context(ax: Axes, map_context: dict[str, Any]) -> None:
    """Plot the static road and lodge snapshot behind the ski data."""
    for feature in map_context["features"]:
        coordinates = feature["geometry"]["coordinates"]
        match feature["properties"]["feature_kind"]:
            case "road":
                ax.plot(
                    [coordinate[0] for coordinate in coordinates],
                    [coordinate[1] for coordinate in coordinates],
                    color=ROAD_COLOR,
                    linewidth=3,
                    solid_capstyle="round",
                    zorder=0,
                )
            case "lodge":
                polygon_coordinates = coordinates[0]
                ax.add_patch(
                    Polygon(
                        polygon_coordinates,
                        closed=True,
                        facecolor=LODGE_COLOR,
                        edgecolor="none",
                        zorder=1.2,
                    )
                )


def _plot_skiway_map_labels(ax: Axes) -> None:
    """Plot labels at their declarative geographic positions."""
    for label in SKIWAY_MAP_LABELS:
        ax.text(
            label.longitude,
            label.latitude,
            label.text,
            color=label.color,
            fontsize=8,
            ha="center",
            va="center",
            rotation=label.rotation,
            zorder=4,
        )


def plot_skiway_segments_with_rose(
    highlight_bin_label: str = "NNE",
    num_bins: int = 32,
    bearings: pl.DataFrame | None = None,
    lift_coordinates: pl.DataFrame | None = None,
    map_context: dict[str, Any] | None = None,
) -> Figure:
    """
    Plot Dartmouth Skiway lifts underneath run-segment arrows.

    Color the arrows by whether their bearing falls in the highlighted compass bin,
    with an inset ski rose highlighting that bin's petal.
    """
    if bearings is None:
        bearings = load_skiway_bearings(num_bins=num_bins)
    if lift_coordinates is None:
        lift_coordinates = load_skiway_lift_coordinates()
    if map_context is None:
        map_context = load_dartmouth_skiway_context()
    (highlight_bin_index,) = bearings.filter(
        pl.col("bin_label") == highlight_bin_label
    )["bin_index"]
    segments = load_skiway_segments().with_columns(
        highlight=cut_bearings_pl(num_bins=num_bins) == highlight_bin_index
    )
    # a figure unmanaged by pyplot to avoid spawning interactive backend windows
    fig = Figure(
        figsize=(
            SKIWAY_FIGURE_WIDTH,
            SKIWAY_MAP_BOUNDS.height_for_width(SKIWAY_FIGURE_WIDTH),
        )
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(SKIWAY_MAP_BOUNDS.west, SKIWAY_MAP_BOUNDS.east)
    ax.set_ylim(SKIWAY_MAP_BOUNDS.south, SKIWAY_MAP_BOUNDS.north)
    # The adaptive canvas height lets the map fill the canvas at true local proportions.
    ax.set_aspect(SKIWAY_MAP_BOUNDS.local_data_aspect())
    _plot_skiway_map_context(ax=ax, map_context=map_context)
    for lift in lift_coordinates.partition_by("lift_id", maintain_order=True):
        ax.plot(
            lift["longitude"],
            lift["latitude"],
            color=LIFT_COLOR,
            linewidth=4,
            solid_capstyle="round",
            zorder=1,
        )
    for row in segments.iter_rows(named=True):
        ax.annotate(
            "",
            xy=(row["longitude_end"], row["latitude_end"]),
            xytext=(row["longitude"], row["latitude"]),
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "->",
                "color": HIGHLIGHT_COLOR if row["highlight"] else MUTED_COLOR,
                "linewidth": 1.5,
                "mutation_scale": 10,
                "shrinkA": 0,
                "shrinkB": 0,
                "zorder": 3 if row["highlight"] else 2,
            },
        )
    _plot_skiway_map_labels(ax=ax)
    rose_ax = fig.add_axes((0.4, 0.04, 0.4, 0.5), projection="polar")
    plot_orientation(
        bin_counts=bearings["bin_count"].to_numpy(),
        bin_centers=bearings["bin_center"].to_numpy(),
        ax=rose_ax,  # type: ignore[arg-type]
        color=MUTED_COLOR,
        margin_text={},
    )
    # opaque circle masks the contextual map layers beneath the rose
    rose_ax.patch.set_facecolor("white")
    rose_ax.patch.set_alpha(1.0)
    (highlight_patch,) = (
        patch
        for patch, bin_index in zip(rose_ax.patches, bearings["bin_index"], strict=True)
        if bin_index == highlight_bin_index
    )
    assert isinstance(highlight_patch, Rectangle)
    highlight_patch.set_facecolor(HIGHLIGHT_COLOR)
    highlight_row = bearings.row(
        by_predicate=pl.col("bin_index") == highlight_bin_index, named=True
    )
    # read the petal's radius from the bar itself rather than re-deriving
    # plot_orientation's area-scaling formula
    radius = highlight_patch.get_height()
    rose_ax.text(
        x=math.radians(highlight_row["bin_center"]),
        y=radius * 0.78,
        s=highlight_bin_label,
        color="white",
        size=9,
        weight="bold",
        ha="center",
        va="center",
        # radial rotation so the label runs along the narrow petal
        rotation=90 - highlight_row["bin_center"],
        rotation_mode="anchor",
    )
    return fig
