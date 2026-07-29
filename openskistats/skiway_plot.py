"""
Dartmouth Skiway manuscript figure:
run segments with one compass direction highlighted and an inset ski rose.
Ported from the retired R implementation at
https://github.com/dhimmel/openskistats/blob/b01f47defbdb0119d76aed25235fa6d0cb887e92/r/02.plot.R.
"""

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import ArrowStyle, Polygon, Rectangle
from matplotlib.path import Path as MatplotlibPath

from openskistats.bearing import cut_bearings_pl
from openskistats.geometry import simplify_segments
from openskistats.plot import plot_orientation
from openskistats.skiway_data import (
    SKIWAY_MAP_BOUNDS,
    load_dartmouth_skiway_context,
    load_dartmouth_skiway_contours,
)

MUTED_COLOR = "#94a3b8"
SKIWAY_HIGHLIGHT_BIN_COLORS = {
    "NNE": "#6d28d9",
    "NWbW": "#15803d",
}
"""
Compass bins to highlight with their colors:
the overall modal bin and the modal bin of the northwest-facing ledge.
"""
SKIWAY_ARROW_ALPHA = 0.75
"""Arrow translucency so overplotted segments darken where they stack."""
SKIWAY_HIGHLIGHT_TAIL_WIDTH = 2.7
SKIWAY_MUTED_TAIL_WIDTH = 1.8
"""Arrow shaft widths in points."""
SKIWAY_ARROW_HEAD_WIDTH_RATIO = 2.6
SKIWAY_ARROW_HEAD_LENGTH_RATIO = 2.0
"""
Arrowhead dimensions as multiples of the shaft width,
so wider highlight arrows keep proportional heads.
A width ratio exceeding the length ratio makes a stout head
with a wide angle at the point.
"""
LIFT_COLOR = "#dceaf2"
LODGE_COLOR = "#f2d49b"
ROAD_COLOR = LODGE_COLOR
ROAD_LABEL_COLOR = ROAD_COLOR
ROAD_LINEWIDTH = 3
TRAIL_COLOR = ROAD_COLOR
TRAIL_LINEWIDTH = ROAD_LINEWIDTH / 3
APPALACHIAN_TRAIL_MARKER_SIZE = 80
PARKING_COLOR = "#f7e5c2"
MAP_LABEL_COLOR = "#4d5961"
CONTOUR_COLOR = "#ebe4da"
INDEX_CONTOUR_COLOR = "#d7cab9"
SKIWAY_ARROW_SIMPLIFICATION_TOLERANCE_METERS = 0.5

SKIWAY_FIGURE_NAME = "dartmouth_nne_light"
SKIWAY_NAME = "Dartmouth Skiway"
"""Ski area for the example figure, chosen for its two distinctly oriented ledges."""


@dataclass(frozen=True)
class GeographicLabel:
    """A map label positioned by the center of its text."""

    text: str
    longitude: float
    latitude: float
    color: str = MAP_LABEL_COLOR
    rotation: float = 0


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
        longitude=-72.1049,
        latitude=43.7777,
    ),
    GeographicLabel(
        text="Winslow\nMountain",
        longitude=-72.0870,
        latitude=43.7833,
    ),
)
"""Editable map labels whose coordinates specify each text bounding-box center."""

APPALACHIAN_TRAIL_MARKER_COORDINATES = (-72.1052, 43.7898)


@cache
def get_appalachian_trail_marker() -> MatplotlibPath:
    """
    Get a Matplotlib path for the filled, interlocked AT letterform.

    Source:
    <https://upload.wikimedia.org/wikipedia/commons/2/24/Appalachian_Trail_Marker_Logo.svg>,
    a public-domain/CC0 vector based on the National Park Service trail marker.
    The letterform's SVG path was converted once using `svgpath2mpl.parse_path`,
    following the same workflow as `get_snowflake_marker`.
    """
    path = MatplotlibPath(
        vertices=[
            [106.49, 102.49],
            [91.355, 127.695],
            [96.966, 127.695],
            [99.6051, 123.3878],
            [105.9132, 123.3878],
            [105.9132, 140.6158],
            [110.7661, 140.6158],
            [110.7661, 123.3878],
            [117.3181, 123.3878],
            [119.8657, 127.695],
            [125.7196, 127.695],
            [110.5546, 102.49],
            [106.49, 102.49],
            [106.49, 102.49],
            [108.4909, 107.9889],
            [115.0047, 119.2699],
            [101.9577, 119.2699],
            [108.4909, 107.9889],
            [108.4909, 107.9889],
        ],
        codes=[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 79, 1, 2, 2, 2, 79],
    )
    vertices = np.asarray(path.vertices, dtype=float).copy()
    vertices[:, 1] *= -1
    return MatplotlibPath(
        vertices=vertices - np.mean(vertices, axis=0),
        codes=path.codes,
    )


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


class RoundTailArrowStyle(ArrowStyle._Base):  # noqa: SLF001
    """
    Arrow drawn as a single filled polygon:
    a straight shaft with a rounded tail cap, like a round-capped stroke,
    and a triangular head whose tip and barb corners are also rounded
    on the scale of the shaft radius.
    A single fill keeps translucency uniform across the whole arrow,
    unlike stroked arrow styles whose overlapping shaft and barb strokes
    darken at the tip when drawn with alpha.
    Dimensions are in points when `mutation_scale` is 1.
    """

    # cubic bezier control-point offset approximating a quarter circle
    _QUARTER_CIRCLE_BEZIER = 0.5522847498

    def __init__(
        self, tail_width: float, head_width: float, head_length: float
    ) -> None:
        self.tail_width = tail_width
        self.head_width = head_width
        self.head_length = head_length
        super().__init__()

    def transmute(
        self, path: MatplotlibPath, mutation_size: float, linewidth: float
    ) -> tuple[MatplotlibPath, bool]:
        endpoints = np.asarray(path.vertices, dtype=np.float64)
        start_x, start_y = map(float, endpoints[0])
        end_x, end_y = map(float, endpoints[-1])
        length = math.hypot(end_x - start_x, end_y - start_y)
        if length == 0:
            return MatplotlibPath([(end_x, end_y)]), False
        unit_x, unit_y = (end_x - start_x) / length, (end_y - start_y) / length
        normal_x, normal_y = -unit_y, unit_x
        tail_radius = self.tail_width * mutation_size / 2
        head_half_width = self.head_width * mutation_size / 2
        head_length = min(self.head_length * mutation_size, length)
        base_x = end_x - unit_x * head_length
        base_y = end_y - unit_y * head_length
        tail_left = (start_x + normal_x * tail_radius, start_y + normal_y * tail_radius)
        tail_back = (start_x - unit_x * tail_radius, start_y - unit_y * tail_radius)
        tail_right = (
            start_x - normal_x * tail_radius,
            start_y - normal_y * tail_radius,
        )
        shaft_left = (base_x + normal_x * tail_radius, base_y + normal_y * tail_radius)
        shaft_right = (base_x - normal_x * tail_radius, base_y - normal_y * tail_radius)
        barb_right = (
            base_x - normal_x * head_half_width,
            base_y - normal_y * head_half_width,
        )
        barb_left = (
            base_x + normal_x * head_half_width,
            base_y + normal_y * head_half_width,
        )
        tip = (end_x, end_y)
        slant_length = math.hypot(tip[0] - barb_right[0], tip[1] - barb_right[1])
        toward_tip_right = (
            (tip[0] - barb_right[0]) / slant_length,
            (tip[1] - barb_right[1]) / slant_length,
        )
        toward_tip_left = (
            (tip[0] - barb_left[0]) / slant_length,
            (tip[1] - barb_left[1]) / slant_length,
        )
        # sharp corners are trimmed by this distance along each adjacent edge
        # and rounded with a quadratic bezier controlled by the corner point
        trim = min(tail_radius, head_half_width - tail_radius, slant_length / 3)
        bezier = self._QUARTER_CIRCLE_BEZIER * tail_radius
        vertices = [
            shaft_left,
            tail_left,
            # tail cap: two quarter-circle cubic arcs bulging behind the start
            (tail_left[0] - unit_x * bezier, tail_left[1] - unit_y * bezier),
            (tail_back[0] + normal_x * bezier, tail_back[1] + normal_y * bezier),
            tail_back,
            (tail_back[0] - normal_x * bezier, tail_back[1] - normal_y * bezier),
            (tail_right[0] - unit_x * bezier, tail_right[1] - unit_y * bezier),
            tail_right,
            shaft_right,
            (barb_right[0] + normal_x * trim, barb_right[1] + normal_y * trim),
            barb_right,
            (
                barb_right[0] + toward_tip_right[0] * trim,
                barb_right[1] + toward_tip_right[1] * trim,
            ),
            (tip[0] - toward_tip_right[0] * trim, tip[1] - toward_tip_right[1] * trim),
            tip,
            (tip[0] - toward_tip_left[0] * trim, tip[1] - toward_tip_left[1] * trim),
            (
                barb_left[0] + toward_tip_left[0] * trim,
                barb_left[1] + toward_tip_left[1] * trim,
            ),
            barb_left,
            (barb_left[0] - normal_x * trim, barb_left[1] - normal_y * trim),
            shaft_left,
        ]
        codes = [
            MatplotlibPath.MOVETO,
            MatplotlibPath.LINETO,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.LINETO,
            MatplotlibPath.LINETO,
            MatplotlibPath.CURVE3,
            MatplotlibPath.CURVE3,
            MatplotlibPath.LINETO,
            MatplotlibPath.CURVE3,
            MatplotlibPath.CURVE3,
            MatplotlibPath.LINETO,
            MatplotlibPath.CURVE3,
            MatplotlibPath.CURVE3,
            MatplotlibPath.CLOSEPOLY,
        ]
        return MatplotlibPath(vertices, codes), True


def _blend_on_white(color: str, alpha: float) -> str:
    """
    Solid color equivalent to drawing `color` at `alpha` over a white background.
    Used for the rose petals to match the translucent arrows
    without underlying grid lines showing through.
    """
    red, green, blue = to_rgb(color)
    return to_hex(
        (
            alpha * red + 1 - alpha,
            alpha * green + 1 - alpha,
            alpha * blue + 1 - alpha,
        )
    )


def _plot_skiway_map_context(ax: Axes, map_context: dict[str, Any]) -> None:
    """Plot the static OpenStreetMap context behind the ski data."""
    for feature in map_context["features"]:
        coordinates = feature["geometry"]["coordinates"]
        match feature["properties"]["feature_kind"]:
            case "road" | "parking_road":
                is_parking_road = (
                    feature["properties"]["feature_kind"] == "parking_road"
                )
                ax.plot(
                    [coordinate[0] for coordinate in coordinates],
                    [coordinate[1] for coordinate in coordinates],
                    color=PARKING_COLOR if is_parking_road else ROAD_COLOR,
                    linewidth=ROAD_LINEWIDTH,
                    solid_capstyle="round",
                    zorder=0,
                )
            case "trail":
                for line_coordinates in coordinates:
                    ax.plot(
                        [coordinate[0] for coordinate in line_coordinates],
                        [coordinate[1] for coordinate in line_coordinates],
                        color=TRAIL_COLOR,
                        linewidth=TRAIL_LINEWIDTH,
                        solid_capstyle="round",
                        zorder=0,
                    )
            case "lodge" | "parking":
                polygon_coordinates = coordinates[0]
                is_lodge = feature["properties"]["feature_kind"] == "lodge"
                ax.add_patch(
                    Polygon(
                        polygon_coordinates,
                        closed=True,
                        facecolor=LODGE_COLOR if is_lodge else PARKING_COLOR,
                        edgecolor="none",
                        zorder=1.2 if is_lodge else -0.5,
                    )
                )


def _plot_skiway_contours(ax: Axes, contours: dict[str, Any]) -> None:
    """Plot static elevation contours beneath the other map geometry."""
    for feature in contours["features"]:
        is_index = feature["properties"]["is_index"]
        for coordinates in feature["geometry"]["coordinates"]:
            ax.plot(
                [coordinate[0] for coordinate in coordinates],
                [coordinate[1] for coordinate in coordinates],
                color=INDEX_CONTOUR_COLOR if is_index else CONTOUR_COLOR,
                linewidth=0.7 if is_index else 0.35,
                zorder=-1,
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
    ax.scatter(
        *APPALACHIAN_TRAIL_MARKER_COORDINATES,
        marker=get_appalachian_trail_marker(),
        s=APPALACHIAN_TRAIL_MARKER_SIZE,
        color=TRAIL_COLOR,
        edgecolors="none",
        zorder=4,
    )


def plot_skiway_segments_with_rose(
    highlight_bin_colors: dict[str, str] | None = None,
    num_bins: int = 32,
    arrow_simplification_tolerance_meters: float = (
        SKIWAY_ARROW_SIMPLIFICATION_TOLERANCE_METERS
    ),
    bearings: pl.DataFrame | None = None,
    lift_coordinates: pl.DataFrame | None = None,
    map_context: dict[str, Any] | None = None,
    contours: dict[str, Any] | None = None,
) -> Figure:
    """
    Plot Dartmouth Skiway lifts underneath run-segment arrows.

    Color the arrows by whether their bearing falls in a highlighted compass bin,
    with an inset ski rose coloring and labeling the matching petals.
    """
    if highlight_bin_colors is None:
        highlight_bin_colors = SKIWAY_HIGHLIGHT_BIN_COLORS
    if bearings is None:
        bearings = load_skiway_bearings(num_bins=num_bins)
    if lift_coordinates is None:
        lift_coordinates = load_skiway_lift_coordinates()
    if map_context is None:
        map_context = load_dartmouth_skiway_context()
    if contours is None:
        contours = load_dartmouth_skiway_contours()
    bin_index_to_color = {
        bearings.row(by_predicate=pl.col("bin_label") == label, named=True)[
            "bin_index"
        ]: color
        for label, color in highlight_bin_colors.items()
    }
    segments = load_skiway_segments().with_columns(
        color=cut_bearings_pl(num_bins=num_bins).replace_strict(
            bin_index_to_color, default=MUTED_COLOR, return_dtype=pl.String
        )
    )
    plot_segments = simplify_segments(
        segments=segments,
        group_columns=["run_id", "color"],
        tolerance_meters=arrow_simplification_tolerance_meters,
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
    _plot_skiway_contours(ax=ax, contours=contours)
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
    for row in plot_segments.iter_rows(named=True):
        highlighted = row["color"] != MUTED_COLOR
        tail_width = (
            SKIWAY_HIGHLIGHT_TAIL_WIDTH if highlighted else SKIWAY_MUTED_TAIL_WIDTH
        )
        ax.annotate(
            "",
            xy=(row["longitude_end"], row["latitude_end"]),
            xytext=(row["longitude"], row["latitude"]),
            annotation_clip=False,
            arrowprops={
                "arrowstyle": RoundTailArrowStyle(
                    tail_width=tail_width,
                    head_width=tail_width * SKIWAY_ARROW_HEAD_WIDTH_RATIO,
                    head_length=tail_width * SKIWAY_ARROW_HEAD_LENGTH_RATIO,
                ),
                "color": to_rgba(row["color"], alpha=SKIWAY_ARROW_ALPHA),
                "linewidth": 0,
                # annotate defaults mutation_scale to the font size;
                # pin to 1 so the arrow dimensions are in points
                "mutation_scale": 1,
                "shrinkA": 0,
                # stop the tip short of the following arrow's round tail cap,
                # which bulges backward by the tail radius
                "shrinkB": tail_width / 2,
                "zorder": 3 if highlighted else 2,
            },
        )
    _plot_skiway_map_labels(ax=ax)
    rose_ax = fig.add_axes((0.4, 0.04, 0.4, 0.5), projection="polar")
    plot_orientation(
        bin_counts=bearings["bin_count"].to_numpy(),
        bin_centers=bearings["bin_center"].to_numpy(),
        ax=rose_ax,  # type: ignore[arg-type]
        color=_blend_on_white(MUTED_COLOR, alpha=SKIWAY_ARROW_ALPHA),
        margin_text={},
    )
    # opaque circle masks the contextual map layers beneath the rose
    rose_ax.patch.set_facecolor("white")
    rose_ax.patch.set_alpha(1.0)
    for label, color in highlight_bin_colors.items():
        bin_row = bearings.row(by_predicate=pl.col("bin_label") == label, named=True)
        (petal,) = (
            patch
            for patch, bin_index in zip(
                rose_ax.patches, bearings["bin_index"], strict=True
            )
            if bin_index == bin_row["bin_index"]
        )
        assert isinstance(petal, Rectangle)
        petal.set_facecolor(_blend_on_white(color, alpha=SKIWAY_ARROW_ALPHA))
        # radial rotation so the label runs along the narrow petal,
        # flipped on west-side petals to keep the text reading upward
        rotation = 90 - bin_row["bin_center"]
        if bin_row["bin_center"] > 180:
            rotation += 180
        # read the petal's radius from the bar itself rather than re-deriving
        # plot_orientation's area-scaling formula
        radius = petal.get_height()
        rose_ax.text(
            x=math.radians(bin_row["bin_center"]),
            y=radius * 0.78,
            s=label,
            color="white",
            size=9,
            weight="bold",
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
        )
    return fig
