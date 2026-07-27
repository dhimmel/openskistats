"""
US state tile map of ski roses for the manuscript ("Which way do you ski?").
Ported from the retired R implementation in r/02.plot.R.
"""

import math

import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.projections.polar import PolarAxes

from openskistats.plot import plot_orientation
from openskistats.utils import get_images_data_directory

ROSE_COLOR = "#f07178"
PETAL_EDGE_COLOR = "#EBEBEB"
CIRCLE_FILL = "#E5E5E5"
CIRCLE_COLOR = "#7F7F7F"
TITLE_COLOR = "#1A1A1A"
ANNOTATION_COLOR = "#4D4D4D"

NUM_BINS = 16
MAX_BORDER_WIDTH = 8.2
MIN_BORDER_WIDTH = 0.27

FIGURE_SIZE = (33.0, 24.0)
EXPORT_PADDING_INCHES = 0.02
STATE_LABEL_SIZE = 28.0
COMPASS_LABEL_SIZE = 24.0
TITLE_SIZE = 81.0
SUBTITLE_SIZE = 45.0
ANNOTATION_SIZE = 28.0

SUBPLOT_MARGIN_X = 0.013
SUBPLOT_MARGIN_Y = 0.017
SUBPLOT_SPACING = 0.31

US_STATE_TO_ABBREVIATION = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}  # fmt: skip

US_TILE_LAYOUT = {
    # US state tile layout slightly modified from NPR: move HI and AK in,
    # switch MA and RI. Four corners are preserved.
    # https://blog.apps.npr.org/2015/05/11/hex-tile-maps.html
    "AK": (0, 0), "ME": (0, 10),
    "VT": (1, 9), "NH": (1, 10),
    "WA": (2, 0), "ID": (2, 1), "MT": (2, 2), "ND": (2, 3), "MN": (2, 4),
    "IL": (2, 5), "WI": (2, 6), "MI": (2, 7), "NY": (2, 8), "MA": (2, 9), "RI": (2, 10),
    "OR": (3, 0), "NV": (3, 1), "WY": (3, 2), "SD": (3, 3), "IA": (3, 4),
    "IN": (3, 5), "OH": (3, 6), "PA": (3, 7), "NJ": (3, 8), "CT": (3, 9),
    "CA": (4, 0), "UT": (4, 1), "CO": (4, 2), "NE": (4, 3), "MO": (4, 4),
    "KY": (4, 5), "WV": (4, 6), "VA": (4, 7), "MD": (4, 8), "DE": (4, 9),
    "AZ": (5, 1), "NM": (5, 2), "KS": (5, 3), "AR": (5, 4), "TN": (5, 5),
    "NC": (5, 6), "SC": (5, 7), "DC": (5, 8),
    "OK": (6, 3), "LA": (6, 4), "MS": (6, 5), "AL": (6, 6), "GA": (6, 7),
    "HI": (7, 0), "TX": (7, 3), "FL": (7, 8),
}  # fmt: skip

TITLE_KEY = "_title"
TITLE_ROWS = slice(0, 2)
TITLE_COLS = slice(2, 9)


def load_us_state_roses(num_bins: int = NUM_BINS) -> pl.DataFrame:
    path = get_images_data_directory().joinpath("region_roses.parquet")
    return (
        pl.read_parquet(path)
        .with_columns(
            abbreviation=pl.col("region").replace_strict(US_STATE_TO_ABBREVIATION),
            border_width=(
                pl.col("combined_vertical") / pl.col("combined_vertical").max()
            )
            .sqrt()
            .mul(MAX_BORDER_WIDTH)
            .add(MIN_BORDER_WIDTH),
        )
        .select("region", "abbreviation", "ski_areas_count", "border_width", "bearings")
        .explode("bearings", empty_as_null=False, keep_nulls=False)
        .unnest("bearings")
        .filter(pl.col("num_bins") == num_bins)
    )


def _style_state_axes(ax: PolarAxes, abbreviation: str, border_width: float) -> None:
    ax.set_facecolor(CIRCLE_FILL)
    ax.grid(False)
    ax.spines["polar"].set_linewidth(border_width)
    ax.spines["polar"].set_color(CIRCLE_COLOR)
    top = ax.get_ylim()[1]
    ax.text(
        x=math.pi,
        y=top,
        s=abbreviation,
        size=STATE_LABEL_SIZE,
        color=TITLE_COLOR,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": CIRCLE_FILL,
            "alpha": 0.8,
            "edgecolor": TITLE_COLOR,
            "linewidth": 0.55,
        },
        zorder=4,
    )


def _plot_empty_state(ax: PolarAxes, abbreviation: str) -> None:
    """States without ski areas: a filled circle with no rose nor border."""
    ax.set_yticks([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    if abbreviation == "FL":
        # a single empty state acts as the compass legend for the whole map
        ax.set_xticks([math.radians(bearing) for bearing in (0, 90, 180, 270)])
        ax.set_xticklabels(
            labels=["N", "E", "", "W"],
            fontdict={"size": COMPASS_LABEL_SIZE, "color": ANNOTATION_COLOR},
        )
    else:
        ax.set_xticks([])
    _style_state_axes(ax=ax, abbreviation=abbreviation, border_width=0)


def plot_us_state_roses() -> Figure:
    """
    Tile map of US states in the shape of the country,
    where each state with ski areas gets a ski rose
    whose circular border width is proportional to the state's combined vertical drop.
    """
    roses = load_us_state_roses()
    fig = Figure(figsize=FIGURE_SIZE)
    mosaic: list[list[str]] = [["." for _ in range(11)] for _ in range(8)]
    for abbreviation, (row, col) in US_TILE_LAYOUT.items():
        mosaic[row][col] = abbreviation
    for row_index in range(TITLE_ROWS.start, TITLE_ROWS.stop):
        for col_index in range(TITLE_COLS.start, TITLE_COLS.stop):
            mosaic[row_index][col_index] = TITLE_KEY
    axes = fig.subplot_mosaic(  # type: ignore[misc]
        mosaic,  # type: ignore[arg-type]
        empty_sentinel=".",
        per_subplot_kw={tuple(US_TILE_LAYOUT): {"projection": "polar"}},
    )
    fig.subplots_adjust(
        left=SUBPLOT_MARGIN_X,
        right=1 - SUBPLOT_MARGIN_X,
        bottom=SUBPLOT_MARGIN_Y,
        top=1 - SUBPLOT_MARGIN_Y,
        wspace=SUBPLOT_SPACING,
        hspace=SUBPLOT_SPACING,
    )
    state_to_rose = roses.partition_by("abbreviation", as_dict=True)
    for abbreviation in US_TILE_LAYOUT:
        ax = axes[abbreviation]
        assert isinstance(ax, PolarAxes)
        state_pl = state_to_rose.get((abbreviation,))
        if state_pl is None:
            _plot_empty_state(ax=ax, abbreviation=abbreviation)
            continue
        plot_orientation(
            bin_counts=state_pl["bin_count"].to_numpy(),
            bin_centers=state_pl["bin_center"].to_numpy(),
            ax=ax,
            color=ROSE_COLOR,
            edgecolor=PETAL_EDGE_COLOR,
            linewidth=0.4,
            disable_xticks=True,
            margin_text={},
        )
        _style_state_axes(
            ax=ax,
            abbreviation=abbreviation,
            border_width=state_pl["border_width"].first(),
        )
    title_ax = axes[TITLE_KEY]
    title_ax.set_axis_off()
    title_ax.text(
        x=0.5, y=0.725, s="Which way do you ski?",
        size=TITLE_SIZE, color=TITLE_COLOR, ha="center", va="center",
        transform=title_ax.transAxes,
    )  # fmt: skip
    ski_areas_count = roses.group_by("abbreviation").first()["ski_areas_count"].sum()
    title_ax.text(
        x=0.5, y=0.40, s=f"Orientations of {ski_areas_count:,} US ski areas",
        size=SUBTITLE_SIZE, color=TITLE_COLOR, ha="center", va="center",
        transform=title_ax.transAxes,
    )  # fmt: skip
    fig.text(
        x=0.985, y=0.043, s="OpenSkiStats.org\nLicense: CC BY 4.0",
        size=ANNOTATION_SIZE, color=ANNOTATION_COLOR, ha="right", va="center",
        linespacing=1.6,
    )  # fmt: skip
    # border-width legend anchored to Montana's axes so it survives
    # the tight bounding box crop applied on save
    axes["MT"].annotate(
        text="border proportional to\ncombined vertical drop",
        xy=(0.28, 0.94),
        xytext=(-0.97, 1.45),
        xycoords="axes fraction",
        textcoords="axes fraction",
        size=ANNOTATION_SIZE,
        color=ANNOTATION_COLOR,
        ha="center",
        va="center",
        linespacing=1.4,
        arrowprops={
            "arrowstyle": "-",
            "color": CIRCLE_COLOR,
            "linewidth": 2,
            "connectionstyle": "arc3,rad=-0.25",
        },
        annotation_clip=False,
    )
    # The manuscript exporter uses bbox_inches="tight" for every figure. Include
    # the intended canvas in that bounding box, inset by the padding that the
    # exporter adds back, so this edge-to-edge tile map retains the same outer
    # margins, dimensions, and 11:8 aspect ratio as the R original.
    frame_x = EXPORT_PADDING_INCHES / FIGURE_SIZE[0]
    frame_y = EXPORT_PADDING_INCHES / FIGURE_SIZE[1]
    fig.add_artist(
        Rectangle(
            (frame_x, frame_y),
            1 - 2 * frame_x,
            1 - 2 * frame_y,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor="none",
        )
    )
    return fig
