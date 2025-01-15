import math
import textwrap
from enum import IntEnum
from functools import cache
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.projections.polar import PolarAxes
from matplotlib.text import Text as MplText
from osmnx.plot import _get_fig_ax

from openskistats.bearing import get_difficulty_color_to_bearing_bin_counts
from openskistats.sunlight import get_solar_location_band

SUBPLOT_FIGSIZE = 4.0
"""Size in inches for each subplot in a grid."""

NARROW_SPACE = "\u202f"  # NARROW NO-BREAK SPACE"


class MarginTextLocation(IntEnum):
    """Enum to map margin text locations to their degree x-values."""

    top_left = 315
    top_right = 45
    bottom_left = 225
    bottom_right = 135

    @property
    def radians(self) -> float:
        return math.radians(self.value)

    @property
    def vertical_alignment(self) -> str:
        return self.name.split("_")[0]

    @property
    def horizontal_alignment(self) -> str:
        return self.name.split("_")[1]


def plot_orientation(  # noqa: C901
    *,
    bin_counts: npt.NDArray[np.float64],
    bin_centers: npt.NDArray[np.float64],
    color_to_bin_counts: dict[str, npt.NDArray[np.float64]] | None = None,
    ax: PolarAxes | None = None,
    figsize: tuple[float, float] = (4.5, 4.5),
    max_bin_count: float | None = None,
    area: bool = True,
    color: str = "#D4A0A7",
    edgecolor: str = "#292929",
    linewidth: float = 0.5,
    alpha: float = 1.0,
    title: str | None = None,
    title_wrap: int | None = 30,
    title_y: float = 1.05,
    title_font_size: float = 22,
    disable_xticks: bool = False,
    xtick_font: dict[str, Any] | None = None,
    margin_text: dict[MarginTextLocation, str] | None = None,
) -> tuple[Figure, PolarAxes]:
    """
    Plot a polar histogram of an edge bearing distribution.

    Modified from the `osmnx.plot_orientation` source code to support consuming
    distribution information directly rather than a graph.
    https://github.com/gboeing/osmnx/blob/d614c9608165a614f796c7ab8e57257e7f9f1a63/osmnx/plot.py#L666-L803

    Parameters
    ----------
    color_to_bin_counts
        If not None, each bin_count is divided into respective groups.
        This should be a dictionary of HTML color to bin_counts for the group.
    ax
        If not None, plot on this pre-existing axes instance (must have
        projection=polar).
    figsize
        If `ax` is None, create new figure with size `(width, height)`.
    max_bin_count
        If not None, set the y-axis upper limit to this value.
        Useful for comparing multiple polar histograms on the same scale.
    area
        If True, set bar length so area is proportional to frequency.
        Otherwise, set bar length so height is proportional to frequency.
    color
        Color of the histogram bars.
    edgecolor
        Color of the histogram bar edges.
    linewidth
        Width of the histogram bar edges.
    alpha
        Opacity of the histogram bars.
    title
        The figure's title.
    title_y
        The y position to place `title`.
    xtick_font
        The xtick labels' `fontdict` to pass to matplotlib.

    Returns
    -------
    fig, ax
    """
    num_bins = len(bin_centers)
    if title and title_wrap is not None:
        title = "\n".join(textwrap.wrap(title, width=title_wrap))
    if xtick_font is None:
        xtick_font = {
            "family": "DejaVu Sans",
            "size": 10,
            "weight": "bold",
            "alpha": 1.0,
            "zorder": 3,
        }

    # positions: where to center each bar
    positions = np.radians(bin_centers)

    # width: make bars fill the circumference without gaps or overlaps
    width = 2 * np.pi / num_bins

    # radius: how long to make each bar. set bar length so either the bar area
    # (ie, via sqrt) or the bar height is proportional to the bin's frequency
    radius = np.sqrt(bin_counts * num_bins / np.pi) if area else bin_counts
    if max_bin_count is None:
        ylim = radius.max()
    else:
        ylim = np.sqrt(max_bin_count * num_bins / np.pi) if area else max_bin_count

    # create PolarAxes (if not passed-in) then set N at top and go clockwise
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize, bgcolor=None, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_ylim(top=ylim)

    # configure the y-ticks and remove their labels
    ax.set_yticks([])
    ax.set_yticklabels(labels="")

    # configure the x-ticks and their labels
    if disable_xticks:
        ax.set_xticks([])
    else:
        # this seemingly no-op line prevents
        # UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
        ax.set_xticks(ax.get_xticks())
        xticklabels = ["N", "", "E", "", "S", "", "W", ""]
        ax.set_xticklabels(labels=xticklabels, fontdict=xtick_font)
        ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars
    # FIXME: deduplicate bar drawing with a more streamlined handling of group_to_bin_counts versus ungrouped
    if not color_to_bin_counts:
        ax.bar(
            positions,
            height=radius,
            width=width,
            align="center",
            bottom=0,
            zorder=2,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
    else:
        cum_value = np.zeros_like(bin_centers, dtype=np.float64)
        cum_radii = np.zeros_like(bin_centers, dtype=np.float64)
        for group_color, group_bin_counts in color_to_bin_counts.items():
            cum_value = cum_value + group_bin_counts  # type: ignore [assignment]
            group_radii = (
                np.sqrt(cum_value * num_bins / np.pi) - cum_radii
                if area
                else group_bin_counts
            )
            ax.bar(
                positions,
                bottom=cum_radii,
                height=group_radii,
                width=width,
                align="center",
                zorder=2,
                facecolor=group_color,
                edgecolor=group_color,
                linewidth=0,
                alpha=alpha,
            )
            cum_radii = cum_radii + group_radii
        # draw the outline
        ax.bar(
            positions,
            height=cum_radii,
            width=width,
            align="center",
            bottom=0,
            zorder=2,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )

    if margin_text is None:
        margin_text = {
            MarginTextLocation.top_right: f"{bin_counts.sum():,.0f}{NARROW_SPACE}m\nskiable\nvert",
        }
    for location, text in margin_text.items():
        if not text:
            continue
        _mpl_add_polar_margin_text(ax=ax, location=location, text=text)

    if title:
        ax.set_title(
            title,
            y=title_y,
            # fallback font families for more character support https://matplotlib.org/stable/users/explain/text/fonts.html
            family=["DejaVu Sans", "Noto Sans CJK JP"],
            size=title_font_size,
            weight="bold",
            pad=13,
        )
    return fig, ax


def _mpl_add_polar_margin_text(
    ax: PolarAxes,
    location: MarginTextLocation,
    text: str,
    color: str = "#95A5A6",
) -> MplText:
    return ax.text(
        x=location.radians,
        y=ax.get_ylim()[1] * 1.58,
        s=text,
        verticalalignment=location.vertical_alignment,
        horizontalalignment=location.horizontal_alignment,
        multialignment=location.horizontal_alignment,
        color=color,
    )


def _generate_margin_text(group_info: dict[str, Any]) -> dict[MarginTextLocation, str]:
    # ideas for margin text
    # top-left: ski areas, country, trail, lift count
    # top-right: total skiable vertical
    # bottom-left: poleward affinity/tendency, polar affinity
    # bottom-right: total trail length and average pitch or max elevation
    margin_text = {}
    if {"ski_areas_count", "run_count", "lift_count"}.issubset(group_info):
        margin_text[MarginTextLocation.top_left] = (
            f"{group_info["ski_areas_count"]:,} ski areas\n{group_info["run_count"]:,} runs\n{group_info["lift_count"]:,} lifts"
        )
    elif {"run_count", "lift_count"}.issubset(group_info):
        margin_text[MarginTextLocation.top_left] = (
            f"{group_info["run_count"]:,} runs\n{group_info["lift_count"]:,} lifts"
        )
    if "combined_vertical" in group_info:
        margin_text[MarginTextLocation.top_right] = (
            f"{group_info["combined_vertical"]:,.0f}{NARROW_SPACE}m\nskiable\nvert"
        )
    if {"poleward_affinity", "eastward_affinity"}.issubset(group_info):
        margin_text[MarginTextLocation.bottom_left] = (
            f"affinity:\n{group_info['poleward_affinity']:.0%} poleward\n{group_info['eastward_affinity']:.0%} eastward"
        )
    if {"min_elevation", "max_elevation"}.issubset(group_info):
        margin_text[MarginTextLocation.bottom_right] = (
            f"elevation:\n{group_info['min_elevation']:,.0f}{NARROW_SPACE}m base\n{group_info['max_elevation']:,.0f}{NARROW_SPACE}m peak"
        )
    return margin_text


def _plot_solar_location_band(
    ax: PolarAxes,
    latitude: float,
    longitude: float,
    elevation: float,
) -> None:
    location_df = get_solar_location_band(
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
    )
    zenith_scaler = ax.get_rmax() / 90
    ax.fill_between(
        x=np.deg2rad(location_df["sun_azimuth_bin_center"]),
        y1=location_df["sun_zenith_min"] * zenith_scaler,
        y2=location_df["sun_zenith_max"] * zenith_scaler,
        color="#ffee9c",
        # alpha=0.5,
        zorder=1,
    )


def _plot_mean_bearing_as_snowflake(
    ax: PolarAxes,
    bearing: float,
    alignment: float,
) -> None:
    ax.scatter(
        x=np.radians(bearing),
        y=alignment * ax.get_ylim()[1],
        color="#9e2096",
        label="Mean Bearing",
        zorder=2,
        marker=get_snowflake_marker(),
        linewidths=0,
        s=230,  # marker size
        alpha=0.7,
    )


def subplot_orientations(
    groups_pl: pl.DataFrame,
    grouping_col: str,
    n_cols: int | None = None,
    free_y: bool = True,
    num_bins: int = 32,
    suptitle: str | None = None,
    sort_groups: bool = True,
    plot_solar_band: bool = False,
) -> plt.Figure:
    """
    Plot orientations from multiple graphs in a grid.
    https://github.com/gboeing/osmnx-examples/blob/bb870c225906db5a7b02c4c87a28095cb9dceb30/notebooks/17-street-network-orientations.ipynb

    Parameters
    ----------
    groups_pl
        A Polars DataFrame with one group per row and a bearings columns with the bearing distributions.
    grouping_col
        The column for naming each subplot/facet.
    n_cols
        The number of columns in the grid of subplots.
    free_y
        If True, each subplot's y-axis will be scaled independently.
        If False, all subplots will be scaled to the same maximum y-value.
    num_bins
        The number of bins in each polar histogram.
        This value must exist in the input groups_pl bearings.num_bins column.
    suptitle
        The figure's super title.
    """
    assert not groups_pl.select(grouping_col).is_duplicated().any()
    names = groups_pl.get_column(grouping_col).to_list()
    if sort_groups:
        names.sort()
    # create figure and axes
    n_subplots = len(names)
    if n_cols is None:
        n_cols = math.ceil(n_subplots**0.5)
    n_rows = math.ceil(n_subplots / n_cols)
    # adjusts the height of the figure to accommodate subplot titles
    figsize = (n_cols * SUBPLOT_FIGSIZE, n_rows * (SUBPLOT_FIGSIZE + 0.37))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
        figsize=figsize,
        subplot_kw={"projection": "polar"},
    )
    dists_pl = (
        groups_pl.select(grouping_col, "bearings")
        .explode("bearings")
        .unnest("bearings")
        .filter(pl.col("num_bins") == num_bins)
    )
    # plot each group's polar histogram
    max_bin_count = None if free_y else dists_pl.get_column("bin_count").max()
    for ax, name in zip(axes.flat, names, strict=False):
        group_info = groups_pl.row(
            by_predicate=pl.col(grouping_col) == name, named=True
        )
        margin_text = _generate_margin_text(group_info)
        group_dist_pl = dists_pl.filter(pl.col(grouping_col) == name)
        fig, ax = plot_orientation(
            bin_counts=group_dist_pl.get_column("bin_count").to_numpy(),
            bin_centers=group_dist_pl.get_column("bin_center").to_numpy(),
            color_to_bin_counts=get_difficulty_color_to_bearing_bin_counts(
                group_dist_pl
            ),
            ax=ax,
            max_bin_count=max_bin_count,
            title=name,
            title_font_size=18,
            area=True,
            color="#D4A0A7",
            margin_text=margin_text or None,
        )
        ax.yaxis.grid(False)
        if free_y and "bearing_mean" in group_info:
            _plot_mean_bearing_as_snowflake(
                ax=ax,
                bearing=group_info["bearing_mean"],
                alignment=group_info["bearing_alignment"],
            )
        if plot_solar_band:
            _plot_solar_location_band(
                ax=ax,
                latitude=group_info["latitude"],
                longitude=group_info["longitude"],
                elevation=group_info["min_elevation"],
            )

    # hide axes for unused subplots
    for ax in axes.flat[n_subplots:]:
        ax.axis("off")
    if suptitle:
        # FIXME: spacing is inconsistent based on number of subplot rows
        fig.suptitle(t=suptitle, y=0.99, fontsize=13, verticalalignment="bottom")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.31)
    plt.close()
    return fig


def plot_mean_bearing(
    ski_areas_pl: pl.DataFrame,
    *,
    ax: PolarAxes | None = None,
    figsize: tuple[float, float] = (4.5, 4.5),
    color: str = "#D4A0A7",
    edgecolor: str = "black",
    linewidth: float = 0.5,
    alpha: float = 0.7,
    title: str | None = None,
    title_y: float = 1.05,
    title_font: dict[str, Any] | None = None,
    xtick_font: dict[str, Any] | None = None,
) -> tuple[Figure, PolarAxes]:
    """
    Polar plot of mean bearings.
    """
    if title_font is None:
        title_font = {"family": "DejaVu Sans", "size": 24, "weight": "bold"}
    if xtick_font is None:
        xtick_font = {
            "family": "DejaVu Sans",
            "size": 10,
            "weight": "bold",
            "alpha": 1.0,
            "zorder": 3,
        }

    # bearings
    positions = (
        ski_areas_pl.select(pl.col("bearing_mean").radians().alias("bearing_mean_rad"))
        .get_column("bearing_mean_rad")
        .to_numpy()
    )
    magnitudes = ski_areas_pl.get_column("bearing_alignment").to_numpy()

    # create PolarAxes (if not passed-in) then set N at top and go clockwise
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize, bgcolor=None, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_ylim(top=1.0)

    # configure the y-ticks and remove their labels
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(labels="")

    # configure the x-ticks and their labels
    xticklabels = ["N", "", "E", "", "S", "", "W", ""]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=xticklabels, fontdict=xtick_font)
    ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars
    ax.scatter(
        x=positions,
        y=magnitudes,
        s=ski_areas_pl.get_column("combined_vertical").to_numpy() / 50,
        zorder=2,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )

    for i, ski_area_name in enumerate(ski_areas_pl.get_column("ski_area_name")):
        ski_area_name = textwrap.fill(ski_area_name, width=10)
        ax.annotate(
            text=ski_area_name,
            xy=(positions[i], magnitudes[i]),
            textcoords="offset points",
            xytext=(5, -5),
            ha="right",
            fontsize=4.5,
        )

    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)
    fig.tight_layout()
    return fig, ax


@cache
def get_snowflake_marker() -> MplPath:
    """
    Get a matplotlib path for a snowflake marker.
    NOTE: Only six sided snowflakes are scientifically accurate.
    Source: <https://www.svgrepo.com/svg/529894/snowflake> by Solar Icons under a CC BY license.
    Downloaded and merged multiple SVG paths in InkScape as per
    <https://graphicdesign.stackexchange.com/a/147508>.
    Then converted to a matplotlib path using `svgpath2mpl.parse_path` as per
    <https://petercbsmith.github.io/marker-tutorial.html>.
    """
    # fmt: off
    path = MplPath(
        vertices=[
            [400.0, 41.666], [400.0, 41.666], [393.37213066, 41.666], [387.00894182, 44.30171912], [382.32233047, 48.98833047], [377.63571912, 53.67494182], [375.0, 60.03813066], [375.0, 66.666], [375.0, 139.584], [317.709, 82.291],
            [317.709, 82.291], [315.38644408, 79.9601895], [312.6264127, 78.11067935], [309.58756784, 76.8487897], [306.54872298, 75.58690005], [303.29043213, 74.93728343], [300.0, 74.93728343], [296.70956787, 74.93728343], [293.45127702, 75.58690005], [290.41243216, 76.8487897],
            [287.3735873, 78.11067935], [284.61355592, 79.9601895], [282.291, 82.291], [282.291, 82.291], [279.9601895, 84.61355592], [278.11067935, 87.3735873], [276.8487897, 90.41243216], [275.58690005, 93.45127702], [274.93728343, 96.70956787], [274.93728343, 100.0], [274.93728343, 103.29043213], [275.58690005, 106.54872298],
            [276.8487897, 109.58756784], [278.11067935, 112.6264127], [279.9601895, 115.38644408], [282.291, 117.709], [375.0, 210.352], [375.0, 356.732], [248.31, 283.589], [214.326, 156.959], [214.326, 156.959], [212.61710612, 150.55577529], [208.43026503, 145.08774359], [202.69433507, 141.76799774], [196.95840512, 138.44825189],
            [190.1314348, 137.54189356], [183.728, 139.25], [183.728, 139.25], [177.32433105, 140.95887098], [171.85594013, 145.14602058], [168.53614477, 150.88241099], [165.21634942, 156.6188014], [164.31028947, 163.44628823], [166.019, 169.85], [187.048, 248.17], [123.831, 211.711], [123.831, 211.711], [120.98872354, 210.06346284], [117.84936826, 208.99222483], [114.59284405, 208.55867636], [111.33631984, 208.1251279],
            [108.02604638, 208.3377122], [104.85172852, 209.18424649], [101.67741066, 210.03078078], [98.70086714, 211.49477911], [96.09267744, 213.4923428], [93.48448774, 215.48990648], [91.29544545, 217.98213366], [89.651, 220.8262], [89.651, 220.8262], [88.00613417, 223.6723168], [86.93865695, 226.81511084], [86.50981596, 230.07426101], [86.08097496, 233.33341119],
            [86.29913185, 236.64536975], [87.15176964, 239.82010963], [88.00440743, 242.9948495], [89.47490117, 245.97046878], [91.47887974, 248.57623821], [93.48285831, 251.18200765], [95.98124763, 253.36711926], [98.8307, 255.0062], [162.0477, 291.4632], [83.7277, 312.4922], [83.7277, 312.4922], [77.32403105, 314.20107098], [71.85564013, 318.38822058],
            [68.53584477, 324.12461099], [65.21604942, 329.8610014], [64.30998947, 336.68848823], [66.0187, 343.0922], [66.0187, 343.0922], [66.87182576, 346.26509609], [68.34213257, 349.23890917], [70.34534349, 351.84318242], [72.34855441, 354.44745567], [74.84564971, 356.6314615], [77.69349767, 358.27001425], [80.54134563, 359.908567],
            [83.68447418, 360.96974997], [86.94272144, 361.39273692], [90.20096869, 361.81572387], [93.51086858, 361.5922756], [96.6827, 360.7352], [223.2427, 326.8152], [350.0127, 400.0222], [223.3127, 473.1702], [96.6827, 439.2502], [96.6827, 439.2502], [93.50649372, 438.394267], [90.19235936, 438.17358083], [86.93070737, 438.60082079],
            [83.66905539, 439.02806076], [80.52357049, 440.09488488], [77.67497354, 441.73999978], [74.82637659, 443.38511469], [72.33028727, 445.57639903], [70.33011603, 448.18795463], [68.3299448, 450.79951024], [66.86474558, 453.78034572], [66.0187, 456.9592], [66.0187, 456.9592], [64.31059356, 463.3626348], [65.21695189, 470.18960512],
            [68.53669774, 475.92553507], [71.85644359, 481.66146503], [77.32447529, 485.84830612], [83.7277, 487.5572], [162.0477, 508.5222], [98.8307, 545.0452], [98.8307, 545.0452], [93.09604894, 548.35414586], [88.90439422, 553.80932206], [87.18395341, 560.20271153], [85.4635126, 566.596101], [86.35140068, 573.41816604],
            [89.651, 579.1582], [89.651, 579.1582], [91.29008074, 582.00765237], [93.47519235, 584.50604169], [96.08096179, 586.51002026], [98.68673122, 588.51399883], [101.6623505, 589.98449257], [104.83709037, 590.83713036], [108.01183025, 591.68976815], [111.32378881, 591.90792504], [114.58293899, 591.47908404], [117.84208916, 591.05024305], [120.9848832, 589.98276583],
            [123.831, 588.3379], [187.048, 551.8149], [166.019, 630.1349], [166.019, 630.1349], [164.31028947, 636.53861177], [165.21634942, 643.3660986], [168.53614477, 649.10248901], [171.85594013, 654.83887942], [177.32433105, 659.02602902], [183.728, 660.7349], [183.728, 660.7349], [190.12246961, 662.44530082], [196.94154461, 661.54797405],
            [202.67584546, 658.24153997], [208.41014632, 654.93510589], [212.6028804, 649.48295402], [214.326, 643.0919], [248.246, 516.4619], [375.006, 443.2859], [375.006, 589.5759], [282.297, 682.2829], [282.297, 682.2829], [279.9661895, 684.60545592], [278.11667935, 687.3654873],
            [276.8547897, 690.40433216], [275.59290005, 693.44317702], [274.94328343, 696.70146787], [274.94328343, 699.9919], [274.94328343, 703.28233213], [275.59290005, 706.54062298], [276.8547897, 709.57946784], [278.11667935, 712.6183127], [279.9661895, 715.37834408], [282.297, 717.7009], [282.297, 717.7009], [284.61955592, 720.0317105], [287.3795873, 721.88122065],
            [290.41843216, 723.1431103], [293.45727702, 724.40499995], [296.71556787, 725.05461657], [300.006, 725.05461657], [303.29643213, 725.05461657], [306.55472298, 724.40499995], [309.59356784, 723.1431103], [312.6324127, 721.88122065], [315.39244408, 720.0317105], [317.715, 717.7009], [375.006, 660.4079], [375.006, 733.3259], [375.006, 733.3259],
            [375.006, 739.95376934], [377.64171912, 746.31695818], [382.32833047, 751.00356953], [387.01494182, 755.69018088], [393.37813066, 758.3259], [400.006, 758.3259], [400.006, 758.3259], [406.63386934, 758.3259], [412.99705818, 755.69018088], [417.68366953, 751.00356953], [422.37028088, 746.31695818], [425.006, 739.95376934],
            [425.006, 733.3259], [425.006, 660.4079], [482.297, 717.7009], [482.297, 717.7009], [484.61955592, 720.0317105], [487.3795873, 721.88122065], [490.41843216, 723.1431103], [493.45727702, 724.40499995], [496.71556787, 725.05461657], [500.006, 725.05461657], [503.29643213, 725.05461657], [506.55472298, 724.40499995], [509.59356784, 723.1431103], [512.6324127, 721.88122065], [515.39244408, 720.0317105],
            [517.715, 717.7009], [517.715, 717.7009], [520.0458105, 715.37834408], [521.89532065, 712.6183127], [523.1572103, 709.57946784], [524.41909995, 706.54062298], [525.06871657, 703.28233213], [525.06871657, 699.9919], [525.06871657, 696.70146787], [524.41909995, 693.44317702], [523.1572103, 690.40433216], [521.89532065, 687.3654873], [520.0458105, 684.60545592],
            [517.715, 682.2829], [425.006, 589.5759], [425.006, 443.3259], [551.766, 516.5269], [585.686, 643.0869], [585.686, 643.0869], [587.4091196, 649.47795402], [591.60185368, 654.93010589], [597.33615454, 658.23653997], [603.07045539, 661.54297405], [609.88953039, 662.44030082], [616.284, 660.7299], [616.284, 660.7299],
            [622.68766895, 659.02102902], [628.15605987, 654.83387942], [631.47585523, 649.09748901], [634.79565058, 643.3610986], [635.70171053, 636.53361177], [633.993, 630.1299], [612.964, 551.8759], [676.181, 588.3329], [676.181, 588.3329], [679.0271168, 589.97776583], [682.16991084, 591.04524305], [685.42906101, 591.47408404], [688.68821119, 591.90292504],
            [692.00016975, 591.68476815], [695.17490963, 590.83213036], [698.3496495, 589.97949257], [701.32526878, 588.50899883], [703.93103821, 586.50502026], [706.53680765, 584.50104169], [708.72191926, 582.00265237], [710.361, 579.1532], [710.361, 579.1532], [713.66059932, 573.41316604], [714.5484874, 566.591101], [712.82804659, 560.19771153], [711.10760578, 553.80432206], [706.91595106, 548.34914586],
            [701.1813, 545.0402], [637.9643, 508.5172], [716.2843, 487.5522], [716.2843, 487.5522], [722.68752471, 485.84330612], [728.15555641, 481.65646503], [731.47530226, 475.92053507], [734.79504811, 470.18460512], [735.70140644, 463.3576348], [733.9933, 456.9542], [733.9933, 456.9542], [733.14725442, 453.77534572], [731.6820552, 450.79451024], [729.68188397, 448.18295463], [727.68171273, 445.57139903],
            [725.18562341, 443.38011469], [722.33702646, 441.73499978], [719.48842951, 440.08988488], [716.34294461, 439.02306076], [713.08129263, 438.59582079], [709.81964064, 438.16858083], [706.50550628, 438.389267], [703.3293, 439.2452], [576.6993, 473.1652], [449.9793, 400.0052], [576.7693, 326.8102], [703.3293, 360.7302], [703.3293, 360.7302],
            [709.72404638, 362.44120415], [716.54363728, 361.54417553], [722.27839835, 358.23769221], [728.01315942, 354.93120889], [732.20620235, 349.47869834], [733.9293, 343.0872], [733.9293, 343.0872], [735.64023408, 336.69218221], [734.74293167, 329.87235246], [731.43604513, 324.13754215], [728.12915859, 318.40273183], [722.67616715, 314.20986721], [716.2843, 312.4872],
            [637.9643, 291.4582], [701.1813, 255.0012], [701.1813, 255.0012], [704.02536634, 253.35675455], [706.51759352, 251.16771226], [708.5151572, 248.55952256], [710.51272089, 245.95133286], [711.97671922, 242.97478934], [712.82325351, 239.80047148], [713.6697878, 236.62615362], [713.8823721, 233.31588016], [713.44882364, 230.05935595], [713.01527517, 226.80283174],
            [711.94403716, 223.66347646], [710.2965, 220.8212], [710.2965, 220.8212], [706.97677946, 215.09695113], [701.51648409, 210.91859433], [695.1235315, 209.21045886], [688.73057891, 207.50232339], [681.91391104, 208.40039365], [676.1815, 211.706], [612.9645, 248.229], [633.9295, 169.844], [633.9295, 169.844], [635.64043408, 163.44898221], [634.74313167, 156.62915246],
            [631.43624513, 150.89434215], [628.12935859, 145.15953183], [622.67636715, 140.96666721], [616.2845, 139.244], [616.2845, 139.244], [609.8810652, 137.53589356], [603.05409488, 138.44225189], [597.31816493, 141.76199774], [591.58223497, 145.08174359], [587.39539388, 150.54977529], [585.6865, 156.953], [551.7025, 283.583], [425.0125, 356.726],
            [425.0125, 210.346], [517.7215, 117.703], [517.7215, 117.703], [520.0523105, 115.38044408], [521.90182065, 112.6204127], [523.1637103, 109.58156784], [524.42559995, 106.54272298], [525.07521657, 103.28443213], [525.07521657, 99.994], [525.07521657, 96.70356787], [524.42559995, 93.44527702], [523.1637103, 90.40643216], [521.90182065, 87.3675873],
            [520.0523105, 84.60755592], [517.7215, 82.285], [517.7215, 82.285], [515.39894408, 79.9541895], [512.6389127, 78.10467935], [509.60006784, 76.8427897], [506.56122298, 75.58090005], [503.30293213, 74.93128343], [500.0125, 74.93128343], [496.72206787, 74.93128343], [493.46377702, 75.58090005], [490.42493216, 76.8427897],
            [487.3860873, 78.10467935], [484.62605592, 79.9541895], [482.3035, 82.285], [425.0125, 139.578], [425.0125, 66.66], [425.0125, 66.66], [425.0125, 60.03213066], [422.37678088, 53.66894182], [417.69016953, 48.98233047], [413.00355818, 44.29571912], [406.64036934, 41.66], [400.0125, 41.66], [400.0, 41.666], [400.0, 41.666],
        ],
        codes=[
            1, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4,
            4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4,
            4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4,
            4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4,
            4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 79,
        ],
    )
    # fmt: on
    # normalize path to have its center at (0, 0)
    return MplPath(
        vertices=path.vertices - np.mean(path.vertices, axis=0),
        codes=path.codes,
    )
