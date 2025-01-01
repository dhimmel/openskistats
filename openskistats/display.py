import unicodedata
from typing import Any

import htmltools
import IPython.display
import polars as pl
import reactable
from mizani.palettes import gradient_n_pal

from openskistats.analyze import (
    get_display_ski_area_filters,
    get_story_ski_area_names,
    load_bearing_distribution_pl,
    load_ski_areas_pl,
)
from openskistats.models import SkiAreaModel
from openskistats.plot import NARROW_SPACE
from openskistats.utils import get_website_source_directory


def embed_reactable_html() -> None:
    website_source_dir = get_website_source_directory()
    html_script = f"""
    <script>
    {website_source_dir.joinpath("ski-areas", "script.js").read_text()}
    </script>
    """
    html_style = f"""
    <style>
    {website_source_dir.joinpath("ski-areas", "style.css").read_text()}
    </style>
    """
    for html_ in html_style, html_script:
        IPython.display.display(IPython.display.HTML(html_))
    reactable.embed_css()


def country_code_to_emoji(country_code: str) -> str:
    assert len(country_code) == 2
    return unicodedata.lookup(
        f"REGIONAL INDICATOR SYMBOL LETTER {country_code[0]}"
    ) + unicodedata.lookup(f"REGIONAL INDICATOR SYMBOL LETTER {country_code[1]}")


def get_ski_area_frontend_table(story: bool = False) -> pl.DataFrame:
    """
    Dataframe of ski areas and their metrics designed for display.
    """
    cardinal_direction_props = (
        load_bearing_distribution_pl()
        .filter(pl.col("num_bins").is_in([2, 4]))
        .with_columns(
            bin_count_label=pl.format("{}_{}", "num_bins", "bin_label"),
        )
        .pivot(on="bin_count_label", index="ski_area_id", values="bin_proportion")
        .select(
            "ski_area_id",
            pl.all().exclude("ski_area_id").round(5).name.prefix("bin_proportion_"),
        )
        .select(
            "ski_area_id",
            "bin_proportion_4_N",
            "bin_proportion_4_E",
            "bin_proportion_4_S",
            "bin_proportion_4_W",
            "bin_proportion_2_N",
        )
    )
    table_df = (
        load_ski_areas_pl()
        .filter(*get_display_ski_area_filters(story=story))
        .join(cardinal_direction_props, on="ski_area_id", how="left")
        # inconveniently, order of selection here defines reactable column order
        # https://github.com/glin/reactable/issues/172
        .select(
            "ski_area_id",
            "ski_area_name",
            "osm_status",
            pl.col("country_code")
            .map_elements(country_code_to_emoji, return_dtype=pl.String)
            .alias("country_emoji"),
            "country",
            "country_code",
            "region",
            "locality",
            "latitude",
            "run_count",
            "lift_count",
            "combined_vertical",
            "min_elevation",
            "max_elevation",
            "vertical_drop",
            pl.col("solar_irradiance_solstice").truediv(1000).round(1),  # kWh/m²/day
            "bearing_mean",
            "bearing_alignment",
            "poleward_affinity",
            "eastward_affinity",
            pl.selectors.starts_with("bin_proportion_"),
            pl.col("ski_area_id").alias("rose"),
        )
        # reduce size of HTML output by rounding floats
        .with_columns(pl.selectors.by_dtype(pl.Float64).round(4))
        .sort("ski_area_name")
    )
    if story:
        if missing := set(get_story_ski_area_names()) - set(
            table_df.get_column("ski_area_name")
        ):
            raise ValueError(
                f"Expected ski areas in get_story_ski_area_names are missing: {missing!r}"
            )
    return table_df


# defining cellLatitude in script.js results in a React not found error
_latitude_cell = reactable.JS(
    """
function cellLatitude(cellInfo) {
  const latitude = cellInfo.value;
  const hemisphereSymbol = latitude > 0 ? "ℕ" : "𝕊";

  // Dynamic background color calculation
  const normalizedLatitude = Math.abs(latitude) / 90;
  const backgroundColor = `rgb(
      ${255 - Math.round(normalizedLatitude * 255)},
      ${255 - Math.round(normalizedLatitude * 255)},
      ${255 - Math.round(normalizedLatitude * 255)}
  )`;

  // Construct the cell's HTML
  return React.createElement(
      "span",
      {
          className: "badge",
          style: { "--badge-bg-color": backgroundColor }
      },
      [
          React.createElement(
              "span",
              { className: "hemisphere-symbol" },
              hemisphereSymbol
          ),
          React.createElement(
              "span",
              { className: "latitude-value" },
              `${latitude.toFixed(1)}°`
          )
      ]
  );
}
"""
)

_numeric_filter = reactable.JS("filterNumeric")
_percent_filter = reactable.JS("filterPercent")

_sequential_percent_palette = gradient_n_pal(["#ffffff", "#a100bf"])
_diverging_percent_palette = gradient_n_pal(["#e89200", "#ffffff", "#007dbf"])


def _percent_sequential_style(ci: reactable.CellInfo) -> dict[str, Any] | None:
    """Style cell background for columns whose values range from 0% to 100%."""
    if not isinstance(ci.value, int | float):
        return None
    color = _sequential_percent_palette(ci.value)
    return {"background": color}


def _percent_diverging_style(ci: reactable.CellInfo) -> dict[str, Any] | None:
    """Style cell background for columns whose values range from -100% to 100%."""
    if not isinstance(ci.value, int | float):
        return None
    color = _diverging_percent_palette((ci.value + 1) / 2)
    return {"background": color}


_percent_with_donut_cell = reactable.JS(
    """
function(cellInfo) {
  return donutChart(cellInfo.value)
}
"""
)

columns_descriptions = {
    # pre-populate descriptions from SkiAreaModel
    **{name: field.description for name, field in SkiAreaModel.model_fields.items()},
    # add custom descriptions including overwriting SkiAreaModel descriptions
    "latitude": "Hemisphere where the ski area is located, either ℕ for north or 𝕊 for south, "
    "as well as the latitude (φ) in decimal degrees.",
    "bin_proportion_4_N": "Proportion of vertical-weighted run segments oriented with a northern cardinal direction.",
    "bin_proportion_4_E": "Proportion of vertical-weighted run segments oriented with an eastern cardinal direction.",
    "bin_proportion_4_S": "Proportion of vertical-weighted run segments oriented with a southern cardinal direction.",
    "bin_proportion_4_W": "Proportion of vertical-weighted run segments oriented with a western cardinal direction.",
    "bin_proportion_2_N": "Proportion of vertical-weighted run segments oriented northward with the remaining proportion oriented southward.",
    "rose": "Compass rose histogram of run segment orientations. "
    "Hover over or click on the 8-bin preview rose to view the full 32-bin rose.",
}


def _format_header(ci: reactable.HeaderCellInfo) -> htmltools.Tag | str:
    """
    Format header cell with tooltip provided by an <abbr> tag.
    FIXME: The title attribute is inaccessible to most keyboard, mobile, and screen reader users,
    so creating tooltips like this is generally discouraged.

    References for tooltip headers:
    https://machow.github.io/reactable-py/get-started/format-header-footer.html#headers
    https://glin.github.io/reactable/articles/cookbook/cookbook.html#tooltips
    https://github.com/glin/reactable/issues/220
    """
    column_id = ci.name
    column_name = ci.value
    if description := columns_descriptions.get(column_id):
        return htmltools.tags.abbr(column_name, title=description)
    return column_name


_column_kwargs_location_str = {
    "default_sort_order": "asc",
    "min_width": 90,
    "footer": reactable.JS("footerDistinctCount"),
}
_column_kwargs_meters = {
    "format": reactable.ColFormat(
        suffix=f"{NARROW_SPACE}m",
        digits=0,
        separators=True,
    ),
    "filter_method": _numeric_filter,
    "min_width": 80,
}
_column_kwargs_bin_proportion = {
    "format": reactable.ColFormat(percent=True, digits=0),
    "min_width": 50,
    "filter_method": _percent_filter,
    "style": _percent_sequential_style,
    "footer": reactable.JS("footerMeanWeightedPercent"),
}

theme_light = reactable.Theme(
    style={
        ".border-left": {"border-left": "2px solid #3f3e3e"},
    },
)

# modified from https://machow.github.io/reactable-py/get-started/style-theming.html#global-theme
theme_dark = reactable.Theme(
    color="#dbdce1",
    background_color="#004B59 !important",
    border_color="#33343d !important",
    # highlight color / style is not working for an unknown reason
    highlight_color="#363845 !important",
    row_highlight_style={"background-color": "#3a3b45 !important"},
    input_style={"background-color": "#002B36 !important"},
    page_button_hover_style={"background-color": "#3a3b45"},
    page_button_active_style={"background-color": "#41424e"},
    style={
        ".border-left": {"border-left": "2px solid ##e0dada !important"},
    },
)


def get_ski_area_reactable(story: bool = False) -> reactable.Reactable:
    """
    Reactable of ski areas and their metrics designed for website display.
    If story is True, only display ski areas that are part of the scrollytelling story
    and subset to select columns.
    """
    data_pl = get_ski_area_frontend_table(story=story)

    def _ski_area_cell(ci: reactable.CellInfo) -> htmltools.Tag:
        ski_area_id = data_pl.item(row=ci.row_index, column="ski_area_id")
        url = htmltools.a(
            ci.value,
            href=f"https://openskimap.org/?obj={ski_area_id}",
            target="blank_",
        )
        return url

    def _country_cell(ci: reactable.CellInfo) -> str:
        country_emoji = data_pl.item(row=ci.row_index, column="country_emoji")
        return f"{country_emoji}<br>{ci.value}" if ci.value else ""

    def _ski_area_metric_style(ci: reactable.CellInfo) -> dict[str, Any] | None:
        """
        Style colored text underline for cell in columns whose that express a sequential metric.
        Must apply the `mu` class to the column, shorthand for "metric underline".
        """
        if not isinstance(ci.value, int | float):
            return None
        color = _sequential_percent_palette(
            ci.value / data_pl.get_column(ci.column_name).max()
        )
        return {"--c": color}

    return reactable.Reactable(
        data=data_pl,
        theme=theme_dark if story else theme_light,
        searchable=False,
        highlight=not story,
        full_width=True,
        default_col_def=reactable.Column(
            header=_format_header,
            default_sort_order="desc",
            filterable=not story,
            align="center",
            v_align="center",
            sort_na_last=True,
        ),
        show_page_size_options=True,
        default_sorted={
            "combined_vertical": "desc",
        },
        columns=[
            reactable.Column(
                id="ski_area_id",
                show=False,
            ),
            reactable.Column(
                id="ski_area_name",
                name="Ski Area",
                cell=_ski_area_cell,
                min_width=150,
                max_width=250,
                align="left",
                sticky="left",  # makes entire group sticky
                default_sort_order="asc",
                footer=reactable.JS("footerDistinctCount"),
            ),
            reactable.Column(
                id="osm_status",
                name="Status",
                show=False,
            ),
            reactable.Column(
                id="country",
                name="Country",
                show=not story,
                cell=_country_cell,
                html=True,
                filter_method=reactable.JS("filterCountry"),
                class_="border-left",
                **_column_kwargs_location_str,
            ),
            reactable.Column(
                id="country_emoji",
                show=False,
            ),
            reactable.Column(
                id="country_code",
                show=False,
            ),
            reactable.Column(
                id="region",
                name="Region",
                show=not story,
                **_column_kwargs_location_str,
            ),
            reactable.Column(
                id="locality",
                name="Locality",
                **_column_kwargs_location_str,
            ),
            reactable.Column(
                id="latitude",
                name=f"ℍ{NARROW_SPACE}φ",
                cell=_latitude_cell,
                filter_method=reactable.JS("filterLatitude"),
                min_width=60,
            ),
            reactable.Column(
                id="run_count",
                name="Runs",
                filter_method=_numeric_filter,
                min_width=60,
                style=_ski_area_metric_style,
                class_="border-left mu",
                footer=reactable.JS("footerSum"),
            ),
            reactable.Column(
                id="lift_count",
                name="Lifts",
                filter_method=_numeric_filter,
                min_width=60,
                style=_ski_area_metric_style,
                class_="mu",
                footer=reactable.JS("footerSum"),
            ),
            reactable.Column(
                id="combined_vertical",
                name="Vertical",
                **_column_kwargs_meters,
                style=_ski_area_metric_style,
                class_="mu",
                footer=reactable.JS("footerSumMeters"),
            ),
            reactable.Column(
                id="vertical_drop",
                name="Drop",
                **_column_kwargs_meters,
                style=_ski_area_metric_style,
                class_="mu",
                footer=reactable.JS("footerSumMeters"),
            ),
            reactable.Column(
                id="min_elevation",
                name="Base Elev",
                **_column_kwargs_meters,
                style=_ski_area_metric_style,
                class_="mu",
                footer=reactable.JS("footerMinMeters"),
            ),
            reactable.Column(
                id="max_elevation",
                name="Peak Elev",
                **_column_kwargs_meters,
                style=_ski_area_metric_style,
                class_="mu",
                footer=reactable.JS("footerMaxMeters"),
            ),
            reactable.Column(
                id="solar_irradiance_solstice",
                name="Sunlight",
                show=not story,
                style=_ski_area_metric_style,
                class_="mu",
            ),
            reactable.Column(
                id="bearing_mean",
                name="Azimuth",
                show=not story,
                # format=reactable.ColFormat(suffix="°", digits=0),
                cell=reactable.JS("cellAzimuth"),
                html=True,
                filter_method=_numeric_filter,
                class_="border-left",
                min_width=60,
            ),
            reactable.Column(
                id="bearing_alignment",
                name="Alignment",
                show=not story,
                cell=_percent_with_donut_cell,
                html=True,
                filter_method=_percent_filter,
            ),
            reactable.Column(
                id="poleward_affinity",
                name="Poleward",
                show=not story,
                format=reactable.ColFormat(percent=True, digits=0),
                filter_method=_percent_filter,
                style=_percent_diverging_style,
            ),
            reactable.Column(
                id="eastward_affinity",
                name="Eastward",
                show=not story,
                format=reactable.ColFormat(percent=True, digits=0),
                filter_method=_percent_filter,
                style=_percent_diverging_style,
            ),
            reactable.Column(
                id="bin_proportion_4_N",
                name="N₄",
                show=not story,
                **_column_kwargs_bin_proportion,
                class_="border-left",
            ),
            reactable.Column(
                id="bin_proportion_4_E",
                name="E₄",
                show=not story,
                **_column_kwargs_bin_proportion,
            ),
            reactable.Column(
                id="bin_proportion_4_S",
                name="S₄",
                show=not story,
                **_column_kwargs_bin_proportion,
            ),
            reactable.Column(
                id="bin_proportion_4_W",
                name="W₄",
                show=not story,
                **_column_kwargs_bin_proportion,
            ),
            reactable.Column(
                id="bin_proportion_2_N",
                name="N₂",
                show=not story,
                **_column_kwargs_bin_proportion,
                class_="border-left",
            ),
            reactable.Column(
                id="rose",
                name="Rose",
                html=True,
                sortable=False,
                filterable=False,
                cell=reactable.JS("cellRose"),
                # max_width=45,
                class_="border-left",
            ),
        ],
        column_groups=[
            reactable.ColGroup(
                name="Ski Area", columns=["ski_area_hyper", "osm_status"]
            ),
            reactable.ColGroup(
                name="Location",
                columns=[
                    "latitude",
                    "country_emoji",
                    "country",
                    "region",
                    "locality",
                    "country_subdiv_code",
                ],
            ),
            reactable.ColGroup(
                name="Downhill Runs",
                columns=[
                    "run_count",
                    "lift_count",
                    "combined_vertical",
                    "min_elevation",
                    "max_elevation",
                    "vertical_drop",
                    "solar_irradiance_solstice",
                ],
            ),
            reactable.ColGroup(
                name="Mean Orientation",
                columns=[
                    "bearing_mean",
                    "bearing_alignment",
                    "poleward_affinity",
                    "eastward_affinity",
                ],
            ),
            reactable.ColGroup(
                name="Cardinal Directions",
                columns=[
                    "bin_proportion_4_N",
                    "bin_proportion_4_E",
                    "bin_proportion_4_S",
                    "bin_proportion_4_W",
                ],
            ),
        ],
    )
