import folium
import polars as pl

from openskistats.utils import get_data_directory

# from folium.plugins.timeline import Timeline, TimelineSlider
# from folium.plugins.timestamped_geo_json import TimestampedGeoJson
# https://python-visualization.github.io/folium/latest/user_guide/plugins/timeline.html


def read_skiway_coords() -> pl.DataFrame:
    path = get_data_directory().joinpath("dartmouth_skiway_solar_irradiance.parquet")
    return pl.read_parquet(path)


def map_the_skiway(time_index: int = 15) -> folium.Map:
    skiway_coords = read_skiway_coords().select(
        "run_id",
        "index",
        "latitude",
        "longitude",
        pl.col("solar_irradiance")
        .list.get(time_index, null_on_oob=True)
        .struct.field("poa_global"),
    )
    _coords = skiway_coords.select("latitude", "longitude")
    min_lat, min_lon = _coords.min().row(index=0)
    max_lat, max_lon = _coords.max().row(index=0)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    map = folium.Map(
        location=(center_lat, center_lon),
        zoom_control=False,
        scrollWheelZoom=False,
        dragging=False,
        max_bounds=True,
        zoomSnap=0.05,
        tiles="CartoDB dark_matter",
    )
    for _df in skiway_coords.partition_by("run_id"):
        folium.ColorLine(
            positions=_df.select(
                pl.concat_list("latitude", "longitude").alias("coords")
            )
            .get_column("coords")
            .to_list(),
            colors=_df.select("poa_global")
            .get_column("poa_global")
            .drop_nulls()
            .to_list(),
            weight=4,
        ).add_to(map)
    map.fit_bounds(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        padding=(10, 10),
    )
    return map
