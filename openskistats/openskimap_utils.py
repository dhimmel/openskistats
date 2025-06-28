import hashlib
import json
import logging
import lzma
import shutil
from collections import Counter
from dataclasses import asdict as dataclass_to_dict
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal

import polars as pl
import requests

from openskistats.models import OpenSkiMapStatus, RunCoordinateModel, SkiRunUsage
from openskistats.utils import (
    get_data_directory,
    get_repo_directory,
    get_request_headers,
    running_in_test,
)
from openskistats.variables import set_variables


def get_openskimap_path(
    name: Literal["runs", "ski_areas", "lifts", "info"],
    testing: bool = False,
) -> Path:
    testing = testing or running_in_test()
    directory = get_data_directory(testing=testing).joinpath("openskimap")
    directory.mkdir(exist_ok=True)
    if name == "info":
        filename = "info.json"
    elif testing:
        filename = f"{name}.geojson"
    else:
        filename = f"{name}.geojson.xz"
    return directory.joinpath(filename)


@dataclass
class OsmDownloadInfo:
    url: str
    relative_path: str
    last_modified: str
    downloaded: str
    content_size_mb: float
    compressed_size_mb: float
    checksum_sha256: str

    def __str__(self) -> str:
        return (
            f"  URL: {self.url}\n"
            f"  Repo-Relative Path: {self.relative_path}\n"
            f"  Last Modified: {self.last_modified}\n"
            f"  Downloaded: {self.downloaded}\n"
            f"  Content Size (MB): {self.content_size_mb:.2f}\n"
            f"  Compressed Size (MB): {self.compressed_size_mb:.2f}\n"
            f"  Checksum (SHA-256): {self.checksum_sha256}"
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def download_openskimap_geojson(
    name: Literal["runs", "ski_areas", "lifts"],
) -> OsmDownloadInfo:
    """Download a single geojson file from OpenSkiMap and save it to disk with compression."""
    url = f"https://tiles.openskimap.org/geojson/{name}.geojson"
    path = get_openskimap_path(name)
    path.parent.mkdir(exist_ok=True)
    logging.info(f"Downloading {url} to {path}")
    headers = get_request_headers()
    response = requests.get(url, allow_redirects=True, headers=headers)
    response.raise_for_status()
    with lzma.open(path, "wb") as write_file:
        write_file.write(response.content)
    info = OsmDownloadInfo(
        url=url,
        relative_path=path.relative_to(get_repo_directory()).as_posix(),
        last_modified=parsedate_to_datetime(
            response.headers["last-modified"]
        ).isoformat(),
        downloaded=parsedate_to_datetime(response.headers["date"]).isoformat(),
        content_size_mb=len(response.content) / 1024**2,
        compressed_size_mb=path.stat().st_size / 1024**2,
        checksum_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    logging.info(f"Download complete:\n{info}")
    return info


def download_openskimap_geojsons() -> None:
    """Download all OpenSkiMap geojson files."""
    download_infos = {}
    for name in ["lifts", "ski_areas", "runs"]:
        info = download_openskimap_geojson(name)  # type: ignore [arg-type]
        download_infos[name] = info.to_dict()
    # write download info to disk
    get_openskimap_path("info").write_text(json.dumps(download_infos, indent=2) + "\n")


def load_openskimap_geojson(
    name: Literal["runs", "ski_areas", "lifts"],
) -> list[dict[str, Any]]:
    path = get_openskimap_path(name)
    logging.info(f"Loading {name} from {path}")
    # polars cannot decompress xz: https://github.com/pola-rs/polars/pull/18536
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path) as read_file:
        data = json.load(read_file)
    assert data["type"] == "FeatureCollection"
    features = data["features"]
    assert isinstance(features, list)
    geometry_types = Counter(feature["geometry"]["type"] for feature in features)
    set_variables(
        **{
            f"openskimap__{name}__counts__01_raw": len(features),
            f"openskimap__{name}__counts__01_raw_by_geometry": dict(geometry_types),
        }
    )
    logging.info(
        f"Loaded {len(features):,} {name} with geometry types {geometry_types}"
    )
    return features


def load_openskimap_download_info() -> dict[str, OsmDownloadInfo]:
    download_infos = json.loads(get_openskimap_path("info").read_text())
    return {
        name: OsmDownloadInfo(**info_dict) for name, info_dict in download_infos.items()
    }


def set_openskimap_download_info_in_variables() -> None:
    set_variables(
        **{
            f"openskimap__{name}__download": info.to_dict()
            for name, info in load_openskimap_download_info().items()
        }
    )


@cache
def load_runs_from_download() -> list[Any]:
    return load_openskimap_geojson("runs")


def _structure_coordinates(
    coordinates: list[tuple[float, float, float]],
) -> list[RunCoordinateModel]:
    return [
        RunCoordinateModel(
            index=i,
            latitude=lat,
            longitude=lon,
            elevation=ele,
        )
        for i, (lon, lat, ele) in enumerate(coordinates)
    ]


def load_downhill_runs_from_download_pl() -> pl.DataFrame:
    """
    Load OpenSkiMap runs from their geojson source into a polars DataFrame.
    Filters for runs with a LineString geometry and for downhill use.
    Rename columns for project nomenclature.
    """
    runs = load_runs_from_download()
    rows = []
    for run in runs:
        if run["geometry"]["type"] != "LineString":
            continue
        row = {}
        run_properties = run["properties"]
        row["run_id"] = run_properties["id"]
        row["run_name"] = run_properties["name"]
        row["run_uses"] = run_properties["uses"]
        row["run_status"] = run_properties["status"]
        row["run_difficulty"] = run_properties["difficulty"]
        row["run_difficulty_convention"] = run_properties["difficultyConvention"]
        row["ski_area_ids"] = sorted(
            ski_area["properties"]["id"] for ski_area in run_properties["skiAreas"]
        )
        row["run_sources"] = sorted(
            openskimap_source_to_url(**source) for source in run_properties["sources"]
        )
        coordinates = run["geometry"]["coordinates"]
        row["run_coordinates_raw_count"] = len(coordinates)
        row["run_coordinates_clean"] = [
            x.model_dump()
            for x in _structure_coordinates(_clean_coordinates(coordinates))
        ]
        row["run_coordinates_filter_count"] = len(coordinates) - len(
            row["run_coordinates_clean"]
        )
        # do we ever need to reverse the elevation profile (e.g. if run coordinates are reversed)?
        if elev_profile := run_properties["elevationProfile"]:
            assert elev_profile["resolution"] == 25
            # It is not a great solution to just filter the missing elevation values,
            # especially without changing the resolution.
            row["run_elevation_profile"] = [
                round(float(x), 2) for x in elev_profile["heights"] if x is not None
            ]
        else:
            row["run_elevation_profile"] = None
        rows.append(row)
    set_variables(
        openskimap__runs__counts__02_linestring=len(rows),
        openskimap__runs__coordinates__counts__01_raw=sum(
            row["run_coordinates_raw_count"] for row in rows
        ),
        openskimap__runs__coordinates__counts__02_clean=sum(
            len(row["run_coordinates_clean"]) for row in rows
        ),
    )
    rows = [row for row in rows if SkiRunUsage.downhill in row["run_uses"]]
    set_variables(
        openskimap__runs__counts__03_downhill=len(rows),
        openskimap__runs__coordinates__counts__03_downhill_raw=sum(
            row["run_coordinates_raw_count"] for row in rows
        ),
        openskimap__runs__coordinates__counts__04_downhill_clean=sum(
            len(row["run_coordinates_clean"]) for row in rows
        ),
    )
    return pl.DataFrame(rows, strict=False, infer_schema_length=None).drop(
        "run_coordinates_raw_count"
    )


def load_lifts_from_download_pl() -> pl.DataFrame:
    """
    Load OpenSkiMap lifts from their geojson source into a polars DataFrame.
    """
    lifts = load_openskimap_geojson("lifts")
    rows = []
    for lift in lifts:
        row = {}
        lift_properties = lift["properties"]
        row["lift_id"] = lift_properties["id"]
        row["lift_name"] = lift_properties["name"]
        row["lift_type"] = lift_properties["liftType"]
        row["lift_status"] = lift_properties["status"]
        # see https://wiki.openstreetmap.org/wiki/Pistes#Ski_lifts
        row["lift_occupancy"] = lift_properties["occupancy"]
        row["lift_capacity"] = lift_properties["capacity"]
        row["lift_duration"] = lift_properties["duration"]
        row["ski_area_ids"] = sorted(
            ski_area["properties"]["id"] for ski_area in lift_properties["skiAreas"]
        )
        row["lift_websites"] = lift_properties["websites"]
        row["lift_sources"] = sorted(
            openskimap_source_to_url(**source) for source in lift_properties["sources"]
        )
        row["lift_geometry_type"] = lift["geometry"]["type"]
        row["lift_coordinates"] = (
            [
                x.model_dump()
                for x in _structure_coordinates(
                    _clean_coordinates(
                        lift["geometry"]["coordinates"], ensure_downhill=False
                    )
                )
            ]
            if row["lift_geometry_type"] == "LineString"
            else None
        )
        rows.append(row)
    return pl.DataFrame(rows, strict=False, infer_schema_length=None)


def load_ski_areas_from_download_pl() -> pl.DataFrame:
    return pl.json_normalize(
        data=[x["properties"] for x in load_openskimap_geojson("ski_areas")],
        separator="__",
        strict=False,
        infer_schema_length=None,
    ).rename(mapping={"id": "ski_area_id", "name": "ski_area_name"})


def openskimap_source_to_url(
    type: Literal["openstreetmap", "skimap.org"],
    id: str | int,
) -> str:
    match type:
        case "openstreetmap":
            return f"https://www.openstreetmap.org/{id}"
        case "skimap.org":
            return f"https://skimap.org/skiareas/view/{id}"
        case _:
            raise ValueError(f"Invalid source {type=} for {id=}")


@cache
def load_downhill_ski_areas_from_download_pl() -> pl.DataFrame:
    lift_metrics = (
        load_lifts_from_download_pl()
        .explode("ski_area_ids")
        .rename({"ski_area_ids": "ski_area_id"})
        .filter(pl.col("ski_area_id").is_not_null())
        .filter(pl.col("lift_status") == OpenSkiMapStatus.operating)
        .group_by("ski_area_id")
        .agg(pl.col("lift_id").n_unique().alias("lift_count"))
    )
    ski_area_df = (
        load_ski_areas_from_download_pl()
        .filter(pl.col("type") == "skiArea")
        .filter(pl.col("activities").list.contains(SkiRunUsage.downhill))
        .select(
            "ski_area_id",
            "ski_area_name",
            pl.col("runConvention").alias("osm_run_convention"),
            pl.col("status").alias("osm_status"),
            pl.col("activities").alias("ski_area_uses"),
            pl.col("location__localized__en__country").alias("country"),
            pl.col("location__localized__en__region").alias("region"),
            pl.col("location__localized__en__locality").alias("locality"),
            pl.col("location__iso3166_1Alpha2").alias("country_code"),
            pl.col("location__iso3166_2").alias("country_subdiv_code"),
            pl.col("websites").alias("ski_area_websites"),
            # sources can have inconsistently typed nested column 'id' as string or int
            # map_elements on struct, see https://github.com/pola-rs/polars/issues/16452#issuecomment-2549487549
            pl.col("sources")
            .list.eval(
                pl.element().map_elements(
                    lambda x: openskimap_source_to_url(
                        type=x["type"],
                        id=x["id"],
                    ),
                    return_dtype=pl.String,
                )
            )
            .alias("ski_area_sources"),
            pl.col("wikidata_id"),
        )
        .join(lift_metrics, on="ski_area_id", how="left")
    )
    set_variables(openskimap__ski_areas__counts__02_downhill=len(ski_area_df))
    return ski_area_df


def get_ski_area_to_runs(
    runs_pl: pl.DataFrame,
) -> dict[str, list[list[tuple[float, float, float]]]]:
    """
    For each ski area, get a list of runs, where each run is a list of (lon, lat, ele) coordinates.
    """
    # ski area names can be duplicated, like 'Black Mountain', so use the id instead.
    return dict(
        runs_pl.explode("ski_area_ids")
        .filter(pl.col("ski_area_ids").is_not_null())
        .select(
            pl.col("ski_area_ids").alias("ski_area_id"),
            pl.col("run_coordinates_clean")
            .list.eval(
                pl.concat_list(
                    pl.element().struct.field("longitude"),
                    pl.element().struct.field("latitude"),
                    pl.element().struct.field("elevation"),
                ).list.to_array(width=3)
            )
            .alias("run_coordinates_tuples"),
        )
        .group_by("ski_area_id")
        .agg(
            pl.col("run_coordinates_tuples").alias("run_coordinates"),
        )
        .iter_rows()
    )


def _clean_coordinates(
    coordinates: list[tuple[float, float, float | None]],
    min_elevation: float = -100.0,
    ensure_downhill: bool = True,
) -> list[tuple[float, float, float]]:
    """
    Sanitize run LineString coordinates to remove floating point errors,
    remove adjacent overlapping (lat, lon) coordinates to prevent problems from zero-length segments,
    and ensure downhill runs if `ensure_downhill` is True.,
    Removes coordinates with missing or bad elevation data.
    NOTE: longitude comes before latitude in GeoJSON and osmnx, which is different than GPS coordinates.
    """
    clean_coords = []
    prior_coord = None
    for coord in coordinates:
        if len(coord) != 3:
            # https://github.com/russellporter/openskimap.org/issues/160
            logging.debug(
                f"Skipping coordinate with unexpected length {len(coord)}, "
                f"expecting length 3 for (lon, lat, ele): {coord}"
            )
            continue
        lon, lat, ele = coord
        if ele is None:
            continue
        if ele < min_elevation:
            # remove extreme negative elevations
            # https://github.com/russellporter/openskimap.org/issues/141
            continue
        # Round coordinates to undo floating point errors.
        # https://github.com/russellporter/openskimap.org/issues/137
        lon_lat_coord = round(lon, 7), round(lat, 7)
        lon_round, lat_round = lon_lat_coord
        if lon_lat_coord == prior_coord:
            logging.info(
                f"Adjacent overlapping coordinates found at <https://www.openstreetmap.org/edit/#map=24/{lat_round}/{lon_round}>."
            )
            continue
        prior_coord = lon_lat_coord
        clean_coords.append((lon_round, lat_round, round(ele, 2)))
    if not clean_coords:
        return clean_coords
    if ensure_downhill and (clean_coords[0][2] < clean_coords[-1][2]):
        # Ensure the run is going downhill, such that starting elevation > ending elevation
        clean_coords.reverse()
    return clean_coords


# ids are fragile, use the name instead as it's more stable
test_ski_area_names = [
    "Whaleback Mountain",
    "Storrs Hill Ski Area",
]


def generate_openskimap_test_data() -> None:
    test_ski_areas = {
        "type": "FeatureCollection",
        "features": [
            x
            for x in load_openskimap_geojson("ski_areas")
            if x["properties"]["name"] in test_ski_area_names
        ],
    }
    assert len(test_ski_areas["features"]) == len(test_ski_area_names)
    test_ski_area_ids: list[str] = [
        ski_area["properties"]["id"]  # type: ignore [index]
        for ski_area in test_ski_areas["features"]
    ]

    def filter_by_ski_areas_property(
        features: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        features_filtered = []
        for feature in features:
            for ski_area in feature["properties"]["skiAreas"]:
                if ski_area["properties"]["id"] in test_ski_area_ids:
                    features_filtered.append(feature)
        return features_filtered

    test_runs = {
        "type": "FeatureCollection",
        "features": filter_by_ski_areas_property(load_runs_from_download()),
    }
    test_lifts = {
        "type": "FeatureCollection",
        "features": filter_by_ski_areas_property(load_openskimap_geojson("lifts")),
    }
    for name, data in [
        ("ski_areas", test_ski_areas),
        ("runs", test_runs),
        ("lifts", test_lifts),
    ]:
        path = get_openskimap_path(name, testing=True)  # type: ignore [arg-type]
        logging.info(f"Writing {len(data['features']):,} {name} to {path}.")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    # copy info.json to testing directory (checksums and sizes will still refer to unfiltered data)
    shutil.copy(
        src=get_openskimap_path("info"), dst=get_openskimap_path("info", testing=True)
    )
