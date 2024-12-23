import hashlib
import json
import logging
import lzma
import os
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

from openskistats.models import RunCoordinateModel
from openskistats.utils import get_data_directory, get_repo_directory


def get_openskimap_path(
    name: Literal["runs", "ski_areas", "lifts", "info"],
    testing: bool = False,
) -> Path:
    testing = testing or "PYTEST_CURRENT_TEST" in os.environ
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


def download_openskimap_geojson(
    name: Literal["runs", "ski_areas", "lifts"],
) -> OsmDownloadInfo:
    """Download a single geojson file from OpenSkiMap and save it to disk with compression."""
    url = f"https://tiles.openskimap.org/geojson/{name}.geojson"
    path = get_openskimap_path(name)
    path.parent.mkdir(exist_ok=True)
    logging.info(f"Downloading {url} to {path}")
    headers = {
        "From": "https://github.com/dhimmel/openskistats",
    }
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
    download_infos = []
    for name in ["lifts", "ski_areas", "runs"]:
        info = download_openskimap_geojson(name)  # type: ignore [arg-type]
        download_infos.append(dataclass_to_dict(info))
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
    logging.info(
        f"Loaded {len(features):,} {name} with geometry types {geometry_types}"
    )
    return features


def load_openskimap_download_info() -> list[OsmDownloadInfo]:
    download_infos = json.loads(get_openskimap_path("info").read_text())
    return [OsmDownloadInfo(**info_dict) for info_dict in download_infos]


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


def load_runs_from_download_pl() -> pl.DataFrame:
    """
    Load OpenSkiMap runs from their geojson source into a polars DataFrame.
    Filters for runs with a LineString geometry.
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
        row["run_convention"] = run_properties["convention"]
        row["ski_area_ids"] = sorted(
            ski_area["properties"]["id"] for ski_area in run_properties["skiAreas"]
        )
        row["run_sources"] = sorted(
            openskimap_source_to_url(**source) for source in run_properties["sources"]
        )
        coordinates = run["geometry"]["coordinates"]
        row["run_coordinates_raw"] = [
            x.model_dump() for x in _structure_coordinates(coordinates)
        ]
        row["run_coordinates_clean"] = [
            x.model_dump()
            for x in _structure_coordinates(_clean_coordinates(coordinates))
        ]
        rows.append(row)
    return pl.DataFrame(rows, strict=False)


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
        # row["lift_uses"] = lift_properties["uses"]
        row["lift_type"] = lift_properties["liftType"]
        row["lift_status"] = lift_properties["status"]
        row["ski_area_ids"] = sorted(
            ski_area["properties"]["id"] for ski_area in lift_properties["skiAreas"]
        )
        row["lift_sources"] = sorted(
            "{type}:{id}".format(**source) for source in lift_properties["sources"]
        )
        row["lift_geometry_type"] = lift["geometry"]["type"]
        # row["lift_coordinates"] = lift["geometry"]["coordinates"]
        rows.append(row)
    return pl.DataFrame(rows, strict=False)


def load_ski_areas_from_download_pl() -> pl.DataFrame:
    return pl.json_normalize(
        data=[x["properties"] for x in load_openskimap_geojson("ski_areas")],
        separator="__",
        strict=False,
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
        .filter(pl.col("lift_status") == "operating")
        .group_by("ski_area_id")
        .agg(pl.col("lift_id").n_unique().alias("lift_count"))
    )
    return (
        load_ski_areas_from_download_pl()
        .filter(pl.col("type") == "skiArea")
        .filter(pl.col("activities").list.contains("downhill"))
        .select(
            "ski_area_id",
            "ski_area_name",
            pl.col("generated").alias("osm_is_generated"),
            pl.col("runConvention").alias("osm_run_convention"),
            pl.col("status").alias("osm_status"),
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
        )
        .join(lift_metrics, on="ski_area_id", how="left")
    )


def get_ski_area_to_runs(
    runs_pl: pl.DataFrame,
) -> dict[str, list[list[tuple[float, float, float]]]]:
    """
    For each ski area, get a list of runs, where each run is a list of (lon, lat, ele) coordinates.
    """
    # ski area names can be duplicated, like 'Black Mountain', so use the id instead.
    return dict(
        runs_pl.filter(pl.col("run_uses").list.contains("downhill"))
        .explode("ski_area_ids")
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
    coordinates: list[tuple[float, float, float]],
    min_elevation: float = -100.0,
    ensure_downhill: bool = True,
) -> list[tuple[float, float, float]]:
    """
    Sanitize run LineString coordinates to remove floating point errors and ensure downhill runs.
    NOTE: longitude comes before latitude in GeoJSON and osmnx, which is different than GPS coordinates.
    """
    # Round coordinates to undo floating point errors.
    # https://github.com/russellporter/openskimap.org/issues/137
    clean_coords = []
    for lon, lat, ele in coordinates:
        if ele < min_elevation:
            # remove extreme negative elevations
            # https://github.com/russellporter/openskimap.org/issues/141
            continue
        # TODO: consider extreme slope filtering
        clean_coords.append((round(lon, 7), round(lat, 7), round(ele, 2)))
    if not clean_coords:
        return clean_coords
    if ensure_downhill and (clean_coords[0][2] < clean_coords[-1][2]):
        # Ensure the run is going downhill, such that starting elevation > ending elevation
        clean_coords.reverse()
    return clean_coords


test_ski_area_ids = [
    "8896cde00150e73de1f1237320c88767c91ce099",  # Whaleback Mountain
    "dc24f332f3117625dc09479b5d10cbb31a592be4",  # Storrs Hill Ski Area
]


def generate_openskimap_test_data() -> None:
    test_ski_areas = {
        "type": "FeatureCollection",
        "features": [
            x
            for x in load_openskimap_geojson("ski_areas")
            if x["properties"]["id"] in test_ski_area_ids
        ],
    }
    assert len(test_ski_areas["features"]) == len(test_ski_area_ids)

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
