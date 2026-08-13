import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import TypeAdapter, ValidationError

from openskistats.analyze import load_ski_areas_pl
from openskistats.ski_area_export import (
    SCHEMA_VERSION,
    OpenSkiMapSource,
    SkiAreaSummary,
    SkiAreaSummaryExport,
    create_ski_area_summary_export,
    export_ski_area_summary_json,
    get_ski_area_summary_frame,
)


@pytest.mark.parametrize(
    "datetime_field",
    [
        pytest.param("last_modified", id="source-modification-time"),
        pytest.param("retrieved_at", id="source-retrieval-time"),
    ],
)
def test_openskimap_source_requires_aware_datetimes(datetime_field: str) -> None:
    aware_datetime = datetime(2026, 8, 13, tzinfo=UTC)
    values = {
        "name": "ski_areas",
        "url": "https://tiles.openskimap.org/geojson/ski_areas.geojson",
        "last_modified": aware_datetime,
        "retrieved_at": aware_datetime,
        "checksum_sha256": "0" * 64,
    }
    values[datetime_field] = aware_datetime.replace(tzinfo=None)

    with pytest.raises(ValidationError, match=datetime_field):
        OpenSkiMapSource.model_validate(values)


def test_get_ski_area_summary_frame() -> None:
    source = load_ski_areas_pl()
    summary = get_ski_area_summary_frame(source)

    assert summary.columns == [
        "ski_area_id",
        "ski_area_name",
        "osm_status",
        "osm_run_convention",
        "ski_area_uses",
        "country",
        "country_code",
        "country_subdiv_code",
        "region",
        "locality",
        "latitude",
        "longitude",
        "ski_area_websites",
        "ski_area_sources",
        "wikidata_id",
        "run_count",
        "lift_count",
        "combined_vertical",
        "combined_distance",
        "vertical_drop",
        "min_elevation",
        "max_elevation",
        "solar_irradiation_season",
        "bearing_mean",
        "bearing_alignment",
        "poleward_affinity",
        "eastward_affinity",
        "run_proportion_4_north",
        "run_proportion_4_east",
        "run_proportion_4_south",
        "run_proportion_4_west",
        "run_proportion_2_north",
    ]
    assert summary.height == source.filter(pl.col("ski_area_name").is_not_null()).height
    assert summary.get_column("ski_area_name").null_count() == 0
    assert summary.get_column("bearing_mean").to_list() == [22.0, 7.4]
    assert summary.get_column("run_proportion_4_north").to_list() == [0.7016, 0.8034]


def test_create_ski_area_summary_export() -> None:
    export = create_ski_area_summary_export(load_ski_areas_pl())

    assert export.schema_version == SCHEMA_VERSION
    assert export.record_count == len(export.ski_areas) == 2
    assert export.record_schema["title"] == "SkiAreaSummary"
    assert set(export.record_schema["properties"]) == set(SkiAreaSummary.model_fields)
    assert all(
        "column_info" not in field_schema
        for field_schema in export.record_schema["properties"].values()
    )


def test_export_ski_area_summary_json(tmp_path: Path) -> None:
    path = export_ski_area_summary_json(
        load_ski_areas_pl(), path=tmp_path.joinpath("ski-areas.json")
    )
    document = json.loads(path.read_text())

    TypeAdapter(SkiAreaSummaryExport).validate_python(document)
    assert path.read_bytes().endswith(b"\n")
    fixture_path = Path(__file__).parent.joinpath(
        "data", "webapp", "data", "ski-areas.json"
    )
    assert path.read_bytes() == fixture_path.read_bytes()
