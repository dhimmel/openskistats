import json
from pathlib import Path

import polars as pl
from pydantic import TypeAdapter

from openskistats.analyze import process_lifts
from openskistats.lift_export import (
    LIFT_SUMMARY_FIELDS,
    SCHEMA_VERSION,
    LiftSummary,
    LiftSummaryExport,
    create_lift_summary_export,
    export_lift_summary_json,
    get_lift_summary_frame,
)


def test_get_lift_summary_frame() -> None:
    lifts = process_lifts()
    summary = get_lift_summary_frame(lifts)

    assert summary.columns == list(LIFT_SUMMARY_FIELDS)
    assert (
        summary.height
        == lifts.filter(
            pl.col("lift_name").is_not_null()
            & pl.col("lift_name").str.strip_chars().ne("")
        ).height
    )
    assert summary.get_column("lift_name").null_count() == 0
    sky_lift = summary.filter(pl.col("lift_name") == "Sky Lift").row(0, named=True)
    assert sky_lift["inclined_length"] == 763.3
    assert sky_lift["vertical_rise"] == 200.2
    assert sky_lift["country_code"] == "US"
    assert sky_lift["ski_area_names"] == ["Whaleback Mountain"]


def test_create_lift_summary_export() -> None:
    export = create_lift_summary_export(process_lifts())

    assert export.schema_version == SCHEMA_VERSION
    assert export.record_count == len(export.lifts) == 4
    assert [source.name for source in export.sources] == ["lifts"]
    assert export.record_schema["title"] == "LiftSummary"
    assert set(export.record_schema["properties"]) == set(LiftSummary.model_fields)


def test_export_lift_summary_json(tmp_path: Path) -> None:
    path = export_lift_summary_json(
        process_lifts(), path=tmp_path.joinpath("lifts.json")
    )
    document = json.loads(path.read_text())

    TypeAdapter(LiftSummaryExport).validate_python(document)
    assert path.read_bytes().endswith(b"\n")
    fixture_path = Path(__file__).parent.joinpath(
        "data", "webapp", "data", "lifts.json"
    )
    assert path.read_bytes() == fixture_path.read_bytes()
