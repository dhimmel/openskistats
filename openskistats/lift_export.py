"""Create the public, self-describing JSON export of named lift summaries."""

import logging
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import polars as pl
from pydantic import AwareDatetime, BeforeValidator, Field, model_validator

from openskistats.models import LiftType, OpenSkiMapStatus
from openskistats.public_data import (
    OpenSkiMapSource,
    PublicDataModel,
    create_openskimap_sources,
)
from openskistats.utils import get_data_directory

SCHEMA_VERSION: Literal["1.0"] = "1.0"


class LiftSummary(PublicDataModel):
    """Curated properties and core metrics for a named lift."""

    lift_id: Annotated[
        str,
        Field(description="Unique OpenSkiMap identifier for the lift."),
    ]
    lift_name: Annotated[str, Field(description="Name of the lift.")]
    lift_type: Annotated[
        LiftType,
        Field(description="Type of lift according to OpenSkiMap."),
    ]
    lift_status: Annotated[
        OpenSkiMapStatus,
        Field(description="Operating status of the lift according to OpenSkiMap."),
    ]
    lift_access: Annotated[
        Literal["private"] | None,
        Field(description="Access restriction for the lift."),
    ]
    lift_oneway: Annotated[
        bool | None,
        Field(
            description="Whether passengers may ride the lift in only one direction."
        ),
    ]
    lift_occupancy: Annotated[
        int | None,
        Field(description="Number of people per carrier.", ge=0),
    ]
    lift_capacity: Annotated[
        float | None,
        Field(description="Transport capacity in people per hour.", ge=0),
    ]
    lift_duration: Annotated[
        int | None,
        Field(description="Typical ride duration in seconds.", ge=0),
    ]
    lift_detachable: Annotated[
        bool | None,
        Field(description="Whether the lift has detachable grips."),
    ]
    lift_bubble: Annotated[
        bool | None,
        Field(description="Whether carriers have weather-protective covers."),
    ]
    lift_heating: Annotated[
        bool | None,
        Field(description="Whether carriers or seats are heated."),
    ]
    ski_area_ids: Annotated[
        list[str],
        Field(description="OpenSkiMap identifiers of associated ski areas."),
    ]
    ski_area_names: Annotated[
        list[str | None],
        Field(
            description="Names of associated ski areas, ordered like `ski_area_ids`."
        ),
    ]
    country: Annotated[
        str | None,
        Field(description="Country where the lift is located."),
    ]
    country_code: Annotated[
        str | None,
        Field(description="ISO 3166-1 alpha-2 country code."),
    ]
    country_subdiv_code: Annotated[
        str | None,
        Field(description="ISO 3166-2 country subdivision code."),
    ]
    region: Annotated[
        str | None,
        Field(description="Region, state, or province where the lift is located."),
    ]
    locality: Annotated[
        str | None,
        Field(description="Locality where the lift is located."),
    ]
    latitude: Annotated[
        float | None,
        Field(description="Mean latitude of lift coordinates in decimal degrees."),
    ]
    longitude: Annotated[
        float | None,
        Field(description="Mean longitude of lift coordinates in decimal degrees."),
    ]
    lift_websites: Annotated[
        list[str],
        Field(description="Websites associated with the lift."),
    ]
    lift_sources: Annotated[
        list[str],
        Field(description="Source URLs for the lift."),
    ]
    wikidata_id: Annotated[
        str | None,
        Field(description="Wikidata identifier for the lift."),
    ]
    inclined_length: Annotated[
        float | None,
        Field(description="Three-dimensional length of the lift in meters.", ge=0),
    ]
    vertical_rise: Annotated[
        float | None,
        Field(
            description="Elevation range from the lowest to highest lift coordinate in meters.",
            ge=0,
        ),
    ]
    min_elevation: Annotated[
        float | None,
        Field(description="Minimum lift elevation in meters."),
    ]
    max_elevation: Annotated[
        float | None,
        Field(description="Maximum lift elevation in meters."),
    ]


LIFT_SUMMARY_FIELDS = tuple(LiftSummary.model_fields)


def _validate_lift_summaries(value: Any) -> list[dict[str, Any]]:
    """Validate and normalize lift records."""
    if not isinstance(value, list):
        raise ValueError("lifts must be a list")
    return [
        LiftSummary.model_validate(record).model_dump(mode="json") for record in value
    ]


class LiftSummaryExport(PublicDataModel):
    """Self-describing document containing named lift summaries."""

    schema_version: Annotated[
        Literal["1.0"],
        Field(description="Major and minor version of the public JSON data contract."),
    ]
    data_updated_at: Annotated[
        AwareDatetime,
        Field(description="Modification time of the OpenSkiMap lift dataset."),
    ]
    license: Annotated[
        Literal["ODbL-1.0"],
        Field(description="SPDX identifier for the source database license."),
    ]
    attribution: Annotated[
        str,
        Field(description="Attribution required when using the public data."),
    ]
    sources: Annotated[
        list[OpenSkiMapSource],
        Field(description="OpenSkiMap datasets used to produce the summaries."),
    ]
    record_count: Annotated[
        int,
        Field(description="Number of objects in `lifts`.", ge=0),
    ]
    record_schema: Annotated[
        dict[str, Any],
        Field(description="JSON Schema for each object in `lifts`."),
    ]
    lifts: Annotated[
        list[dict[str, Any]],
        BeforeValidator(_validate_lift_summaries),
        Field(description="Curated summaries for named lifts."),
    ]

    @model_validator(mode="after")
    def record_count_matches_data(self) -> Self:
        """Require `record_count` to agree with the number of records."""
        if self.record_count != len(self.lifts):
            raise ValueError(
                f"record_count={self.record_count} does not match "
                f"{len(self.lifts)} lift records"
            )
        return self


def get_lift_summary_json_path() -> Path:
    """Return the deployed path of the public lift summary JSON."""
    return get_data_directory().joinpath("webapp", "data", "lifts.json")


def get_lift_summary_frame(lifts: pl.DataFrame) -> pl.DataFrame:
    """Select, round, and order named public lift-summary fields."""
    return (
        lifts.filter(
            pl.col("lift_name").is_not_null()
            & pl.col("lift_name").str.strip_chars().ne("")
        )
        .select(*LIFT_SUMMARY_FIELDS)
        .with_columns(
            pl.col("latitude", "longitude").cast(pl.Float64).round(5),
            pl.col(
                "inclined_length",
                "vertical_rise",
                "min_elevation",
                "max_elevation",
            )
            .cast(pl.Float64)
            .round(1),
        )
        .sort("lift_id")
    )


def create_lift_summary_export(lifts: pl.DataFrame) -> LiftSummaryExport:
    """Build and validate the public lift-summary document."""
    summary = get_lift_summary_frame(lifts)
    sources = create_openskimap_sources(("lifts",))
    return LiftSummaryExport(
        schema_version=SCHEMA_VERSION,
        data_updated_at=sources[0].last_modified,
        license="ODbL-1.0",
        attribution=(
            "OpenSkiStats; source data from OpenSkiMap and OpenStreetMap contributors."
        ),
        sources=sources,
        record_count=summary.height,
        record_schema=LiftSummary.model_json_schema(mode="serialization"),
        lifts=summary.to_dicts(),
    )


def export_lift_summary_json(lifts: pl.DataFrame, path: Path | None = None) -> Path:
    """Write named public lift summaries as indented, UTF-8 JSON."""
    path = path or get_lift_summary_json_path()
    path.parent.mkdir(exist_ok=True, parents=True)
    export = create_lift_summary_export(lifts)
    path.write_text(export.model_dump_json(indent=2) + "\n")
    logging.info(f"Wrote {export.record_count:,} lift summaries to {path}")
    return path
