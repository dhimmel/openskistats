"""Create the public, self-describing JSON export of ski-area summaries."""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import polars as pl
from pydantic import (
    AwareDatetime,
    BeforeValidator,
    Field,
    create_model,
    model_validator,
)
from pydantic.fields import FieldInfo

from openskistats.models import SkiAreaModel
from openskistats.public_data import (
    OpenSkiMapSource as OpenSkiMapSource,
)
from openskistats.public_data import (
    PublicDataModel,
    create_openskimap_sources,
)
from openskistats.utils import get_data_directory

SCHEMA_VERSION: Literal["1.0"] = "1.0"


SKI_AREA_SUMMARY_SOURCE_FIELDS = (
    # Identity and classification
    "ski_area_id",
    "ski_area_name",
    "osm_status",
    "osm_run_convention",
    "ski_area_uses",
    # Location
    "country",
    "country_code",
    "country_subdiv_code",
    "region",
    "locality",
    "latitude",
    "longitude",
    # Links and provenance
    "ski_area_websites",
    "ski_area_sources",
    "wikidata_id",
    # Size and elevation
    "run_count",
    "lift_count",
    "combined_vertical",
    "combined_distance",
    "vertical_drop",
    "min_elevation",
    "max_elevation",
    # Sun and orientation
    "solar_irradiation_season",
    "bearing_mean",
    "bearing_alignment",
    "poleward_affinity",
    "eastward_affinity",
)


def _source_field_definition(name: str) -> tuple[Any, FieldInfo]:
    """Copy a field definition from `SkiAreaModel` for the public model."""
    field = deepcopy(SkiAreaModel.model_fields[name])
    # Patito adds dataframe-specific metadata that is not part of this JSON contract.
    field.json_schema_extra = None
    annotation = str if name == "ski_area_name" else field.annotation
    return annotation, field


_ski_area_summary_field_definitions: dict[str, Any] = {
    name: _source_field_definition(name) for name in SKI_AREA_SUMMARY_SOURCE_FIELDS
}
_ski_area_summary_field_definitions.update(
    {
        # Cardinal proportions
        "run_proportion_4_north": (
            float | None,
            Field(
                description="Proportion of vertical-drop-weighted run segments facing north "
                "in a four-cardinal-direction partition.",
                ge=0,
                le=1,
            ),
        ),
        "run_proportion_4_east": (
            float | None,
            Field(
                description="Proportion of vertical-drop-weighted run segments facing east "
                "in a four-cardinal-direction partition.",
                ge=0,
                le=1,
            ),
        ),
        "run_proportion_4_south": (
            float | None,
            Field(
                description="Proportion of vertical-drop-weighted run segments facing south "
                "in a four-cardinal-direction partition.",
                ge=0,
                le=1,
            ),
        ),
        "run_proportion_4_west": (
            float | None,
            Field(
                description="Proportion of vertical-drop-weighted run segments facing west "
                "in a four-cardinal-direction partition.",
                ge=0,
                le=1,
            ),
        ),
        "run_proportion_2_north": (
            float | None,
            Field(
                description="Proportion of vertical-drop-weighted run segments facing north "
                "rather than south in a two-direction partition.",
                ge=0,
                le=1,
            ),
        ),
    }
)

SkiAreaSummary = create_model(
    "SkiAreaSummary",
    __base__=PublicDataModel,
    __doc__="Curated ski-area properties and metrics for public use.",
    **_ski_area_summary_field_definitions,
)


def _validate_ski_area_summaries(value: Any) -> list[dict[str, Any]]:
    """Validate and normalize records with the dynamic Pydantic model."""
    if not isinstance(value, list):
        raise ValueError("ski_areas must be a list")
    return [
        SkiAreaSummary.model_validate(record).model_dump(mode="json")
        for record in value
    ]


class SkiAreaSummaryExport(PublicDataModel):
    """Self-describing document containing the public ski-area summary dataset."""

    schema_version: Annotated[
        Literal["1.0"],
        Field(description="Major and minor version of the public JSON data contract."),
    ]
    data_updated_at: Annotated[
        AwareDatetime,
        Field(
            description="Most recent source modification time among the input datasets."
        ),
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
        Field(description="OpenSkiMap datasets used to produce the summary records."),
    ]
    record_count: Annotated[
        int,
        Field(description="Number of objects in `ski_areas`.", ge=0),
    ]
    record_schema: Annotated[
        dict[str, Any],
        Field(description="JSON Schema for each object in `ski_areas`."),
    ]
    ski_areas: Annotated[
        list[dict[str, Any]],
        BeforeValidator(_validate_ski_area_summaries),
        Field(description="Curated ski-area summary records."),
    ]

    @model_validator(mode="after")
    def record_count_matches_data(self) -> Self:
        """Require `record_count` to agree with the number of records."""
        if self.record_count != len(self.ski_areas):
            raise ValueError(
                f"record_count={self.record_count} does not match "
                f"{len(self.ski_areas)} ski-area records"
            )
        return self


def get_ski_area_summary_json_path() -> Path:
    """Return the deployed path of the public ski-area summary JSON."""
    return get_data_directory().joinpath("webapp", "data", "ski-areas.json")


def get_ski_area_summary_frame(ski_areas: pl.DataFrame) -> pl.DataFrame:
    """Select, derive, round, and order public ski-area summary fields."""
    bearing_proportions = (
        ski_areas.select("ski_area_id", "bearings")
        .explode("bearings", empty_as_null=False, keep_nulls=False)
        .filter(pl.col("bearings").is_not_null())
        .unnest("bearings")
        .filter(pl.col("num_bins").is_in([2, 4]))
        .with_columns(
            proportion_name=pl.format(
                "run_proportion_{}_{}",
                "num_bins",
                pl.col("bin_label").replace_strict(
                    {"N": "north", "E": "east", "S": "south", "W": "west"}
                ),
            )
        )
        .pivot(on="proportion_name", index="ski_area_id", values="bin_proportion")
    )
    summary = (
        ski_areas.filter(pl.col("ski_area_name").is_not_null())
        .join(bearing_proportions, on="ski_area_id", how="left")
        .select(*SkiAreaSummary.model_fields)
        .with_columns(
            pl.col("latitude", "longitude").cast(pl.Float64).round(5),
            pl.col(
                "combined_vertical",
                "combined_distance",
                "vertical_drop",
                "min_elevation",
                "max_elevation",
            )
            .cast(pl.Float64)
            .round(1),
            pl.col("bearing_mean").cast(pl.Float64).round(1).mod(360),
            pl.col("solar_irradiation_season").cast(pl.Float64).round(3),
            pl.col(
                "bearing_alignment",
                "poleward_affinity",
                "eastward_affinity",
                "run_proportion_4_north",
                "run_proportion_4_east",
                "run_proportion_4_south",
                "run_proportion_4_west",
                "run_proportion_2_north",
            )
            .cast(pl.Float64)
            .round(4),
        )
        .sort("ski_area_id")
    )
    return summary


def create_ski_area_summary_export(
    ski_areas: pl.DataFrame,
) -> SkiAreaSummaryExport:
    """Build and validate the self-describing public data document."""
    summary = get_ski_area_summary_frame(ski_areas)
    sources = create_openskimap_sources(("ski_areas", "runs", "lifts"))
    return SkiAreaSummaryExport(
        schema_version=SCHEMA_VERSION,
        data_updated_at=max(source.last_modified for source in sources),
        license="ODbL-1.0",
        attribution=(
            "OpenSkiStats; source data from OpenSkiMap and OpenStreetMap contributors."
        ),
        sources=sources,
        record_count=summary.height,
        record_schema=SkiAreaSummary.model_json_schema(mode="serialization"),
        ski_areas=summary.to_dicts(),
    )


def export_ski_area_summary_json(
    ski_areas: pl.DataFrame, path: Path | None = None
) -> Path:
    """Write the public ski-area summary as indented, UTF-8 JSON."""
    path = path or get_ski_area_summary_json_path()
    path.parent.mkdir(exist_ok=True, parents=True)
    export = create_ski_area_summary_export(ski_areas)
    path.write_text(export.model_dump_json(indent=2) + "\n")
    logging.info(f"Wrote {export.record_count:,} ski-area summaries to {path}")
    return path
