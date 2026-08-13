"""Shared models and helpers for public data exports."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from openskistats.openskimap_utils import load_openskimap_download_info

OpenSkiMapDatasetName = Literal["ski_areas", "runs", "lifts"]


class PublicDataModel(BaseModel):
    """Base configuration for models in the public JSON data contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class OpenSkiMapSource(PublicDataModel):
    """Provenance for one OpenSkiMap input dataset."""

    name: Annotated[
        OpenSkiMapDatasetName,
        Field(description="Name of the OpenSkiMap source dataset."),
    ]
    url: Annotated[str, Field(description="Canonical URL of the source dataset.")]
    last_modified: Annotated[
        AwareDatetime,
        Field(description="Time the source dataset was last modified."),
    ]
    retrieved_at: Annotated[
        AwareDatetime,
        Field(description="Time OpenSkiStats retrieved the source dataset."),
    ]
    checksum_sha256: Annotated[
        str,
        Field(
            description="SHA-256 checksum of the stored compressed source file.",
            pattern=r"^[0-9a-f]{64}$",
        ),
    ]


def create_openskimap_sources(
    names: tuple[OpenSkiMapDatasetName, ...],
) -> list[OpenSkiMapSource]:
    """Create public provenance records for OpenSkiMap datasets."""
    download_info = load_openskimap_download_info()
    sources = []
    for name in names:
        info = download_info[name]
        sources.append(
            OpenSkiMapSource(
                name=name,
                url=info.url,
                last_modified=datetime.fromisoformat(info.last_modified),
                retrieved_at=datetime.fromisoformat(info.downloaded),
                checksum_sha256=info.checksum_sha256,
            )
        )
    return sources
