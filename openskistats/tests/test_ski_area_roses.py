from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl
import pytest

import openskistats.ski_area_roses as ski_area_roses


class RoseInputs(NamedTuple):
    info: dict[str, Any]
    ski_areas: pl.DataFrame
    bearings: pl.DataFrame


@pytest.fixture
def rose_inputs() -> RoseInputs:
    info: dict[str, Any] = dict.fromkeys(ski_area_roses.ROSE_INFO_FIELDS, 0)
    info.update(
        ski_area_id="ski-area",
        ski_area_name="Ski Area",
        osm_run_convention="north_america",
    )
    ski_areas = pl.DataFrame([{**info, "bearings": None}])
    bearings = pl.DataFrame(
        [
            {
                "ski_area_id": info["ski_area_id"],
                "num_bins": num_bins,
                "bin_center": 0.0,
                "bin_count": 1.0,
                "bin_count_other": 0.0,
                "bin_count_easy": 0.25,
                "bin_count_intermediate": 0.25,
                "bin_count_advanced": 0.5,
            }
            for num_bins in (32, 8)
        ]
    )
    return RoseInputs(info=info, ski_areas=ski_areas, bearings=bearings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("ski_area_name", "Changed Name", id="metadata"),
        pytest.param("latitude", 1.0, id="solar-band-location"),
    ],
)
def test_rose_fingerprint_changes_with_info(
    rose_inputs: RoseInputs, field: str, value: object
) -> None:
    fingerprint = ski_area_roses._get_ski_area_rose_fingerprint(
        rose_inputs.info, rose_inputs.bearings
    )
    rose_inputs.info[field] = value
    assert (
        ski_area_roses._get_ski_area_rose_fingerprint(
            rose_inputs.info, rose_inputs.bearings
        )
        != fingerprint
    )


def test_rose_fingerprint_changes_with_bearings(rose_inputs: RoseInputs) -> None:
    fingerprint = ski_area_roses._get_ski_area_rose_fingerprint(
        rose_inputs.info, rose_inputs.bearings
    )
    assert (
        ski_area_roses._get_ski_area_rose_fingerprint(
            rose_inputs.info,
            rose_inputs.bearings.with_columns(pl.col("bin_count") + 1),
        )
        != fingerprint
    )


def test_rose_fingerprint_ignores_float_noise(rose_inputs: RoseInputs) -> None:
    info = {**rose_inputs.info, "latitude": 45.0}
    fingerprint = ski_area_roses._get_ski_area_rose_fingerprint(
        info, rose_inputs.bearings
    )
    info["latitude"] += 1e-10
    noisy_bearings = rose_inputs.bearings.with_columns(
        pl.col("bin_count") + 1e-10,
        pl.col("bin_count_other") - 1e-10,
    )
    assert (
        ski_area_roses._get_ski_area_rose_fingerprint(info, noisy_bearings)
        == fingerprint
    )


def test_create_ski_area_roses_reuses_complete_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    rose_inputs: RoseInputs,
    tmp_path: Path,
) -> None:
    rendered_ids: list[str] = []

    def load_ski_areas_pl(ski_area_filters: list[pl.Expr]) -> pl.DataFrame:
        return rose_inputs.ski_areas

    def load_bearing_distribution_pl(
        ski_area_filters: list[pl.Expr],
    ) -> pl.DataFrame:
        return rose_inputs.bearings

    def create_ski_area_rose(
        info: dict[str, Any],
        bearing_pl: pl.DataFrame,
        preview_path: Path,
        full_path: Path,
        openskimap_path: Path,
    ) -> None:
        rendered_ids.append(info["ski_area_id"])
        for path in preview_path, full_path, openskimap_path:
            path.write_text("rose", encoding="utf-8")

    monkeypatch.setattr(ski_area_roses, "get_data_directory", lambda: tmp_path)
    monkeypatch.setattr(ski_area_roses, "load_ski_areas_pl", load_ski_areas_pl)
    monkeypatch.setattr(
        ski_area_roses,
        "load_bearing_distribution_pl",
        load_bearing_distribution_pl,
    )
    monkeypatch.setattr(ski_area_roses, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(ski_area_roses, "_create_ski_area_rose", create_ski_area_rose)

    ski_area_roses.create_ski_area_roses()
    assert rendered_ids == ["ski-area"]

    rose_directory = tmp_path.joinpath("webapp", "ski-areas")
    obsolete_paths = [
        rose_directory.joinpath(variant, "obsolete.svg")
        for variant in ("roses-preview", "roses-full", "roses-openskimap")
    ]
    for path in obsolete_paths:
        path.write_text("obsolete", encoding="utf-8")

    rendered_ids.clear()
    ski_area_roses.create_ski_area_roses()
    assert not rendered_ids
    assert not any(path.exists() for path in obsolete_paths)

    openskimap_path = rose_directory.joinpath("roses-openskimap", "ski-area.svg")
    openskimap_path.unlink()
    ski_area_roses.create_ski_area_roses()
    assert rendered_ids == ["ski-area"]
