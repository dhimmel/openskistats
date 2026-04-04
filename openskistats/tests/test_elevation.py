"""
Tests for elevation distribution histograms.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from openskistats.analyze import analyze_all_ski_areas_polars, load_ski_areas_pl
from openskistats.elevation import (
    _get_elevation_segments,
    _split_segment_across_bins,
    get_elevation_histogram_data,
    plot_elevation_histogram,
)


@pytest.fixture(scope="module")
def first_ski_area_id() -> str:
    """Run analysis pipeline and return an arbitrary ski area ID from the test dataset."""
    analyze_all_ski_areas_polars()
    return str(load_ski_areas_pl().row(0, named=True)["ski_area_id"])


def test_get_elevation_segments(first_ski_area_id: str) -> None:
    segments = _get_elevation_segments(first_ski_area_id)
    assert not segments.is_empty()
    assert segments["elevation"].null_count() == 0
    assert segments["distance_vertical"].null_count() == 0
    assert segments["distance_3d"].null_count() == 0
    assert (segments["distance_3d"] > 0).all()
    assert segments["run_difficulty_condensed"].null_count() == 0
    assert set(segments.columns) == {
        "run_difficulty_condensed",
        "elevation",
        "distance_vertical",
        "distance_3d",
        "distance_vertical_drop",
    }


def test_get_elevation_histogram_data(first_ski_area_id: str) -> None:
    segments = _get_elevation_segments(first_ski_area_id)
    histogram = get_elevation_histogram_data(segments=segments, bin_width=25.0)
    assert not histogram.is_empty()
    assert histogram["elevation_bin_center"].unique().len() > 1
    assert (histogram["distance_vertical_drop"] > 0).all()
    # Conservation: proportional split must preserve total skiable vertical
    assert (
        abs(
            histogram["distance_vertical_drop"].sum()
            - segments["distance_vertical_drop"].sum()
        )
        < 0.01
    )


def test_get_elevation_histogram_data_distance_3d(first_ski_area_id: str) -> None:
    """Proportional split must also conserve total 3-D run distance."""
    segments = _get_elevation_segments(first_ski_area_id)
    histogram = get_elevation_histogram_data(
        segments=segments, bin_width=25.0, metric="distance_3d"
    )
    assert not histogram.is_empty()
    assert abs(histogram["distance_3d"].sum() - segments["distance_3d"].sum()) < 0.01


def test_get_elevation_histogram_data_narrow_bins(first_ski_area_id: str) -> None:
    """Exercise histogram with a narrower bin width."""
    segments = _get_elevation_segments(first_ski_area_id)
    histogram = get_elevation_histogram_data(segments=segments, bin_width=10.0)
    assert not histogram.is_empty()
    assert (histogram["distance_vertical_drop"] > 0).all()


def test_plot_elevation_histogram(first_ski_area_id: str) -> None:
    fig = plot_elevation_histogram(first_ski_area_id)
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# Unit tests for _split_segment_across_bins
# These test the core proportional-split logic in isolation, independent of
# any real ski area data.
# ---------------------------------------------------------------------------

BW = 25.0  # bin width used throughout unit tests


def _bins(results: list[dict[str, float]]) -> dict[float, float]:
    """Convert split results to {bin_center: value} for easy assertion."""
    return {r["elevation_bin_center"]: r["binned_value"] for r in results}


@pytest.mark.parametrize(
    "elevation, distance_vertical, value, expected_bins",
    [
        pytest.param(
            210.0,
            -10.0,
            100.0,
            {212.5: 100.0},
            id="within_single_bin",
        ),
        pytest.param(
            # spans [200, 230]: bin [200,225)=25, bin [225,250)=5, total=30
            230.0,
            -30.0,
            60.0,
            {212.5: 50.0, 237.5: 10.0},
            id="crosses_one_boundary",
        ),
        pytest.param(
            # spans [200, 275]: three equal 25m bins
            275.0,
            -75.0,
            90.0,
            {212.5: 30.0, 237.5: 30.0, 262.5: 30.0},
            id="spans_three_bins",
        ),
        pytest.param(
            # ascending: elevation=200, dv=+50 => spans [200, 250], two equal bins
            200.0,
            50.0,
            100.0,
            {212.5: 50.0, 237.5: 50.0},
            id="ascending_segment",
        ),
        pytest.param(
            210.0,
            0.0,
            42.0,
            {212.5: 42.0},
            id="flat_segment",
        ),
        pytest.param(
            # upper end sits exactly on boundary 250; must not bleed into [250,275)
            250.0,
            -50.0,
            100.0,
            {212.5: 50.0, 237.5: 50.0},
            id="exact_bin_boundary",
        ),
    ],
)
def test_split_segment_shape(
    elevation: float,
    distance_vertical: float,
    value: float,
    expected_bins: dict[float, float],
) -> None:
    result = _bins(_split_segment_across_bins(elevation, distance_vertical, value, BW))
    assert set(result.keys()) == set(expected_bins.keys())
    for center, expected_val in expected_bins.items():
        assert abs(result[center] - expected_val) < 1e-9


@pytest.mark.parametrize(
    "elevation, distance_vertical, value",
    [
        pytest.param(300.0, -73.0, 55.0, id="crosses_several_bins"),
        pytest.param(100.0, 0.0, 10.0, id="flat"),
        pytest.param(225.0, -25.0, 25.0, id="exactly_one_bin_width"),
        pytest.param(412.5, -87.3, 1.0, id="odd_numbers"),
    ],
)
def test_split_segment_conservation(
    elevation: float, distance_vertical: float, value: float
) -> None:
    """Sum of split values must always equal the original value."""
    result = _split_segment_across_bins(elevation, distance_vertical, value, BW)
    total = sum(r["binned_value"] for r in result)
    assert abs(total - value) < 1e-9
