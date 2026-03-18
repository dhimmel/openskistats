"""
Tests for elevation distribution histograms.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openskistats.analyze import analyze_all_ski_areas_polars
from openskistats.elevation import (
    _get_elevation_segments,
    get_elevation_histogram_data,
    plot_elevation_histogram,
)

_WHALEBACK_ID = "28cb013693827a90d778603d8bb6bc0cfd3999c0"
_STORRS_HILL_ID = "dc24f332f3117625dc09479b5d10cbb31a592be4"


def test_get_elevation_segments() -> None:
    analyze_all_ski_areas_polars()
    segments = _get_elevation_segments(_WHALEBACK_ID)
    assert not segments.is_empty()
    assert segments["elevation_midpoint"].null_count() == 0
    assert segments["distance_3d"].null_count() == 0
    assert (segments["distance_3d"] > 0).all()
    assert segments["run_difficulty_condensed"].null_count() == 0
    expected_cols = {
        "run_difficulty_condensed",
        "elevation_midpoint",
        "distance_3d",
    }
    assert expected_cols == set(segments.columns)


def test_get_elevation_histogram_data() -> None:
    analyze_all_ski_areas_polars()
    segments = _get_elevation_segments(_WHALEBACK_ID)
    histogram = get_elevation_histogram_data(segments=segments, bin_width=25.0)
    assert not histogram.is_empty()
    centers = histogram["elevation_bin_center"].unique().sort()
    assert centers.len() > 1
    assert (histogram["distance_3d"] > 0).all()
    total_from_histogram = histogram["distance_3d"].sum()
    total_from_segments = segments["distance_3d"].sum()
    assert total_from_histogram is not None
    assert total_from_segments is not None
    assert abs(total_from_histogram - total_from_segments) < 0.01


def test_plot_elevation_histogram() -> None:
    analyze_all_ski_areas_polars()
    fig = plot_elevation_histogram(_WHALEBACK_ID)
    assert fig is not None
    plt.close(fig)


def test_elevation_histogram_storrs_hill() -> None:
    """Test with Storrs Hill Ski Area, the second test ski area."""
    analyze_all_ski_areas_polars()
    segments = _get_elevation_segments(_STORRS_HILL_ID)
    histogram = get_elevation_histogram_data(segments=segments, bin_width=10.0)
    assert not histogram.is_empty()
    assert (histogram["distance_3d"] > 0).all()
