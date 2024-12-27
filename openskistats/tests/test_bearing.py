import itertools
from dataclasses import dataclass
from typing import Literal

import polars as pl
import pytest

from openskistats.analyze import (
    aggregate_ski_areas_pl,
    analyze_all_ski_areas_polars,
    load_runs_pl,
)
from openskistats.bearing import (
    cut_bearing_breakpoints_pl,
    cut_bearings_pl,
    get_bearing_summary_stats,
)
from openskistats.openskimap_utils import get_ski_area_to_runs
from openskistats.osmnx_utils import create_networkx_with_metadata


def test_cut_bearings_pl__num_bins_4() -> None:
    bearings = [0, 45, 90, 180, 270, 315]
    cuts = (
        pl.DataFrame({"bearing": bearings})
        .with_columns(cut_bearings_pl(num_bins=4))
        .with_columns(*cut_bearing_breakpoints_pl(num_bins=4))
        .to_dict(as_series=False)
    )
    assert cuts["bin_index"] == [1, 2, 2, 3, 4, 1]
    assert cuts["bin_lower"] == [315.0, 45.0, 45.0, 135.0, 225.0, 315.0]
    assert cuts["bin_center"] == [0.0, 90.0, 90.0, 180.0, 270.0, 0.0]
    assert cuts["bin_upper"] == [45.0, 135.0, 135.0, 225.0, 315.0, 45.0]


def test_cut_bearings_pl__num_bins_1() -> None:
    bearings = [0, 180]
    cuts = (
        pl.DataFrame({"bearing": bearings})
        .with_columns(cut_bearings_pl(num_bins=1))
        .with_columns(*cut_bearing_breakpoints_pl(num_bins=1))
        .to_dict(as_series=False)
    )
    assert cuts["bin_index"] == [1, 1]
    assert cuts["bin_lower"] == [180.0, 180.0]
    assert cuts["bin_center"] == [0.0, 0.0]
    assert cuts["bin_upper"] == [180.0, 180.0]


@dataclass
class BearingSummaryStatsPytestParam:
    """
    Dataclass for named pytest parameter definitions.
    https://github.com/pytest-dev/pytest/issues/9216
    """

    bearings: list[float]
    weights: list[float] | None
    combined_vertical: list[float] | None
    hemisphere: Literal["north", "south"] | None
    expected_bearing: float
    expected_strength: float
    expected_poleward_affinity: float | None = None
    excepted_eastward_affinity: float | None = None


@pytest.mark.parametrize(
    "param",
    [
        BearingSummaryStatsPytestParam(
            bearings=[0.0],
            weights=[2.0],
            combined_vertical=[2.0],
            hemisphere="north",
            expected_bearing=0.0,
            expected_strength=1.0,
            expected_poleward_affinity=1.0,
            excepted_eastward_affinity=0.0,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            weights=[1.0, 1.0],
            combined_vertical=None,
            hemisphere="south",
            expected_bearing=45.0,
            expected_strength=0.7071068,
            expected_poleward_affinity=-0.5,
            excepted_eastward_affinity=0.5,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            weights=[0.5, 0.5],
            combined_vertical=[0.5, 0.5],
            hemisphere="north",
            expected_bearing=45.0,
            expected_strength=0.7071068,
            expected_poleward_affinity=0.5,
            excepted_eastward_affinity=0.5,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            weights=None,
            combined_vertical=[2.0, 2.0],
            hemisphere="north",
            expected_bearing=45.0,
            expected_strength=0.3535534,
            expected_poleward_affinity=0.25,
            excepted_eastward_affinity=0.25,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 360.0],
            weights=[1.0, 1.0],
            combined_vertical=None,
            hemisphere="north",
            expected_bearing=0.0,
            expected_strength=1.0,
            expected_poleward_affinity=1.0,
            excepted_eastward_affinity=0.0,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            weights=[0.0, 1.0],
            combined_vertical=[0.5, 1.5],
            hemisphere="north",
            expected_bearing=90.0,
            expected_strength=0.5,
            expected_poleward_affinity=0.0,
            excepted_eastward_affinity=0.5,
        ),
        BearingSummaryStatsPytestParam(
            bearings=[90.0, 270.0],
            weights=None,
            combined_vertical=None,
            hemisphere="north",
            expected_bearing=0.0,
            expected_strength=0.0,
            expected_poleward_affinity=0.0,
            excepted_eastward_affinity=0.0,
        ),  # should cancel each other out
        BearingSummaryStatsPytestParam(
            bearings=[90.0],
            weights=[0.0],
            combined_vertical=None,
            hemisphere="north",
            expected_bearing=0.0,
            expected_strength=0.0,
            expected_poleward_affinity=0.0,
            excepted_eastward_affinity=0.0,
        ),  # strength can only be 0 when weight is 0
        # weights and strengths
        BearingSummaryStatsPytestParam(
            bearings=[0.0, 90.0],
            weights=[2, 4],
            combined_vertical=[10.0, 10.0],
            hemisphere="north",
            expected_bearing=63.4349488,
            expected_strength=0.2236068,
            expected_poleward_affinity=0.1,
            excepted_eastward_affinity=0.2,
        ),
    ],
)
def test_get_bearing_summary_stats(param: BearingSummaryStatsPytestParam) -> None:
    stats = get_bearing_summary_stats(
        bearings=param.bearings,
        net_magnitudes=param.weights,
        cum_magnitudes=param.combined_vertical,
        hemisphere=param.hemisphere,
    )
    assert stats.bearing_mean == pytest.approx(param.expected_bearing)
    assert stats.bearing_alignment == pytest.approx(param.expected_strength)
    assert stats.poleward_affinity == pytest.approx(param.expected_poleward_affinity)
    assert stats.eastward_affinity == pytest.approx(param.excepted_eastward_affinity)


def test_get_bearing_summary_stats_repeated_aggregation() -> None:
    """
    https://github.com/dhimmel/openskistats/issues/1
    """
    analyze_all_ski_areas_polars()
    # aggregate all runs at once
    # we cannot create networkx graph directly from all runs because get_ski_area_to_runs performs some filtering
    ski_area_to_runs = get_ski_area_to_runs(runs_pl=load_runs_pl().collect())
    all_runs_filtered = list(itertools.chain.from_iterable(ski_area_to_runs.values()))
    assert len(all_runs_filtered) > 0
    combined_graph = create_networkx_with_metadata(
        all_runs_filtered, ski_area_metadata={}
    )
    single_pass = combined_graph.graph
    # aggregate runs by ski area and then aggregate ski areas
    # group by hemisphere to avoid polars
    # ComputeError: at least one key is required in a group_by operation
    hemisphere_pl = aggregate_ski_areas_pl(group_by=["hemisphere"])
    double_pass = hemisphere_pl.row(by_predicate=pl.lit(True), named=True)
    for key in [
        "run_count",
        "bearing_mean",
        "bearing_alignment",
        "bearing_magnitude_net",
        "bearing_magnitude_cum",
        "poleward_affinity",
        "eastward_affinity",
    ]:
        assert single_pass[key] == pytest.approx(
            double_pass[key]
        ), f"value mismatch for {key}"
