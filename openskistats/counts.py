from dataclasses import dataclass

import polars as pl

from openskistats.analyze import load_ski_areas_pl
from openskistats.models import OpenSkiMapStatus


@dataclass
class NamedPolarFilter:
    name: str
    display_name: str
    filter: pl.Expr
    keep_count_sequential: int | None = None
    keep_count_solo: int | None = None
    drop_count_sequential: int | None = None
    drop_count_solo: int | None = None
    keep_ids_solo: list[str] | None = None


ski_area_filters = [
    NamedPolarFilter(
        name="all",
        display_name="No Filters",
        filter=pl.lit(True),
    ),
    # NamedPolarFilter(
    #     "downhill",
    #     display_name="Downhill",
    #     filter=pl.col("run_uses").list.contains(SkiRunUsage.downhill),
    # ),
    NamedPolarFilter(
        name="operating",
        display_name="Operating",
        filter=pl.col("osm_status") == OpenSkiMapStatus.operating,
    ),
    NamedPolarFilter(
        name="named",
        display_name="Named",
        filter=pl.col("ski_area_name").is_not_null(),
    ),
    NamedPolarFilter(
        name="run_count_3",
        display_name="Runs ≥ 3",
        filter=pl.col("run_count") >= 3,
    ),
    NamedPolarFilter(
        name="combined_vertical_50",
        display_name="Combined Vertical ≥ 50 m",
        filter=pl.col("run_count") >= 5,
    ),
    NamedPolarFilter(
        name="lift_count_1",
        display_name="Lifts ≥ 1",
        filter=pl.col("lift_count") >= 1,
    ),
    NamedPolarFilter(
        name="lift_count_5",
        display_name="Lifts ≥ 5",
        filter=pl.col("lift_count") >= 5,
    ),
]


def get_ski_area_counts() -> pl.DataFrame:
    ski_areas = load_ski_areas_pl()
    _ski_areas_filtered = ski_areas
    for filter_ in ski_area_filters:
        filter_no_nulls = filter_.filter.fill_null(False)
        sequential_dropped = _ski_areas_filtered.filter(~filter_no_nulls)
        _ski_areas_filtered = _ski_areas_filtered.filter(filter_no_nulls)
        filter_.keep_count_sequential = _ski_areas_filtered.height
        filter_.keep_count_solo = ski_areas.filter(filter_no_nulls).height
        filter_.drop_count_sequential = sequential_dropped.height
        filter_.drop_count_solo = ski_areas.filter(~filter_no_nulls).height
        filter_.keep_ids_solo = (
            ski_areas.filter(filter_no_nulls).get_column("ski_area_id").to_list()
        )
    return pl.DataFrame(ski_area_filters).drop("filter")


def get_ski_area_comparable_counts() -> dict[str, int]:
    ski_areas = load_ski_areas_pl()
    ski_areas_equipped = ski_areas.filter(
        pl.col("osm_status") == OpenSkiMapStatus.operating
    ).filter(pl.col("lift_count") >= 1)
    return {
        "openskimap__ski_areas__counts__04_downhill_operating_1_lift": ski_areas_equipped.height,
        "openskimap__ski_areas__counts__04_downhill_operating_1_lift_us": ski_areas_equipped.filter(
            pl.col("country_code") == "US"
        ).height,
        "openskimap__ski_areas__counts__05_downhill_operating_5_lift": ski_areas_equipped.filter(
            pl.col("lift_count") >= 5
        ).height,
        "openskimap__countries__counts__ski_areas_04_downhill_operating_1_lift": ski_areas_equipped[
            "country_code"
        ]
        .drop_nulls()
        .n_unique(),
    }
