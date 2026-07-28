import polars as pl

from openskistats.analyze import load_ski_areas_pl
from openskistats.models import OpenSkiMapStatus


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
