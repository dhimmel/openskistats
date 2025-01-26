from datetime import date

import polars as pl

from openskistats.nesh.timelines import read_nesh_timelines_skimap_key


def test_read_nesh_timelines_skimap_key() -> None:
    df = read_nesh_timelines_skimap_key()
    whaleback_2023_timeline = df.row(
        by_predicate=pl.col.skimap_url.eq(
            "https://skimap.org/skiareas/view/1078"
        ).__and__(pl.col.season == 2023),
        named=True,
    )
    assert whaleback_2023_timeline["opening_date"] == date.fromisoformat("2023-12-26")
    assert whaleback_2023_timeline["closing_date"] == date.fromisoformat("2024-03-10")
    assert whaleback_2023_timeline["season_duration"] == 75
    assert whaleback_2023_timeline["nesh_sources"] == [
        "https://www.newenglandskihistory.com/NewHampshire/whaleback.php"
    ]
