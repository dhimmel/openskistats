from openskistats.sunlight import SkiSeasonDatetimes


def test_get_typical_ski_season_dates__solstice() -> None:
    north_solstice_open, north_solstice_close = SkiSeasonDatetimes(
        "north", "solstice"
    ).ski_season_dates
    assert north_solstice_open == north_solstice_close
    south_solstice_open, south_solstice_close = SkiSeasonDatetimes(
        "south", "solstice"
    ).ski_season_dates
    assert south_solstice_open == south_solstice_close
    assert 179 <= (north_solstice_open - south_solstice_open).days <= 186


def test_ski_season_dates__season() -> None:
    north_season_open, north_season_close = SkiSeasonDatetimes(
        "north", "season"
    ).ski_season_dates
    north_season_duration = (north_season_close - north_season_open).days
    south_season_open, south_season_close = SkiSeasonDatetimes(
        "south", "season"
    ).ski_season_dates
    south_season_duration = (south_season_close - south_season_open).days
    assert north_season_duration == south_season_duration
    assert north_season_open.isoformat() == "2024-12-01"
    assert north_season_close.isoformat() == "2025-03-31"
    assert south_season_open.isoformat() == "2024-05-31"
    assert south_season_close.isoformat() == "2024-09-28"
