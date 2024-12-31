from openskistats.sunlight import SkiSeasonDatetimes

get_open_close = SkiSeasonDatetimes.get_typical_ski_season_dates


def test_get_typical_ski_season_dates__solstice() -> None:
    north_solstice_open, north_solstice_close = get_open_close("north", "solstice")
    assert north_solstice_open == north_solstice_close
    south_solstice_open, south_solstice_close = get_open_close("south", "solstice")
    assert south_solstice_open == south_solstice_close
    assert 179 <= (north_solstice_open - south_solstice_open).days <= 186


def test_get_typical_ski_season_dates__season() -> None:
    north_season_open, north_season_close = get_open_close("north", "season")
    north_season_duration = (north_season_close - north_season_open).days
    south_season_open, south_season_close = get_open_close("south", "season")
    south_season_duration = (south_season_close - south_season_open).days
    assert north_season_duration == south_season_duration
    assert north_season_open.isoformat() == "2024-12-01"
    assert north_season_close.isoformat() == "2025-03-31"
    assert south_season_open.isoformat() == "2024-05-31"
    assert south_season_close.isoformat() == "2024-09-28"
