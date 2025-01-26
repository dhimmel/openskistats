"""
Extract ski area opening and closing dates from New England Ski History Timelines.
For more information on New England Ski History, see

- https://www.newenglandskihistory.com
- https://skinewengland.net/
- https://skinewenglandnet.substack.com/
- https://github.com/dhimmel/openskistats/issues/23
"""

import calendar
import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal

import polars as pl
import requests
from bs4 import BeautifulSoup

from openskistats.utils import get_request_headers


@dataclass(frozen=True)
class NewEnglandSkiHistoryTimelineScraper:
    season: int
    moment: Literal["opening", "closing"]
    NESH_URL: ClassVar[str] = "https://www.newenglandskihistory.com"
    JSON_PATH: ClassVar[Path] = Path(__file__).parent.joinpath(
        "new_england_ski_history_timelines.json"
    )

    @property
    def season_str(self) -> str:
        """Returns a string representation of the season like '2024-25'."""
        return f"{self.season}-{str(self.season + 1)[-2:]}"

    @lru_cache(maxsize=500)  # noqa: B019
    def get_response_text(self) -> str:
        """Get the HTML content of the page."""
        url = f"{self.NESH_URL}/timeline/{self.moment}dates.php"
        time.sleep(1)
        response = requests.get(
            url=url, params={"season": self.season_str}, headers=get_request_headers()
        )
        logging.info(
            f"Request to {response.url} returned status code {response.status_code}."
        )
        response.raise_for_status()
        assert isinstance(response.text, str)
        return response.text

    def extract_ski_area_dates(self) -> list[dict[str, Any]]:
        soup = BeautifulSoup(markup=self.get_response_text(), features="html.parser")
        # find all state tables
        state_tables = soup.find_all("table", {"bgcolor": "#DCDCDC", "width": "100%"})
        opening_days = []
        for table in state_tables:
            header_row, *rows = table.find_all("tr")
            state_name = header_row.find("b").text
            for row in rows:
                ski_area_cell, date_cell = row.find_all("td")
                date_raw = date_cell.get_text(strip=True)
                ski_area_url = None
                if link_elem := ski_area_cell.find("a"):
                    ski_area_url = f"{self.NESH_URL}{link_elem.get('href')}"
                opening_days.append(
                    {
                        "season": self.season,
                        "season_str": self.season_str,
                        "moment": self.moment,
                        "state": state_name,
                        "ski_area_name": ski_area_cell.get_text(strip=True),
                        "ski_area_page": ski_area_url,
                        "date_raw": date_raw,
                        "date_iso": self.parse_raw_date(date_raw),
                    }
                )
        return opening_days

    def parse_raw_date(self, date_raw: str) -> str:
        """
        Parse the raw date string into an ISO 8601 date string.
        """
        month_abbr, day = date_raw.split()
        month_index = list(calendar.month_abbr).index(month_abbr)
        return date(
            year=self.season + self.get_season_year_offset(month_abbr),
            month=month_index,
            day=int(day),
        ).isoformat()

    @staticmethod
    def get_season_year_offset(month_abbr: str) -> int:
        """
        Offset from the starting year of the ski season for the Northern Hemisphere.
        """
        return {
            "Jan": 1,
            "Feb": 1,
            "Mar": 1,
            "Apr": 1,
            "May": 1,
            "Jun": 1,
            "Jul": 1,
            "Aug": 1,
            "Sep": 0,
            "Oct": 0,
            "Nov": 0,
            "Dec": 0,
        }[month_abbr]

    @staticmethod
    def get_all_seasons(starting_year: int = 1936) -> list[int]:
        current_year = date.today().year
        return list(range(starting_year, current_year + 1))

    @classmethod
    def scrape_all_seasons(cls) -> list[dict[str, Any]]:
        """Get all seasons and moments."""
        rows = []
        for season in cls.get_all_seasons():
            for moment in ["opening", "closing"]:
                scraper = cls(season=season, moment=moment)  # type: ignore [arg-type]
                rows.extend(scraper.extract_ski_area_dates())
        rows.sort(key=lambda x: (x["ski_area_name"], x["date_iso"]))
        json_str = json.dumps(rows, indent=2, ensure_ascii=False)
        cls.JSON_PATH.write_text(json_str + "\n")
        return rows


def read_nesh_timelines() -> pl.DataFrame:
    df = (
        pl.read_json(NewEnglandSkiHistoryTimelineScraper.JSON_PATH)
        .pivot(
            index=["ski_area_name", "season", "state", "ski_area_page"],
            on="moment",
            values="date_iso",
            maintain_order=True,
        )
        .with_columns(
            pl.col("opening", "closing").cast(pl.Date).name.suffix("_date"),
        )
        .drop("opening", "closing")
        .with_columns(
            season_duration=pl.col("closing_date").sub("opening_date").dt.total_days()
        )
    )
    return df
