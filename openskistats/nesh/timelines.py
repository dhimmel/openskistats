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
            season_duration=pl.col("closing_date").sub("opening_date").dt.total_days(),
            skimap_url=pl.col("ski_area_page").replace_strict(nesh_to_skimap),
        )
    )
    return df


def read_nesh_timelines_skimap_key() -> pl.DataFrame:
    return (
        read_nesh_timelines()
        .filter(pl.col("skimap_url").is_not_null())
        .group_by("skimap_url", "season")
        .agg(
            pl.min("opening_date"),
            pl.max("closing_date"),
            pl.col("ski_area_page").alias("nesh_sources"),
        )
        .with_columns(
            season_duration=pl.col("closing_date").sub("opening_date").dt.total_days(),
        )
        .sort("skimap_url", "season")
    )


nesh_to_skimap = {
    "https://www.newenglandskihistory.com/NewHampshire/abenaki.php": "https://skimap.org/skiareas/view/4091",
    "https://www.newenglandskihistory.com/NewHampshire/arrowhead.php": "https://skimap.org/skiareas/view/2146",
    "https://www.newenglandskihistory.com/Vermont/ascutney.php": "https://skimap.org/skiareas/view/206",
    "https://www.newenglandskihistory.com/NewHampshire/attitash.php": "https://skimap.org/skiareas/view/349",
    "https://www.newenglandskihistory.com/Maine/bakermtn.php": "https://skimap.org/skiareas/view/3216",
    "https://www.newenglandskihistory.com/Maine/baldmtn.php": None,
    "https://www.newenglandskihistory.com/Maine/baldmtn1.php": "https://skimap.org/skiareas/view/16187",
    "https://www.newenglandskihistory.com/NewHampshire/balsamswilderness.php": "https://skimap.org/skiareas/view/350",
    "https://www.newenglandskihistory.com/Vermont/bearcreek.php": "https://skimap.org/skiareas/view/1090",
    "https://www.newenglandskihistory.com/Massachusetts/beartownmtn.php": "https://skimap.org/skiareas/view/14913",
    "https://www.newenglandskihistory.com/NewHampshire/gunstock.php": "https://skimap.org/skiareas/view/342",
    "https://www.newenglandskihistory.com/Vermont/bellowsfalls.php": "https://skimap.org/skiareas/view/13027",
    "https://www.newenglandskihistory.com/Massachusetts/benjaminhill.php": None,
    "https://www.newenglandskihistory.com/Massachusetts/berkshireeast.php": "https://skimap.org/skiareas/view/435",
    "https://www.newenglandskihistory.com/Massachusetts/berkshiresnowbasin.php": "https://skimap.org/skiareas/view/4942",
    "https://www.newenglandskihistory.com/Maine/mtagamenticus.php": "https://skimap.org/skiareas/view/2149",
    "https://www.newenglandskihistory.com/Maine/squawmtn.php": None,
    "https://www.newenglandskihistory.com/Maine/bigrock.php": "https://skimap.org/skiareas/view/455",
    "https://www.newenglandskihistory.com/NewHampshire/blackmtn.php": "https://skimap.org/skiareas/view/348",
    "https://www.newenglandskihistory.com/Maine/blackmtn.php": "https://skimap.org/skiareas/view/454",
    "https://www.newenglandskihistory.com/Massachusetts/blandford.php": "https://skimap.org/skiareas/view/438",
    "https://www.newenglandskihistory.com/Massachusetts/bluehills.php": "https://skimap.org/skiareas/view/440",
    "https://www.newenglandskihistory.com/NewHampshire/crotchedmtn.php": "https://skimap.org/skiareas/view/343",
    "https://www.newenglandskihistory.com/Vermont/boltonvalley.php": "https://skimap.org/skiareas/view/216",
    "https://www.newenglandskihistory.com/Massachusetts/bostonhills.php": "https://skimap.org/skiareas/view/4698",
    "https://www.newenglandskihistory.com/Massachusetts/bousquet.php": "https://skimap.org/skiareas/view/444",
    "https://www.newenglandskihistory.com/Vermont/livingmemorialpark.php": "https://skimap.org/skiareas/view/4101",
    "https://www.newenglandskihistory.com/NewHampshire/brettonwoods.php": "https://skimap.org/skiareas/view/346",
    "https://www.newenglandskihistory.com/Massachusetts/brodie.php": "https://skimap.org/skiareas/view/2292",
    "https://www.newenglandskihistory.com/Vermont/bromley.php": "https://skimap.org/skiareas/view/217",
    "https://www.newenglandskihistory.com/NewHampshire/brookline.php": "https://skimap.org/skiareas/view/15111",
    "https://www.newenglandskihistory.com/Vermont/suicidesix.php": "https://skimap.org/skiareas/view/203",
    "https://www.newenglandskihistory.com/Vermont/burkemtn.php": "https://skimap.org/skiareas/view/208",
    "https://www.newenglandskihistory.com/Maine/burntmeadowmtn.php": "https://skimap.org/skiareas/view/12818",
    "https://www.newenglandskihistory.com/Vermont/burringtonhill.php": None,
    "https://www.newenglandskihistory.com/Massachusetts/butternut.php": "https://skimap.org/skiareas/view/441",
    "https://www.newenglandskihistory.com/Maine/camdensnowbowl.php": "https://skimap.org/skiareas/view/452",
    "https://www.newenglandskihistory.com/NewHampshire/camptonmtn.php": "https://skimap.org/skiareas/view/3044",
    "https://www.newenglandskihistory.com/NewHampshire/cannonmtn.php": "https://skimap.org/skiareas/view/347",
    "https://www.newenglandskihistory.com/Vermont/carinthia.php": None,  # currently a trail complex at Mount Snow
    "https://www.newenglandskihistory.com/Massachusetts/catamount.php": "https://skimap.org/skiareas/view/323",
    "https://www.newenglandskihistory.com/Massachusetts/chickleyalp.php": "https://skimap.org/skiareas/view/5403",
    "https://www.newenglandskihistory.com/Vermont/cochrans.php": "https://skimap.org/skiareas/view/207",
    "https://www.newenglandskihistory.com/NewHampshire/cograilway.php": None,
    "https://www.newenglandskihistory.com/Maine/quarryroad.php": "https://skimap.org/skiareas/view/4154",
    "https://www.newenglandskihistory.com/NewHampshire/copplecrownmtn.php": "https://skimap.org/skiareas/view/15173",
    "https://www.newenglandskihistory.com/Vermont/cosmichill.php": None,
    "https://www.newenglandskihistory.com/NewHampshire/cranmore.php": "https://skimap.org/skiareas/view/344",
    "https://www.newenglandskihistory.com/NewHampshire/crotchedmtneast.php": None,
    "https://www.newenglandskihistory.com/NewHampshire/dartmouth.php": "https://skimap.org/skiareas/view/345",
    "https://www.newenglandskihistory.com/Vermont/dutchhill.php": "https://skimap.org/skiareas/view/4908",
    "https://www.newenglandskihistory.com/Vermont/northeastslopes.php": "https://skimap.org/skiareas/view/3019",
    "https://www.newenglandskihistory.com/Massachusetts/eastover.php": "https://skimap.org/skiareas/view/4920",
    "https://www.newenglandskihistory.com/Maine/eaton.php": "https://skimap.org/skiareas/view/451",
    "https://www.newenglandskihistory.com/Maine/enchantedmtn.php": "https://skimap.org/skiareas/view/3276",
    "https://www.newenglandskihistory.com/Maine/evergreenvalley.php": "https://skimap.org/skiareas/view/3217",
    "https://www.newenglandskihistory.com/Vermont/farrshill.php": None,
    "https://www.newenglandskihistory.com/NewHampshire/gatewayhills.php": "https://skimap.org/skiareas/view/12239",
    "https://www.newenglandskihistory.com/Vermont/timberridge.php": "https://skimap.org/skiareas/view/2753",
    # Sugarbush North was originally developed as a standalone area known as Glen Ellen
    "https://www.newenglandskihistory.com/Vermont/glenellen.php": None,
    "https://www.newenglandskihistory.com/NewHampshire/granitegorge.php": "https://skimap.org/skiareas/view/1076",
    "https://www.newenglandskihistory.com/Vermont/hardack.php": "https://skimap.org/skiareas/view/12835",
    "https://www.newenglandskihistory.com/Vermont/haystack.php": "https://skimap.org/skiareas/view/2133",
    "https://www.newenglandskihistory.com/Maine/hermonmtn.php": "https://skimap.org/skiareas/view/453",
    # https://www.newenglandskihistory.com/Maine/hermon.php returns a 404, likely a duplicate of hermonmtn
    "https://www.newenglandskihistory.com/Maine/hermon.php": None,
    "https://www.newenglandskihistory.com/Vermont/highpond.php": "https://skimap.org/skiareas/view/2687",
    "https://www.newenglandskihistory.com/NewHampshire/highlands.php": "https://skimap.org/skiareas/view/4788",
    "https://www.newenglandskihistory.com/Vermont/hogback.php": "https://skimap.org/skiareas/view/4623",
    "https://www.newenglandskihistory.com/Vermont/jaypeak.php": "https://skimap.org/skiareas/view/202",
    "https://www.newenglandskihistory.com/Massachusetts/jiminypeak.php": "https://skimap.org/skiareas/view/443",
    "https://www.newenglandskihistory.com/Massachusetts/jugend.php": "https://skimap.org/skiareas/view/12820",
    "https://www.newenglandskihistory.com/NewHampshire/kanc.php": "https://skimap.org/skiareas/view/13097",
    "https://www.newenglandskihistory.com/Vermont/killington.php": "https://skimap.org/skiareas/view/211",
    "https://www.newenglandskihistory.com/NewHampshire/kingpine.php": "https://skimap.org/skiareas/view/354",
    "https://www.newenglandskihistory.com/NewHampshire/kingridge.php": "https://skimap.org/skiareas/view/2656",
    "https://www.newenglandskihistory.com/Massachusetts/atlanticforests.php": "https://skimap.org/skiareas/view/4693",
    "https://www.newenglandskihistory.com/Maine/lonesomepine.php": "https://skimap.org/skiareas/view/3204",
    "https://www.newenglandskihistory.com/NewHampshire/loon.php": "https://skimap.org/skiareas/view/352",
    "https://www.newenglandskihistory.com/Maine/lostvalley.php": "https://skimap.org/skiareas/view/449",
    "https://www.newenglandskihistory.com/Vermont/lyndon.php": "https://skimap.org/skiareas/view/3962",
    "https://www.newenglandskihistory.com/Vermont/madriverglen.php": "https://skimap.org/skiareas/view/200",
    "https://www.newenglandskihistory.com/Vermont/smugglersnotch.php": "https://skimap.org/skiareas/view/209",
    "https://www.newenglandskihistory.com/Vermont/magicmtn.php": "https://skimap.org/skiareas/view/201",
    "https://www.newenglandskihistory.com/Vermont/maplevalley.php": "https://skimap.org/skiareas/view/2660",
    "https://www.newenglandskihistory.com/Maine/maymtn.php": "https://skimap.org/skiareas/view/4497",
    "https://www.newenglandskihistory.com/NewHampshire/mcintyre.php": "https://skimap.org/skiareas/view/3012",
    "https://www.newenglandskihistory.com/Vermont/middlebury.php": "https://skimap.org/skiareas/view/214",
    # sub-peak of Cannon Mountain known as Mt. Jackson or Mittersill Peak
    "https://www.newenglandskihistory.com/NewHampshire/mittersill.php": None,
    "https://www.newenglandskihistory.com/Connecticut/mohawk.php": "https://skimap.org/skiareas/view/493",
    "https://www.newenglandskihistory.com/Massachusetts/mtmohawk.php": "https://skimap.org/skiareas/view/17680",
    "https://www.newenglandskihistory.com/NewHampshire/monteau.php": "https://skimap.org/skiareas/view/4625",
    "https://www.newenglandskihistory.com/Vermont/mtsnow.php": "https://skimap.org/skiareas/view/210",
    "https://www.newenglandskihistory.com/Maine/mtabram.php": "https://skimap.org/skiareas/view/447",
    "https://www.newenglandskihistory.com/Vermont/mtaeolus.php": None,
    "https://www.newenglandskihistory.com/NewHampshire/mteustis.php": "https://skimap.org/skiareas/view/2686",
    "https://www.newenglandskihistory.com/Massachusetts/mtgreylockskiclub.php": "https://skimap.org/skiareas/view/3225",
    "https://www.newenglandskihistory.com/Maine/mtjefferson.php": "https://skimap.org/skiareas/view/450",
    "https://www.newenglandskihistory.com/Vermont/stowe.php": "https://skimap.org/skiareas/view/212",
    "https://www.newenglandskihistory.com/NewHampshire/mtprospect.php": "https://skimap.org/skiareas/view/3224",
    "https://www.newenglandskihistory.com/Connecticut/mtsouthington.php": "https://skimap.org/skiareas/view/494",
    "https://www.newenglandskihistory.com/NewHampshire/sunapee.php": "https://skimap.org/skiareas/view/357",
    "https://www.newenglandskihistory.com/Massachusetts/mttom.php": "https://skimap.org/skiareas/view/2250",
    "https://www.newenglandskihistory.com/Vermont/mttom.php": "https://skimap.org/skiareas/view/4708",
    "https://www.newenglandskihistory.com/NewHampshire/mtwhittier.php": "https://skimap.org/skiareas/view/2747",
    "https://www.newenglandskihistory.com/Massachusetts/nashobavalley.php": "https://skimap.org/skiareas/view/436",
    "https://www.newenglandskihistory.com/Vermont/okemo.php": "https://skimap.org/skiareas/view/204",
    "https://www.newenglandskihistory.com/Massachusetts/osceola.php": "https://skimap.org/skiareas/view/4081",
    "https://www.newenglandskihistory.com/Massachusetts/otisridge.php": "https://skimap.org/skiareas/view/434",
    "https://www.newenglandskihistory.com/NewHampshire/patspeak.php": "https://skimap.org/skiareas/view/355",
    "https://www.newenglandskihistory.com/Vermont/pico.php": "https://skimap.org/skiareas/view/1091",
    "https://www.newenglandskihistory.com/Massachusetts/pineridge.php": "https://skimap.org/skiareas/view/12246",
    "https://www.newenglandskihistory.com/Maine/pinnacle.php": "https://skimap.org/skiareas/view/3275",
    "https://www.newenglandskihistory.com/Maine/shawneepeak.php": "https://skimap.org/skiareas/view/456",
    "https://www.newenglandskihistory.com/Connecticut/powderridge.php": "https://skimap.org/skiareas/view/492",
    "https://www.newenglandskihistory.com/Maine/powderhousehill.php": "https://skimap.org/skiareas/view/3210",
    "https://www.newenglandskihistory.com/Vermont/prospectmtn.php": "https://skimap.org/skiareas/view/5007",
    "https://www.newenglandskihistory.com/Vermont/quechee.php": "https://skimap.org/skiareas/view/215",
    "https://www.newenglandskihistory.com/Maine/quoggyjo.php": "https://skimap.org/skiareas/view/3211",
    "https://www.newenglandskihistory.com/NewHampshire/raggedmtn.php": "https://skimap.org/skiareas/view/351",
    "https://www.newenglandskihistory.com/NewHampshire/redhilloutingclub.php": "https://skimap.org/skiareas/view/4158",
    "https://www.newenglandskihistory.com/Maine/saddleback.php": "https://skimap.org/skiareas/view/446",
    "https://www.newenglandskihistory.com/Connecticut/sundown.php": "https://skimap.org/skiareas/view/496",
    "https://www.newenglandskihistory.com/Massachusetts/bradford.php": "https://skimap.org/skiareas/view/439",
    "https://www.newenglandskihistory.com/RhodeIsland/skivalley.php": "https://skimap.org/skiareas/view/4528",
    "https://www.newenglandskihistory.com/Massachusetts/ward.php": "https://skimap.org/skiareas/view/442",
    "https://www.newenglandskihistory.com/NewHampshire/whaleback.php": "https://skimap.org/skiareas/view/1078",
    "https://www.newenglandskihistory.com/Vermont/snowvalley.php": "https://skimap.org/skiareas/view/2668",
    "https://www.newenglandskihistory.com/NewHampshire/snowsmtn.php": "https://skimap.org/skiareas/view/2655",
    "https://www.newenglandskihistory.com/Vermont/sonnenberg.php": "https://skimap.org/skiareas/view/4722",
    "https://www.newenglandskihistory.com/Maine/sprucemtn.php": "https://skimap.org/skiareas/view/2252",
    "https://www.newenglandskihistory.com/NewHampshire/storrshill.php": "https://skimap.org/skiareas/view/4118",
    "https://www.newenglandskihistory.com/Vermont/strattonmtn.php": "https://skimap.org/skiareas/view/213",
    "https://www.newenglandskihistory.com/Vermont/sugarbush.php": "https://skimap.org/skiareas/view/205",
    "https://www.newenglandskihistory.com/Maine/sugarloaf.php": "https://skimap.org/skiareas/view/448",
    "https://www.newenglandskihistory.com/Maine/sundayriver.php": "https://skimap.org/skiareas/view/459",
    "https://www.newenglandskihistory.com/Connecticut/woodbury.php": "https://skimap.org/skiareas/view/495",
    "https://www.newenglandskihistory.com/NewHampshire/templemtn.php": "https://skimap.org/skiareas/view/2142",
    "https://www.newenglandskihistory.com/NewHampshire/tenneymtn.php": "https://skimap.org/skiareas/view/1077",
    "https://www.newenglandskihistory.com/NewHampshire/thornmtn.php": None,
    "https://www.newenglandskihistory.com/Maine/titcomb.php": "https://skimap.org/skiareas/view/458",
    "https://www.newenglandskihistory.com/NewHampshire/tyrol.php": "https://skimap.org/skiareas/view/4984",
    "https://www.newenglandskihistory.com/NewHampshire/veteransmemorial.php": "https://skimap.org/skiareas/view/4092",
    "https://www.newenglandskihistory.com/Massachusetts/wachusett.php": "https://skimap.org/skiareas/view/437",
    "https://www.newenglandskihistory.com/NewHampshire/watervillevalley.php": "https://skimap.org/skiareas/view/353",
    "https://www.newenglandskihistory.com/NewHampshire/wildcatmtn.php": "https://skimap.org/skiareas/view/356",
    "https://www.newenglandskihistory.com/NewHampshire/woodyglen.php": "https://skimap.org/skiareas/view/16746",
    "https://www.newenglandskihistory.com/RhodeIsland/yawgoovalley.php": "https://skimap.org/skiareas/view/236",
}
"""
Mapping of ski areas from New England Ski History to SkiMap.org.
Mapping established manually by inspecting the respective ski area pages and metadata.
NESH ski areas that only map to a SkiMap ski area that absorbed them are mapped to None rather than the contemporary absorbing ski area.
"""
