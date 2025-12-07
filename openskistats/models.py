"""
See some OpenSkiMap schemas at
https://github.com/russellporter/openskidata-format
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from patito import Field, Model


class OpenSkiMapStatus(StrEnum):
    """
    Operating status of a run, ski area, or lift according to OpenSkiMap.
    Null values can arise for elements without an OSM source. "
    See OpenSkiMap processing code at
    <https://github.com/russellporter/openskidata-processor/blob/bc168105f4d90c19817189ebe47e7bee20a2dcbc/src/transforms/Status.ts#L3-L8>.
    """

    proposed = "proposed"
    planned = "planned"
    construction = "construction"
    operating = "operating"
    disused = "disused"
    abandoned = "abandoned"
    never_opened = "never_opened"


class RunDifficultyConvention(StrEnum):
    """
    Convention for the ski run difficulty coloring.
    See <https://github.com/russellporter/openskidata-format/blob/e421ef0a0814437c362611e07cb43c32bac4172c/src/Run.ts#L113>.
    """

    north_america = "north_america"
    europe = "europe"
    japan = "japan"

    def display_name(self) -> str:
        return self.value.replace("_", " ").title()

    @classmethod
    def display_names(cls) -> dict[RunDifficultyConvention, str]:
        return {convention: convention.display_name() for convention in cls}


@dataclass
class _SkiRunColors:
    green: str
    blue: str
    red: str
    black: str
    orange: str
    gray: str


# from https://github.com/russellporter/openskidata-format/blob/e421ef0a0814437c362611e07cb43c32bac4172c/src/Run.ts#L87-L92
ski_run_colors_bright = _SkiRunColors(
    green="#00a80e",
    blue="#005aa8",
    red="#f8161a",
    black="#000000",
    orange="#ff9100",
    gray="#595959",
)

ski_run_colors_subtle = _SkiRunColors(
    green="#badbb8",
    blue="#a1a1d8",
    red="#ff8c8e",
    black="#828282",
    orange="#e5c47e",
    gray="#d9d9d9",
)


class SkiRunDifficulty(StrEnum):
    """
    https://wiki.openstreetmap.org/wiki/Key:piste:difficulty
    """

    novice = "novice"
    easy = "easy"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"
    extreme = "extreme"
    freeride = "freeride"
    other = "other"

    @classmethod
    def condense(cls) -> dict[SkiRunDifficulty, SkiRunDifficulty]:
        """Map to a condensed set of difficulty levels."""
        return {
            cls.novice: cls.easy,
            cls.easy: cls.easy,
            cls.intermediate: cls.intermediate,
            cls.advanced: cls.advanced,
            cls.expert: cls.advanced,
            cls.extreme: cls.advanced,
            cls.freeride: cls.advanced,
            cls.other: cls.other,
        }

    @classmethod
    def condensed_values(cls) -> list[SkiRunDifficulty]:
        """Get the condensed set of difficulty levels."""
        return [
            cls.easy,
            cls.intermediate,
            cls.advanced,
            cls.other,
        ]

    @classmethod
    def colormap(
        cls,
        condense: bool = False,
        subtle: bool = True,
        convention: RunDifficultyConvention = RunDifficultyConvention.north_america,
    ) -> dict[SkiRunDifficulty, str]:
        """
        Difficulty to color mapping according to the local convention.
        See:
        <https://github.com/russellporter/openskidata-format/blob/e421ef0a0814437c362611e07cb43c32bac4172c/src/Run.ts#L113-L166>
        <https://www.nsaa.org/NSAA/Safety/Trail_Signage/NSAA/Safety/Trail_Signage.aspx>
        """
        colors = ski_run_colors_subtle if subtle else ski_run_colors_bright
        if convention == RunDifficultyConvention.north_america:
            colormap = {
                cls.novice: colors.green,
                cls.easy: colors.green,
                cls.intermediate: colors.blue,
                cls.advanced: colors.black,
                cls.expert: colors.black,
                cls.extreme: colors.orange,
                cls.freeride: colors.orange,
                cls.other: colors.gray,
            }
        elif convention == RunDifficultyConvention.europe:
            colormap = {
                cls.novice: colors.green,
                cls.easy: colors.blue,
                cls.intermediate: colors.red,
                cls.advanced: colors.black,
                cls.expert: colors.black,
                cls.extreme: colors.orange,
                cls.freeride: colors.orange,
                cls.other: colors.gray,
            }
        elif convention == RunDifficultyConvention.japan:
            colormap = {
                cls.novice: colors.green,
                cls.easy: colors.green,
                cls.intermediate: colors.red,
                cls.advanced: colors.black,
                cls.expert: colors.black,
                cls.extreme: colors.orange,
                cls.freeride: colors.orange,
                cls.other: colors.gray,
            }
        else:
            raise ValueError(f"Unsupported run coloring convention: {convention}")
        if condense:
            colormap = {
                key: value
                for key, value in colormap.items()
                if key in cls.condensed_values()
            }
        return colormap

    def color(
        self,
        subtle: bool = True,
        convention: RunDifficultyConvention = RunDifficultyConvention.north_america,
    ) -> str:
        """Get the color for the difficulty level."""
        return self.colormap(subtle=subtle, convention=convention)[self]

    @classmethod
    def markdown_description(cls, mode: Literal["phrase", "list"] = "phrase") -> str:
        """Get a markdown description of the difficulty levels that describes how values are condensed."""
        from openskistats.utils import oxford_join

        condensed_to_sources: dict[SkiRunDifficulty, list[SkiRunDifficulty]] = {}
        for raw, cond in cls.condense().items():
            condensed_to_sources.setdefault(cond, []).append(raw)
        condensed_strings = []
        for cond, raws in condensed_to_sources.items():
            color = cond.color(subtle=True)
            raw_strings = [raw.value for raw in raws]
            if cond == cls.other:
                raw_strings.append("_missing_")
            condensed_strings.append(
                f"""<span style="background-color: {color}">{cond.value}</span> combines {oxford_join(raw_strings)}"""
            )
        if mode == "list":
            return "\n".join(f"- {s}" for s in condensed_strings)
        if mode == "phrase":
            return oxford_join(condensed_strings, sep="; ")
        raise ValueError(f"Invalid mode: {mode}")


class SkiRunUsage(StrEnum):
    connection = "connection"
    downhill = "downhill"
    hike = "hike"
    nordic = "nordic"
    playground = "playground"
    skitour = "skitour"
    sled = "sled"
    snow_park = "snow_park"


_solar_irradiation_description = (
    "This value measures solar irradiation by treating each run segment as a plane according to its latitude, longitude, elevation, bearing, and slope. "
    "Solar irradiance is computed using clear sky estimates for diffuse normal irradiance, global horizontal irradiance, and direct horizontal irradiance according to the Ineichen and Perez model with Linke turbidity."
)


class RunCoordinateModel(Model):  # type: ignore [misc]
    index: Annotated[
        int,
        Field(description="Zero-indexed order of the coordinate in the run."),
    ]
    latitude: Annotated[
        float,
        Field(description="Latitude of the coordinate in decimal degrees."),
    ]
    longitude: Annotated[
        float,
        Field(description="Longitude of the coordinate in decimal degrees."),
    ]
    elevation: Annotated[
        float,
        Field(description="Elevation of the coordinate in meters."),
    ]


class RunSegmentModel(Model):  # type: ignore [misc]
    segment_hash: Annotated[
        int | None,
        Field(
            description="Hash of the run segment, "
            "where the segment refers to the segment from the preceding coordinate to the current coordinate. "
            "Null for the first coordinate of a run as each run has one fewer segments than coordinates. "
            "The hash uniquely identifies a segment by its coordinates as (latitude, longitude) pairs."
        ),
    ]
    distance_vertical: Annotated[
        float | None,
        Field(
            description="Vertical distance from the previous coordinate to the current coordinate in meters."
        ),
    ]
    distance_vertical_drop: Annotated[
        float | None,
        Field(
            ge=0,
            description="Vertical drop/descent between from the previous coordinate to the current coordinate in meters. "
            "Segments that ascend are set to zero.",
        ),
    ]
    distance_horizontal: Annotated[
        float | None,
        Field(ge=0, description="Horizontal distance of the segment in meters."),
    ]
    distance_3d: Annotated[
        float | None,
        Field(ge=0, description="3D distance of the segment in meters."),
    ]
    bearing: Annotated[
        float | None,
        Field(
            ge=0,
            lt=360,
            description="Bearing of the segment from the previous coordinate to the current coordinate in degrees.",
        ),
    ]
    gradient: Annotated[
        float | None,
        Field(
            description="Gradient of the segment from the previous coordinate to the current coordinate."
        ),
    ]
    slope: Annotated[
        float | None,
        Field(
            description="Slope of the segment from the previous coordinate to the current coordinate in degrees."
        ),
    ]
    solar_irradiation_season: Annotated[
        float | None,
        Field(
            ge=0,
            lt=15,  # practical limit
            description=f"Average daily solar irradiation received by the segment over the course of a typical 115 ski season in kilowatt-hours per square meter (kW/m²/day). {_solar_irradiation_description}",
        ),
    ]
    solar_irradiation_solstice: Annotated[
        float | None,
        Field(
            ge=0,
            lt=15,  # practical limit
            description="Solar irradiation received by the segment on the winter solstice in kilowatt-hours per square meter (kW/m²/day).",
        ),
    ]
    solar_cache_version: Annotated[
        int | None,
        Field(
            description="Version of the solar irradiation calculation and parameters used to compute the solar irradiation values. "
            "This version can be incremented in the source code to invalidate cached solar irradiation values.",
        ),
    ]


class RunCoordinateSegmentModel(RunSegmentModel, RunCoordinateModel):
    # NOTE: fields from RunCoordinateModel are placed before fields from RunSegmentModel
    pass


class RunModel(Model):  # type: ignore [misc]
    run_id: Annotated[
        str,
        Field(
            unique=True,
            description="Unique OpenSkiMap identifier for a run.",
            examples=["d0d6b1c60f677c255eb19dad12c6dd33a141831c"],
        ),
    ]
    run_name: Annotated[
        str | None,
        Field(
            description="Name of the run.",
            examples=["Howard Chivers"],
        ),
    ]
    run_uses: Annotated[
        list[SkiRunUsage] | None,
        Field(description="OpenSkiMap usage types for the run."),
    ]
    run_status: Annotated[
        OpenSkiMapStatus | None,
        Field(description="Operating status of the run according to OpenSkiMap."),
    ]
    run_difficulty: Annotated[
        SkiRunDifficulty | None,
        Field(description="OpenSkiMap difficulty rating for the run."),
    ]
    run_difficulty_convention: Annotated[
        RunDifficultyConvention | None,
        Field(description="OpenSkiMap convention for the run difficulty."),
    ]
    ski_area_ids: Annotated[
        list[str] | None,
        Field(description="Ski areas containing the run."),
    ]
    run_sources: Annotated[
        list[str] | None,
        Field(
            description="List of sources for the run from OpenSkiMap.",
        ),
    ]
    run_coordinates_filter_count: Annotated[
        int,
        Field(
            description="Number of coordinates removed from the run during filtering. "
            "Coordinates with missing or bad elevation data are filtered out.",
        ),
    ]
    run_coordinates_clean: Annotated[
        list[RunCoordinateSegmentModel] | None,
        Field(
            description="Nested coordinates and segment metrics for the run. "
            "Coordinates with extreme negative values are removed as per <https://github.com/russellporter/openskimap.org/issues/141>. "
            "Ascending runs are reversed to ensure all runs are descending.",
        ),
    ]
    run_elevation_profile: Annotated[
        list[float] | None,
        Field(
            description="Elevation in meters of the run at every 25 meter increment of horizontal distance.",
        ),
    ]


class BearingStatsModel(Model):  # type: ignore [misc]
    bearing_mean: Annotated[
        float | None,
        Field(
            description="The mean bearing in degrees.",
            ge=0,
            lt=360,
        ),
    ]
    bearing_alignment: Annotated[
        float | None,
        Field(
            description="Bearing alignment score, representing the concentration / consistency / cohesion of the bearings.",
            ge=0,
            le=1,
        ),
    ]
    bearing_magnitude_net: Annotated[
        float | None,
        Field(
            description="Weighted vector summation of all segments of the ski area. "
            "Used to calculate the mean bearing and mean bearing strength.",
        ),
    ]
    bearing_magnitude_cum: Annotated[
        float | None,
        Field(
            description="Weighted vector summation of all segments of the ski area. "
            "Used to calculate the mean bearing and mean bearing strength.",
        ),
    ]
    poleward_affinity: Annotated[
        float | None,
        Field(
            description="The poleward affinity, representing the tendency of bearings to cluster towards the neatest pole (1.0) or equator (-1.0). "
            "Positive values indicate bearings cluster towards the pole of the hemisphere in which the ski area is located. "
            "Negative values indicate bearings cluster towards the equator.",
            ge=-1,
            le=1,
        ),
    ]
    eastward_affinity: Annotated[
        float | None,
        Field(
            description="The eastern affinity, representing the tendency of bearings to cluster towards the east (1.0) or west (-1.0). "
            "Positive values indicate bearings cluster towards the east. "
            "Negative values indicate bearings cluster towards the west.",
            ge=-1,
            le=1,
        ),
    ]


class SkiAreaBearingDistributionModel(Model):  # type: ignore [misc]
    num_bins: int = Field(
        description="Number of bins in the bearing distribution.",
        # multi-column primary key / uniqueness constraint
        # https://github.com/JakobGM/patito/issues/14
        # constraints=[pl.struct("num_bins", "bin_index").is_unique()],
        ge=1,
    )
    bin_index: int = Field(
        description="Index of the bearing bin starting at 1.",
        ge=1,
    )
    bin_center: float = Field(
        description="Center of the bearing bin in degrees.",
        ge=0,
        lt=360,
    )
    bin_count: float = Field(
        description="Weighted count of bearings in the bin.",
        ge=0,
    )
    bin_proportion: float = Field(
        description="Weighted proportion of bearings in the bin.",
        ge=0,
        le=1,
    )
    bin_count_easy: float = Field(
        description="Weighted count of bearings in the bin for easy difficulty runs.",
        ge=0,
    )
    bin_count_intermediate: float = Field(
        description="Weighted count of bearings in the bin for intermediate difficulty runs.",
        ge=0,
    )
    bin_count_advanced: float = Field(
        description="Weighted count of bearings in the bin for advanced difficulty runs.",
        ge=0,
    )
    bin_count_other: float = Field(
        description="Weighted count of bearings in the bin for other difficulty runs.",
        ge=0,
    )
    bin_label: str | None = Field(
        description="Human readable short label of the bearing bin.",
        examples=["N", "NE", "NEbE", "ENE"],
    )


class SkiAreaBaseModel(Model):  # type: ignore [misc]
    ski_area_id: str = Field(
        unique=True,
        description="Unique OpenSkiMap identifier for a ski area.",
        examples=["fe8efce409aa78cfa20a1e6b5dd5e32369dbe687"],
    )
    ski_area_name: str | None = Field(
        description="Name of the ski area.",
        examples=["Black Mountain"],
    )
    osm_run_convention: Annotated[
        RunDifficultyConvention,
        Field(
            description="OpenSkiMap difficulty convention that determines how runs are colored."
        ),
    ]
    osm_status: OpenSkiMapStatus | None = Field(
        description="Operating status of the ski area according to OpenSkiMap. "
        "Null values arise for ski areas without an OSM source in ski_area_sources. "
        "See OpenSkiMap processing code at <https://github.com/russellporter/openskidata-processor/blob/bc168105f4d90c19817189ebe47e7bee20a2dcbc/src/transforms/Status.ts#L3-L8>."
    )
    ski_area_uses: list[SkiRunUsage] | None = Field(
        description="OpenSkiMap usage types / activities for the ski area.",
        examples=["downhill"],
    )
    country: str | None = Field(
        description="Country where the ski area is located.",
        examples=["United States"],
    )
    region: str | None = Field(
        description="Region/subdivision/province/state where the ski area is located.",
        examples=["New Hampshire"],
    )
    locality: str | None = Field(
        description="Locality/town/city where the ski area is located.",
        examples=["Jackson"],
    )
    country_code: str | None = Field(
        description="ISO 3166-1 alpha-2 two-letter country code.",
        examples=["US", "FR"],
    )
    country_subdiv_code: str | None = Field(
        description="ISO 3166-2 code for principal subdivision (e.g., province or state) of the country.",
        examples=["US-NH", "JP-01", "FR-ARA"],
    )
    ski_area_websites: list[str] | None = Field(
        description="List of URLs for the ski area.",
        examples=["https://www.blackmt.com/"],
    )
    ski_area_sources: Annotated[
        list[str] | None,
        Field(
            description="List of sources for the ski area from OpenSkiMap.",
            examples=[
                "https://www.openstreetmap.org/relation/2873910",
                "https://www.openstreetmap.org/way/387976210",
                "https://skimap.org/skiareas/view/17533",
            ],
        ),
    ]
    wikidata_id: str | None = Field(
        description="Wikidata identifier for the ski area. "
        "See <https://github.com/dhimmel/openskistats/issues/10> and <https://github.com/russellporter/openskimap.org/issues/153>.",
        examples=["Q5225725"],
    )
    run_count: int = Field(
        description="Number of downhill runs in the ski area with supported geometries.",
        default=0,
        ge=0,
    )
    lift_count: int = Field(
        description="Number of operating lifts.",
        default=0,
        ge=0,
    )
    coordinate_count: Annotated[
        int,
        Field(
            description="Number of coordinates in the ski area's runs. "
            "Each coordinate is a point along a run.",
            default=0,
            ge=0,
        ),
    ]
    segment_count: Annotated[
        int,
        Field(
            description="Number of segments in the ski area's runs. "
            "Each segment is a line between two coordinates. "
            "A run with N coordinates has N-1 segments.",
            default=0,
            ge=0,
        ),
    ]
    combined_vertical: float | None = Field(
        description="Total combined vertical drop of the ski area in meters. "
        "If you skied every run in the ski area exactly once, this is the total vertical drop you would accumulate.",
        default=0,
        ge=0,
    )
    combined_distance: Annotated[
        float | None,
        Field(
            description="Total distance of the ski area's runs in meters. "
            "If you skied every run in the ski area exactly once, this is the total distance you would accumulate.",
            default=0,
            ge=0,
        ),
    ]
    vertical_drop: Annotated[
        float | None,
        Field(
            description="Vertical drop of the ski area in meters. "
            "Vertical drop is the difference in elevation between the highest and lowest points of the ski area's runs.",
            default=0,
            ge=0,
        ),
    ]
    latitude: float | None = Field(
        description="Latitude of the ski area in decimal degrees. "
        "Latitude measures the distance north (positive values) or south (negative values) of the equator.",
        ge=-90,
        le=90,
    )
    longitude: float | None = Field(
        description="Longitude of the ski area in decimal degrees. "
        "Longitude measures the distance east (positive values) or west (negative values) of the prime meridian.",
        ge=-180,
        le=180,
    )
    hemisphere: Literal["north", "south"] | None = Field(
        description="Hemisphere of the ski area.",
    )
    min_elevation: Annotated[
        float | None,
        Field(
            description="Base elevation of the ski area in meters computed as the lowest elevation along all runs.",
        ),
    ]
    max_elevation: Annotated[
        float | None,
        Field(
            description="Peak elevation of the ski area in meters computed as the highest elevation along all runs.",
        ),
    ]
    solar_irradiation_season: Annotated[
        float | None,
        Field(
            description="Average daily solar irradiation received by run segments over the course of a typical 120 day ski season in kilowatt-hours per square meter (kW/m²/day). "
            "The average is weighted by the vertical drop of each segment. "
            f"{_solar_irradiation_description}",
        ),
    ]
    solar_irradiation_solstice: Annotated[
        float | None,
        Field(
            description="Average daily solar irradiation received by run segments on the winter solstice in kilowatt-hours per square meter (kW/m²/day). "
            "The average is weighted by the vertical drop of each segment. "
            f"{_solar_irradiation_description}",
        ),
    ]


class SkiAreaModel(BearingStatsModel, SkiAreaBaseModel):
    # Fields start with inherited superclasses (reverse order) and then include the fields defined below.
    # See https://github.com/pydantic/pydantic/issues/1010
    bearings: list[SkiAreaBearingDistributionModel] | None = Field(
        description="Bearing histogram/distribution of the ski area.",
    )
