import logging
from typing import Annotated

import typer

from openskistats.analyze import (
    analyze_all_ski_areas_polars,
    create_ski_area_roses,
    load_ski_areas_pl,
    ski_rose_the_world,
)
from openskistats.models import SkiAreaModel
from openskistats.openskimap_utils import (
    download_openskimap_geojsons,
    generate_openskimap_test_data,
)

cli = typer.Typer(pretty_exceptions_show_locals=False)


class Commands:
    @staticmethod
    @cli.command(name="download")  # type: ignore [misc]
    def download() -> None:
        """Download latest OpenSkiMap source data."""
        download_openskimap_geojsons()

    @staticmethod
    @cli.command(name="analyze")  # type: ignore [misc]
    def analyze(
        skip_runs: Annotated[bool, typer.Option("--skip-runs")] = False,
    ) -> None:
        """Extract ski area metadata and metrics."""
        analyze_all_ski_areas_polars(skip_runs=skip_runs)

    @staticmethod
    @cli.command(name="validate")  # type: ignore [misc]
    def validate() -> None:
        """Validate ski area metadata and metrics."""
        ski_areas = load_ski_areas_pl()
        SkiAreaModel.validate(ski_areas, allow_superfluous_columns=True)
        logging.info("SkiAreaModel.validate success.")

    @staticmethod
    @cli.command(name="visualize")  # type: ignore [misc]
    def visualize(
        overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    ) -> None:
        """Perform ski area aggregations and export visualizations."""
        ski_rose_the_world()
        create_ski_area_roses(overwrite=overwrite)

    @staticmethod
    @cli.command(name="generate_test_data")  # type: ignore [misc]
    def generate_test_data() -> None:
        """
        Generate test data for ski area metadata and metrics.
        """
        generate_openskimap_test_data()

    @staticmethod
    def command() -> None:
        """
        Run like `uv run ski_bearings`
        """
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)
        cli()
