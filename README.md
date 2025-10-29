# OpenSkiStats: Shredding Data Like Powder

[![GitHub Actions CI Tests Status](https://img.shields.io/github/actions/workflow/status/dhimmel/openskistats/tests.yaml?branch=main&label=actions&style=for-the-badge&logo=github&logoColor=white)](https://github.com/dhimmel/openskistats/actions/workflows/tests.yaml)

> [!IMPORTANT]
> This project is currently under heavy development.
> Methods and results are still preliminary and subject to change.

This project generates statistics on downhill ski slopes and areas from around the globe powered by the underlying OpenSkiMap/OpenStreetMap data.
The first application is the creation of roses showing the compass orientations of ski areas.

## Development

The [`analyze` workflow](.github/workflows/analyze.yaml) performs a complete installation of dependencies on Linux and runs the analysis.
It serves as the reference for installing dependencies and executing the analysis, as well as the sole deployment route.
For convenience, we provide some local development instructions below.

### Installation

Installation and execution requires several system dependencies,
whose installation and presence varies by platform.
The following commands are one method of installation on macOS:

```shell
brew install imagemagick@6

# install the fallback font if you don't have it
# otherwise openskistats visualize will warn:
# WARNING:matplotlib.font_manager:findfont: Font family 'Noto Sans CJK JP' not found.
brew install --cask font-noto-sans-cjk
```

For initial Python setup, first [install uv](https://docs.astral.sh/uv/getting-started/installation/).
Then run the following commands:

```shell
# install the pixi environment in pixi.lock
pixi install

# install the pre-commit git hooks
pre-commit install
```

Install the [quarto](https://quarto.org) extensions:

```shell
# install quarto story extension
(cd website/story && quarto add --no-prompt https://github.com/qmd-lab/closeread/archive/e3645070dd668004056ae508d2d25d05baca5ad1.zip)
```

Check that [R](https://cran.r-project.org/) is installed:

```shell
# Check that R is installed by running:
R --version
```

### Execution

For commands that require access to the python environment,
which includes those beginning with `openskistats` and `quarto`,
you can activate the `pixi` environment any of the following ways:

- configure your IDE to activate the venv automatically, e.g. via "Python: Select Interpreter" [in](https://code.visualstudio.com/docs/python/environments) Visual Studio Code.
- prefix the command with `pixi run`, e.g. `pixi run openskistats --help`
- [activate](https://pixi.sh/dev/workspace/environment/#activation) the environment via `pixi shell`

To execute the Python analysis, run the following commands:

```shell
# download latest OpenSkiMap data
# run infrequently as we want to minimize stress on the OpenSkiMap servers
# downloads persist locally
openskistats download

# extract ski area metadata and metrics
openskistats analyze

openskistats visualize

# run python test suite
pytest

# run the full pre-commit suite
pre-commit run --all
```

To execute the R analysis, run the following command:

```shell
cd r
Rscript 01.data.R
Rscript 02.plot.R
```

To render the website, use either:

```shell
# using quarto preview to render and serve
quarto preview website

# render and serve to <http://localhost:8000> manually 
quarto render website
python -m http.server --directory=data/webapp
```

## References

List of related webpages not yet mentioned in the manuscript:

- https://avalanche.org/avalanche-encyclopedia/terrain/slope-characteristics/aspect/
- https://www.onxmaps.com/backcountry/app/features/slope-aspect-map
- https://en.wikipedia.org/wiki/Aspect_(geography)
- https://gisgeography.com/aspect-map/
- https://www.nsaa.org/NSAA/Media/Industry_Stats.aspx
- https://www.skitalk.com/threads/comparing-latitude-and-elevation-at-western-us-resorts.9980/
- https://gitlab.com/hugfr/european-ski-resorts-snow-reliability and https://zenodo.org/records/8047168
- https://mapsynergy.com/ and https://mapsynergy.maps.arcgis.com/apps/dashboards/44d9c8422f3c4cc898642d75392337db
- https://snowbrains.com/inconsistencies-in-classification-of-ski-slope-difficulty-ratings/
- https://steepseeker.com/
- https://bestsnow.net/
- https://www.zrankings.com/

## Publicity & Users

- [2025-01-04 Closeread Prize Submission: Which way do you ski?](https://forum.posit.co/t/which-way-do-you-ski-closeread-prize/196451)
- [2025-01-12 Reddit r/skiing](https://www.reddit.com/r/skiing/comments/1i03r3c/which_way_do_you_ski_introducing_openskistats/)
- [2025-01-12 Reddit r/dataisbeautiful](https://www.reddit.com/r/dataisbeautiful/comments/1i0fmb2/oc_which_way_do_you_ski_orientations_metrics_for/)
- [2025-01-15 Maps Mania Blog Post](https://googlemapsmania.blogspot.com/2025/01/global-ski-slope-orientations.html)
- [2025-01-24 AnyChart DataViz Weekly](https://www.anychart.com/blog/2025/01/24/compeling-fresh-data-visualizations/)
- [Italian Ski Forums](https://www.skiforum.it/board/index.php?threads/104961/post-2665811)

## Wild Ideas

- Table of all OpenStreetMap users that have contributed to ski areas, i.e. top skiers
- Max slope v difficulty by region
- fix matplotlib super title spacing

## Upstream issue tracking

- [openskimap.org/issues/82](https://github.com/russellporter/openskimap.org/issues/82): Add slope aspect information
- [openskimap.org/issues/135](https://github.com/russellporter/openskimap.org/issues/135): ski_areas.geojson location information is missing
- [openskimap.org/issues/137](https://github.com/russellporter/openskimap.org/issues/137): Restrict coordinate precision to prevent floating-point rounding errors
- [openskimap.org/issues/141](https://github.com/russellporter/openskimap.org/issues/141): Extreme negative elevation values in some run coordinates
- [openskimap.org/issues/143](https://github.com/russellporter/openskimap.org/issues/143) Data downloads block access from GitHub Issues
- [photon/issues/838](https://github.com/komoot/photon/issues/838) and [openskimap.org/issues/139](https://github.com/russellporter/openskimap.org/issues/139): Black Mountain of New Hampshire USA is missing location region metadata
- [osmnx/issues/1137](https://github.com/gboeing/osmnx/issues/1137) and [osmnx/pull/1139](https://github.com/gboeing/osmnx/pull/1139): Support directed bearing/orientation distributions and plots
- [osmnx/issues/1143](https://github.com/gboeing/osmnx/issues/1143) and [osmnx/pull/1147](https://github.com/gboeing/osmnx/pull/1147): _bearings_distribution: defer weighting to np.histogram
- [osmnx/pull/1149](https://github.com/gboeing/osmnx/pull/1149): _bearings_distribution: bin_centers terminology
- [patito/issues/103](https://github.com/JakobGM/patito/issues/103): Validation fails on an empty list
- [patito/issues/104](https://github.com/JakobGM/patito/issues/104): Optional list field with nested model fails to validate
- [polars/issues/19771](https://github.com/pola-rs/polars/issues/19771): A no-op filter errors when the dataframe has an all null column
- [polars/issues/15322](https://github.com/pola-rs/polars/issues/15322#issuecomment-2570076975): skip_nulls does not work in map_elements
- [reactable-py/issues/25](https://github.com/machow/reactable-py/issues/25): Column default sort order does not override global default
- [reactable-py/issues/28](https://github.com/machow/reactable-py/issues/28): Column class_ argument only sets the dev class for the first row
- [reactable-py/issues/29](https://github.com/machow/reactable-py/issues/29): Should great_tables be a dependency (currently dev dependency)
- [reactable-py/issues/38](https://github.com/machow/reactable-py/issues/38): How to call custom javascript after the table is loaded?
- [quarto-cli/issues/11656](https://github.com/quarto-dev/quarto-cli/issues/11656): YAML bibliographies should accept list format, currently requires a dictionary with references
- [pandoc/issues/10452](https://github.com/jgm/pandoc/issues/10452): YAML bibliographies require an object with references and do not accept arrays
- [quarto-cli/discussions/11668](https://github.com/quarto-dev/quarto-cli/discussions/11668): markdown visual editor sentence wrap in figure captions

## License

The code in this repository is released under a [BSD-2-Clause Plus Patent License](LICENSE.md).

This project is built on data from [OpenSkiMap](https://openskimap.org/), which is based on [OpenStreetMap](https://www.openstreetmap.org/).
OpenStreetMap and OpenSkiMap data are released under the [Open Data Commons Open Database License](https://opendatacommons.org/licenses/odbl/).
Learn more at <https://www.openstreetmap.org/copyright>.
