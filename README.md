# OpenSkiStats: Shredding Data Like Powder

[![GitHub Actions CI Tests Status](https://img.shields.io/github/actions/workflow/status/dhimmel/openskistats/tests.yaml?branch=main&label=actions&style=for-the-badge&logo=github&logoColor=white)](https://github.com/dhimmel/openskistats/actions/workflows/tests.yaml)

> [!IMPORTANT]
> This project is currently under heavy development and is not yet ready for public consumption.
> If you happen to locate the results of the analyses in the meantime,
> please do not disseminate them before contacting the authors.

This project generates statistics on downhill ski slopes and areas from around the globe powered by the underlying OpenSkiMap/OpenStreetMap data.
The first application is the creation of roses showing the compass orientations of ski areas.

## Development

```shell
# download latest OpenSkiMap data
uv run openskistats download

# extract ski area metadata and metrics
uv run openskistats analyze

uv run openskistats visualize
uv run openskistats display

# skirolly dependencies (must install R and renv first)
(cd website/skirolly && quarto add --no-prompt https://github.com/qmd-lab/closeread/archive/e3645070dd668004056ae508d2d25d05baca5ad1.zip)
Rscript -e "setwd('website/skirolly'); renv::restore()"


quarto render website
quarto preview website

# webserver for viewing http://localhost:8000
python -m http.server --directory=data/webapp
```

Commands that you will have to run less frequently:

```shell
# install the uv environment in uv.lock
uv sync --extra=dev

# install the pre-commit git hooks
pre-commit install
```

## References

1. **Urban spatial order: street network orientation, configuration, and entropy**  
Geoff Boeing  
*Applied Network Science* (2019-08-23) <https://doi.org/gf8srn>  
DOI: [10.1007/s41109-019-0189-1](https://doi.org/10.1007/s41109-019-0189-1)

2. **OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks**  
Geoff Boeing  
*Computers, Environment and Urban Systems* (2017-09) <https://doi.org/gbvjxq>  
DOI: [10.1016/j.compenvurbsys.2017.05.004](https://doi.org/10.1016/j.compenvurbsys.2017.05.004)

3. **Climate change exacerbates snow-water-energy challenges for European ski tourism**  
Hugues François, Raphaëlle Samacoïts, David Neil Bird, Judith Köberl, Franz Prettenthaler, Samuel Morin  
*Nature Climate Change* (2023-08-28) <https://doi.org/gsnhbv>  
DOI: [10.1038/s41558-023-01759-5](https://doi.org/10.1038/s41558-023-01759-5)

4. **Vulnerability of ski tourism towards internal climate variability and climate change in the Swiss Alps**  
Fabian Willibald, Sven Kotlarski, Pirmin Philipp Ebner, Mathias Bavay, Christoph Marty, Fabian V Trentini, Ralf Ludwig, Adrienne Grêt-Regamey  
*Science of The Total Environment* (2021-08) <https://doi.org/gvzmqw>  
DOI: [10.1016/j.scitotenv.2021.147054](https://doi.org/10.1016/j.scitotenv.2021.147054) · PMID: [33894612](https://www.ncbi.nlm.nih.gov/pubmed/33894612)

5. **2022 International Report on Snow & Mountain Tourism: Overview of the key industry figures for ski resorts**  
Laurent Vanat  
(2022-04) <https://www.vanat.ch/international-report-on-snow-mountain-tourism>  
ISBN: [9782970102892](https://www.thebookedition.com/fr/2022-international-snow-report-p-389872.html)

6. **SkiVis: Visual Exploration and Route Planning in Ski Resorts**  
Julius Rauscher, Raphael Buchmüller, Daniel A Keim, Matthias Miller  
*IEEE Transactions on Visualization and Computer Graphics* (2023) <https://doi.org/g8qtfb>  
DOI: [10.1109/tvcg.2023.3326940](https://doi.org/10.1109/tvcg.2023.3326940) · PMID: [37874714](https://www.ncbi.nlm.nih.gov/pubmed/37874714)

7. **OSM Ski Resort Routing**
Wenzel Friedsam, Robin Hieber, Alexander Kharitonov, Tobias Rupp  
*Proceedings of the 29th International Conference on Advances in Geographic Information Systems* (2021-11-02) <https://doi.org/g8qtf6>  
DOI: [10.1145/3474717.3483628](https://doi.org/10.1145/3474717.3483628)


List of related webpages:

- https://forums.alpinezone.com/threads/mountain-slopes-facing-side.11630/
- https://avalanche.org/avalanche-encyclopedia/terrain/slope-characteristics/aspect/
- https://www.onxmaps.com/backcountry/app/features/slope-aspect-map
- https://en.wikipedia.org/wiki/Aspect_(geography)
- https://gisgeography.com/aspect-map/
- https://geoffboeing.com/2018/07/comparing-city-street-orientations/
- https://verticalfeet.com/
- https://en.wikipedia.org/wiki/Comparison_of_North_American_ski_resorts
- https://www.nsaa.org/NSAA/Media/Industry_Stats.aspx
- https://www.skitalk.com/threads/comparing-latitude-and-elevation-at-western-us-resorts.9980/
- https://www.stormskiing.com/p/there-are-505-active-ski-areas-in
- https://github.com/pirxpilot/liftie
- https://gitlab.com/hugfr/european-ski-resorts-snow-reliability and https://zenodo.org/records/8047168
- https://mapsynergy.com/ and https://mapsynergy.maps.arcgis.com/apps/dashboards/44d9c8422f3c4cc898642d75392337db

## Wild Ideas

- Table of all OpenStreetMap users that have contributed to ski areas, i.e. top skiers
- Display table webpage background of falling snowflakes ([examples](https://freefrontend.com/css-snow-effects/))
- Max slope v difficulty by region
- fix matplotlib super title spacing
- How many ski areas in the world, comparing to the Vanat report
- Total combined vert of ski areas by rank of ski area (how much do big resorts drive the aggregated metrics)
- fix snowflake alignment

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
