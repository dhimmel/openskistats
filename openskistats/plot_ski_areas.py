import plotnine as pn
import polars as pl
from matplotlib.figure import Figure
from mizani.formatters import percent_format

from openskistats.analyze import load_ski_areas_pl
from openskistats.plot import subplot_orientations
from openskistats.utils import gini_coefficient, running_in_test


def get_ski_area_metric_ecdfs(
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    """
    Cumulative distribution metrics for select ski area metrics.
    """
    metrics = [
        "lift_count",
        "run_count",
        "coordinate_count",
        "segment_count",
        "combined_vertical",
        "combined_distance",
        "vertical_drop",
    ]
    return (
        load_ski_areas_pl(ski_area_filters)
        .unpivot(on=metrics, index=["ski_area_id", "ski_area_name"])
        .sort("variable", "value", "ski_area_id")
        .with_columns(
            pl.col("value").rank(method="ordinal").over("variable").alias("value_rank"),
            pl.col("value").cum_sum().over("variable").alias("value_cumsum"),
        )
        .with_columns(
            pl.col("value_cumsum")
            .truediv(pl.sum("value").over("variable"))
            .alias("value_cdf"),
            pl.col("value_rank")
            .truediv(pl.count("value").over("variable"))
            .alias("value_rank_pctl"),
        )
    )


def plot_ski_area_metric_ecdfs(
    ski_area_filters: list[pl.Expr] | None = None,
) -> tuple[pn.ggplot, pn.ggplot]:
    ecdf_df = get_ski_area_metric_ecdfs(ski_area_filters)
    gini_df = (
        ecdf_df.group_by("variable", maintain_order=True)
        .agg(
            gini=pl.col("value").map_batches(gini_coefficient, returns_scalar=True),
        )
        .with_columns(
            variable_label=pl.col("variable").str.replace("_", " ").str.to_titlecase()
        )
        .sort("gini")
    )
    metrics_enum = pl.Enum(gini_df["variable"])
    gini_df = gini_df.with_columns(variable=gini_df["variable"].cast(metrics_enum))
    lorenz_plot = (
        pn.ggplot(
            data=ecdf_df.with_columns(variable=pl.col("variable").cast(metrics_enum)),
            mapping=pn.aes(
                x="value_rank_pctl",
                y="value_cdf",
                color="variable",
            ),
        )
        + pn.geom_abline(intercept=0, slope=1, linetype="dashed")
        + pn.scale_x_continuous(
            name="Ski Area Percentile",
            labels=percent_format(),
            expand=(0.01, 0.01),
        )
        + pn.scale_y_continuous(
            name="Cumulative Share",
            labels=percent_format(),
            expand=(0.01, 0.01),
        )
        + pn.scale_color_discrete()
        + pn.geom_path(show_legend=False)
        + pn.coord_equal()
        + pn.theme_bw()
        + pn.theme(figure_size=(4.2, 4))
    )
    gini_plot = (
        pn.ggplot(
            data=gini_df,
            mapping=pn.aes(
                x="variable", y="gini", fill="variable", label="variable_label"
            ),
        )
        + pn.geom_col(show_legend=False)
        + pn.scale_y_continuous(
            name="Gini Coefficient",
            labels=percent_format(),
            expand=(0, 0),
            limits=(0, 1),
        )
        + pn.geom_text(y=0.05, ha="left")
        + pn.scale_x_discrete(
            name="", limits=list(reversed(metrics_enum.categories)), labels=None
        )
        + pn.coord_flip()
        + pn.theme_bw()
        + pn.theme(figure_size=(3, 4))
    )
    return lorenz_plot, gini_plot


class SkiAreaSubsetPlot:
    @classmethod
    def get_ski_area_names(self) -> list[str]:
        if running_in_test():
            return ["Storrs Hill Ski Area"]
        return [
            "Les Trois Vallées",  # biggest
            "Dartmouth Skiway",  # bimodal
            # "ニセコユナイテッド, Niseko United",  # japan coloring convention
            "Killington Resort",  # eastfacing
            "Mt. Bachelor",  # difficulty by orientation
            "Olos Ski Resort",  # darkest resort in the world
            "Etna Sud/Nicolosi",  # sunniest
            "Jackson Hole",  # southfacing
            "Narvikfjellet Ski Resort",  # northernmost ski resort, https://www.openstreetmap.org/relation/12567328 should be Narvikfjellet
            "Cerro Castor",  # southernmost ski area/resort
        ]

    @classmethod
    def get_ski_areas_df(cls) -> pl.DataFrame:
        ski_areas = (
            pl.Series(name="ski_area_name", values=cls.get_ski_area_names())
            .to_frame()
            .join(
                load_ski_areas_pl(),
                on="ski_area_name",
                how="left",
                maintain_order="left",
            )
        )
        if unmatched_names := ski_areas.filter(pl.col("ski_area_id").is_null())[
            "ski_area_name"
        ].to_list():
            raise ValueError(f"Unmatched ski area names: {unmatched_names}")
        if empty_ski_areas := ski_areas.filter(
            pl.col("combined_vertical").fill_null(0) <= 5
        )["ski_area_name"].to_list():
            # https://github.com/dhimmel/openskistats/issues/64
            raise ValueError(
                f"Ski areas with insufficient combined vertical: {empty_ski_areas}"
            )
        return ski_areas

    @classmethod
    def plot_rose_grid(cls) -> Figure:
        ski_areas = cls.get_ski_areas_df()
        fig = subplot_orientations(
            groups_pl=ski_areas,
            grouping_col="ski_area_name",
            sort_groups=False,
            plot_solar_band=True,
            color_convention=None,
        )
        return fig
