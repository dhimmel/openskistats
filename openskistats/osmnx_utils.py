import contextlib
import itertools
import statistics
import warnings
from collections.abc import Generator
from typing import Any

import networkx as nx
import osmnx
from osmnx.bearing import add_edge_bearings
from osmnx.distance import add_edge_lengths

from openskistats.bearing import get_bearing_summary_stats


@contextlib.contextmanager
def suppress_user_warning(
    category: type[Warning] = UserWarning, message: str = ""
) -> Generator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=message, category=category)
        yield


def create_networkx(
    runs: list[list[tuple[float, float, float]]],
) -> nx.MultiDiGraph:
    """
    Convert run coordinates to a newtorkx MultiDiGraph compatible with OSMnx.
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")  # https://epsg.io/4326
    graph.graph["run_count"] = len(runs)
    for lon, lat, elevation in itertools.chain.from_iterable(runs):
        graph.add_node((lon, lat), x=lon, y=lat, elevation=elevation)
    for coordinates in runs:
        if len(coordinates) < 2:
            continue
        coordinates = coordinates.copy()
        lon_0, lat_0, elevation_0 = coordinates.pop(0)
        for lon_1, lat_1, elevation_1 in coordinates:
            graph.add_edge(
                (lon_0, lat_0),
                (lon_1, lat_1),
                vertical=max(0.0, elevation_0 - elevation_1),
            )
            lon_0, lat_0, elevation_0 = lon_1, lat_1, elevation_1
    if graph.number_of_edges() > 0:
        graph = add_edge_bearings(graph)
        graph = add_edge_lengths(graph)
    return graph


def create_networkx_with_metadata(
    runs: list[list[tuple[float, float, float]]],
    ski_area_metadata: dict[str, Any],
) -> nx.MultiDiGraph:
    graph = create_networkx(runs)
    graph.graph = ski_area_metadata | graph.graph
    if graph.number_of_nodes() > 0:
        graph.graph["latitude"] = statistics.mean(
            lat for _, lat in graph.nodes(data="y")
        )
        graph.graph["longitude"] = statistics.mean(
            lon for _, lon in graph.nodes(data="x")
        )
        graph.graph["hemisphere"] = "north" if graph.graph["latitude"] > 0 else "south"
    if graph.number_of_edges() > 0:
        with suppress_user_warning():
            bearings, weights = osmnx.bearing._extract_edge_bearings(
                graph, min_length=0, weight="vertical"
            )
        graph.graph["combined_vertical"] = sum(weights)
        stats = get_bearing_summary_stats(
            bearings=bearings,
            cum_magnitudes=weights,
            hemisphere=graph.graph["hemisphere"],
        )
        graph.graph.update(stats.model_dump())
    return graph
