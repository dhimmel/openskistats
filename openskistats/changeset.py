import requests
import xml.etree.ElementTree as ET
import polars as pl
from openskistats.analyze import load_runs_pl, load_ski_areas_pl

import re

def parse_osm_url(url: str):
    """Extract type and ID from an OpenStreetMap URL."""
    match = re.match(r"https://www\.openstreetmap\.org/(way|relation)/(\d+)", url)
    if match:
        osm_type, osm_id = match.groups()
        return osm_type, int(osm_id)
    return None, None

def fetch_changeset_history(osm_type: str, osm_id: int):
    """Fetch the changeset history for a given OSM entity."""
    url = f"https://api.openstreetmap.org/api/0.6/{osm_type}/{osm_id}/history"
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse the XML response
    root = ET.fromstring(response.content)
    changesets = []
    for element in root.findall(f"./{osm_type}"):
        changesets.append({
            "changeset_id": element.attrib.get("changeset"),
            "user": element.attrib.get("user"),
            "timestamp": element.attrib.get("timestamp"),
        })
    return changesets

def process_batch(batch: pl.DataFrame) -> pl.DataFrame:
    """Process a batch of OSM URLs to fetch their changeset histories."""
    changeset_records = []
    for url in batch["osm_url"]:
        osm_type, osm_id = parse_osm_url(url)
        if osm_type and osm_id:
            history = fetch_changeset_history(osm_type, osm_id)
            for record in history:
                record["osm_url"] = url
                record["osm_type"] = osm_type
                record["osm_id"] = osm_id
                changeset_records.append(record)
    return pl.DataFrame(changeset_records)

def get_changesets_for_runs_and_ski_areas() -> pl.DataFrame:
    run_sources = (
        load_runs_pl()
        .explode("run_sources")
        .select(
            "run_id",
            "run_name",
            "ski_area_ids",
            pl.col("run_sources").alias("run_source"),
        )
        .collect()
    )

    ski_area_sources = (
        load_ski_areas_pl()
        .explode("ski_area_sources")
        .select(
            "ski_area_id",
            "ski_area_name",
            pl.col("ski_area_sources").alias("ski_area_source"),
        )
        .filter(pl.col("ski_area_source").str.starts_with("https://www.openstreetmap.org"))
    )

    osm_urls = sorted(set(ski_area_sources["ski_area_source"].to_list()) | set(run_sources["run_source"].to_list()))

    osm_urls_df = pl.DataFrame({"osm_url": osm_urls})

    # Process the OSM URLs using a map function
    changeset_pl_df = osm_urls_df.select(
        pl.col("osm_url").map_elements(
            lambda url: process_batch(pl.DataFrame({"osm_url": [url]})),
            return_dtype=pl.Object
        ).alias("changeset_data")
    ).explode("changeset_data")

    # Display the resulting DataFrame
    return changeset_pl_df