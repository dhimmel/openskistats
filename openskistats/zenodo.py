"""
Deposit OpenSkiStats snapshots to Zenodo for long-term archival with DOIs.
Phase 0 scope per `local/zenodo-deposit-plan.md`:
a minimal deposit of `ski_area_metrics.parquet` plus a stub README
to the Zenodo sandbox to exercise the create/draft/publish/version lifecycle.

Uses Zenodo's InvenioRDM REST API directly
(<https://inveniordm.docs.cern.ch/reference/rest_api_drafts_records/>),
which unlike the legacy deposit API supports multiple licenses per record.
"""

import dataclasses
import logging
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import httpx2
import yaml
from dotenv import load_dotenv
from markdown_it import MarkdownIt

from openskistats.utils import (
    get_data_directory,
    get_repo_directory,
    get_website_source_directory,
)

DEPOSIT_TITLE = "OpenSkiStats snapshot (development test)"

# Authored in markdown and rendered to HTML at deposit time,
# since the InvenioRDM description field accepts only sanitized HTML.
DEPOSIT_DESCRIPTION = """
Test deposit for developing the OpenSkiStats archival pipeline.
Do not cite: this record exercises deposit automation and will be superseded.

OpenSkiStats generates statistics on downhill ski slopes and areas worldwide
from OpenSkiMap/OpenStreetMap data.
See [openskistats.org](https://openskistats.org)
and [github.com/dhimmel/openskistats](https://github.com/dhimmel/openskistats).

Licensing varies by component as described in the deposited README:
data derived from OpenSkiMap/OpenStreetMap is ODbL,
code is BSD-2-Clause-Patent,
and produced works such as figures are CC-BY-4.0.
"""


def get_deposit_creators() -> list[dict[str, Any]]:
    """
    Derive Zenodo creators from the manuscript's Quarto `author` frontmatter,
    the canonical authorship source for the project.
    Expects the explicit Quarto author schema
    (`name: {given: ..., family: ...}` with an optional `orcid`).
    """
    manuscript_path = get_website_source_directory().joinpath("manuscript", "index.qmd")
    frontmatter = yaml.safe_load(manuscript_path.read_text().split("---\n")[1])
    creators = []
    for author in frontmatter["author"]:
        person: dict[str, Any] = {
            "type": "personal",
            "given_name": author["name"]["given"],
            "family_name": author["name"]["family"],
        }
        if orcid := author.get("orcid"):
            person["identifiers"] = [{"scheme": "orcid", "identifier": orcid}]
        creators.append({"person_or_org": person})
    return creators


def get_deposit_metadata() -> dict[str, Any]:
    """Record-level metadata for a snapshot deposit in InvenioRDM format."""
    return {
        "resource_type": {"id": "dataset"},
        "title": DEPOSIT_TITLE,
        "publication_date": date.today().isoformat(),
        "description": MarkdownIt().render(DEPOSIT_DESCRIPTION),
        # Required for DOI registration.
        "publisher": "Zenodo",
        "creators": get_deposit_creators(),
        # License ids from the /api/vocabularies/licenses vocabulary.
        "rights": [
            {"id": "odbl-1.0"},
            {"id": "bsd-2-clause-patent"},
            {"id": "cc-by-4.0"},
        ],
    }


@dataclasses.dataclass(frozen=True)
class ZenodoClient:
    """Minimal Zenodo InvenioRDM API client scoped to what depositing requires."""

    access_token: str
    base_url: str

    @classmethod
    def from_environment(cls, sandbox: bool = True) -> ZenodoClient:
        """
        Authenticate from `ZENODO_SANDBOX_API_TOKEN` (or `ZENODO_API_TOKEN` when `sandbox=False`),
        loading the repository `.env` file if present.
        """
        load_dotenv(dotenv_path=get_repo_directory().joinpath(".env"))
        variable = "ZENODO_SANDBOX_API_TOKEN" if sandbox else "ZENODO_API_TOKEN"
        access_token = os.environ.get(variable)
        if not access_token:
            raise RuntimeError(
                f"Set {variable} in the environment or the repository .env file."
            )
        base_url = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
        return cls(access_token=access_token, base_url=base_url)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx2.Response:
        response = httpx2.request(
            method=method,
            url=f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=300,
            **kwargs,
        )
        if not response.is_success:
            raise RuntimeError(
                f"Zenodo API {method} {path} failed with {response.status_code}: {response.text}"
            )
        return response

    def create_draft(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Create a new draft record and return its JSON."""
        response = self._request("POST", "/api/records", json={"metadata": metadata})
        result: dict[str, Any] = response.json()
        return result

    def create_version_draft(self, record_id: str) -> dict[str, Any]:
        """Create a draft for a new version of a published record and return its JSON."""
        response = self._request("POST", f"/api/records/{record_id}/versions")
        result: dict[str, Any] = response.json()
        return result

    def update_draft_metadata(
        self, record_id: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Replace a draft's metadata (a new-version draft clears `publication_date`)."""
        response = self._request(
            "PUT", f"/api/records/{record_id}/draft", json={"metadata": metadata}
        )
        result: dict[str, Any] = response.json()
        return result

    def upload_file(self, record_id: str, path: Path) -> None:
        """Register, upload, and commit one file to a draft record."""
        self._request(
            "POST",
            f"/api/records/{record_id}/draft/files",
            json=[{"key": path.name}],
        )
        # Read fully into memory to send a Content-Length header:
        # generator content triggers chunked transfer encoding,
        # which Zenodo's file endpoint silently stores as zero bytes.
        self._request(
            "PUT",
            f"/api/records/{record_id}/draft/files/{path.name}/content",
            content=path.read_bytes(),
        )
        self._request(
            "POST", f"/api/records/{record_id}/draft/files/{path.name}/commit"
        )

    def publish_draft(self, record_id: str) -> dict[str, Any]:
        """Publish a draft record and return the published record JSON."""
        response = self._request(
            "POST", f"/api/records/{record_id}/draft/actions/publish"
        )
        result: dict[str, Any] = response.json()
        return result


def write_deposit_readme(directory: Path) -> Path:
    """Write the deposit README that orients readers landing on the Zenodo record."""
    path = directory.joinpath("README.md")
    path.write_text(
        "# OpenSkiStats snapshot (development test)\n\n"
        f"Deposited {date.today().isoformat()}.\n"
        "This is a test deposit for developing the OpenSkiStats archival pipeline.\n"
        "Do not cite.\n\n"
        "- `ski_area_metrics.parquet`: per-ski-area metrics (ODbL,"
        " derived from OpenSkiMap/OpenStreetMap)\n"
    )
    return path


def deposit_snapshot(
    record_id: str | None = None,
    publish: bool = False,
    sandbox: bool = True,
) -> dict[str, Any]:
    """
    Deposit a snapshot to Zenodo and return the resulting record JSON.
    Without `record_id`, creates a new record (the concept record);
    with one, creates a new version of that record.
    Deposits remain unpublished drafts unless `publish` is set.
    """
    client = ZenodoClient.from_environment(sandbox=sandbox)
    if record_id is None:
        draft = client.create_draft(metadata=get_deposit_metadata())
    else:
        draft = client.create_version_draft(record_id=record_id)
        draft = client.update_draft_metadata(
            record_id=draft["id"], metadata=get_deposit_metadata()
        )
    draft_id = draft["id"]
    metrics_path = get_data_directory().joinpath("ski_area_metrics.parquet")
    with tempfile.TemporaryDirectory() as temp_dir:
        for path in [write_deposit_readme(Path(temp_dir)), metrics_path]:
            client.upload_file(record_id=draft_id, path=path)
    record = client.publish_draft(record_id=draft_id) if publish else draft
    links = record.get("links", {})
    logging.info(f"Record {record['id']} status={record.get('status')}")
    logging.info(f"View at {links.get('self_html')}")
    return record
