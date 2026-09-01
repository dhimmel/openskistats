"""
Deposit OpenSkiStats snapshots to Zenodo for long-term archival with DOIs.

Uses Zenodo's InvenioRDM REST API directly
(<https://inveniordm.docs.cern.ch/reference/rest_api_drafts_records/>),
which unlike the legacy deposit API supports multiple licenses per record.
"""

import dataclasses
import logging
import os
import subprocess
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import httpx2
import yaml
from dotenv import load_dotenv
from markdown_it import MarkdownIt

from openskistats.openskimap_utils import load_openskimap_download_info
from openskistats.utils import (
    get_data_directory,
    get_repo_directory,
    get_website_source_directory,
)

DEPOSIT_TITLE = "OpenSkiStats snapshot"


def get_deposit_readme_markdown(commit_sha: str) -> str:
    """
    Markdown that single-sources the deposit README and the record description:
    the README travels with the downloaded files,
    while the record page shows its HTML rendering as the description
    (the InvenioRDM description field accepts only sanitized HTML).
    """
    download_lines = "\n".join(
        f"  - `openskimap/{Path(info.relative_path).name}`"
        f" retrieved {info.downloaded} (upstream last modified {info.last_modified})"
        for info in load_openskimap_download_info().values()
    )
    return f"""\
# {DEPOSIT_TITLE}

Archival snapshot of [OpenSkiStats](https://openskistats.org), deposited {date.today().isoformat()},
produced by [`dhimmel/openskistats@{commit_sha[:7]}`](https://github.com/dhimmel/openskistats/tree/{commit_sha}).

OpenSkiStats generates statistics on downhill ski slopes and areas worldwide
from OpenSkiMap/OpenStreetMap data.
This deposit contains the exact inputs, source code, and outputs of one analysis run:

- `code.zip`: repository source code at the producing commit
- `openskimap/`: GeoJSON inputs downloaded from [OpenSkiMap](https://openskimap.org),
  with download provenance in `openskimap/info.json`:
{download_lines}
- `*.parquet`: derived outputs for runs, lifts, and ski areas
- `_variables.yaml`: computed statistics interpolated into the website and manuscript
- `webapp.zip`: the rendered website served at [openskistats.org](https://openskistats.org)
- `images.zip`: figures

Licensing varies by component:
data derived from OpenSkiMap/OpenStreetMap (`openskimap/`, `*.parquet`, `_variables.yaml`)
is released under the [Open Database License](https://opendatacommons.org/licenses/odbl/) (ODbL);
code is BSD-2-Clause-Patent;
produced works such as the website and figures are CC-BY-4.0.
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


def get_deposit_payload(commit_sha: str) -> dict[str, Any]:
    """Full draft payload: record metadata plus Zenodo custom fields."""
    return {
        "metadata": get_deposit_metadata(commit_sha=commit_sha),
        "custom_fields": get_deposit_custom_fields(),
    }


def get_deposit_custom_fields() -> dict[str, Any]:
    """
    Zenodo custom fields (the deposit form's Software section),
    from the CodeMeta-derived `code:` namespace.
    """
    return {
        "code:codeRepository": "https://github.com/dhimmel/openskistats",
        "code:developmentStatus": {"id": "active"},
        "code:programmingLanguage": [{"id": "python"}, {"id": "typescript"}],
    }


def get_deposit_metadata(commit_sha: str) -> dict[str, Any]:
    """Record-level metadata for a snapshot deposit in InvenioRDM format."""
    return {
        "resource_type": {"id": "dataset"},
        "title": DEPOSIT_TITLE,
        "publication_date": date.today().isoformat(),
        "version": date.today().isoformat(),
        # Strip the README's title heading since the record page shows the title itself.
        "description": MarkdownIt().render(
            get_deposit_readme_markdown(commit_sha=commit_sha).removeprefix(
                f"# {DEPOSIT_TITLE}\n\n"
            )
        ),
        # Required for DOI registration.
        "publisher": "Zenodo",
        "creators": get_deposit_creators(),
        # License ids from the /api/vocabularies/licenses vocabulary.
        "rights": [
            {"id": "odbl-1.0"},
            {"id": "bsd-2-clause-patent"},
            {"id": "cc-by-4.0"},
        ],
        # Relation ids from the /api/vocabularies/relationtypes vocabulary.
        # `issupplementto` for the repository matches Zenodo's own GitHub integration.
        "related_identifiers": [
            {
                "identifier": f"https://github.com/dhimmel/openskistats/tree/{commit_sha}",
                "scheme": "url",
                "relation_type": {"id": "issupplementto"},
                "resource_type": {"id": "software"},
            },
            {
                "identifier": "https://openskistats.org",
                "scheme": "url",
                "relation_type": {"id": "issourceof"},
            },
            {
                "identifier": "https://openskimap.org",
                "scheme": "url",
                "relation_type": {"id": "isderivedfrom"},
            },
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

    def create_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a new draft record from a payload of `metadata` and `custom_fields`."""
        response = self._request("POST", "/api/records", json=payload)
        result: dict[str, Any] = response.json()
        return result

    def create_version_draft(self, record_id: str) -> dict[str, Any]:
        """Create a draft for a new version of a published record and return its JSON."""
        response = self._request("POST", f"/api/records/{record_id}/versions")
        result: dict[str, Any] = response.json()
        return result

    def update_draft_metadata(
        self, record_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Replace a draft's metadata (a new-version draft clears `publication_date`)."""
        response = self._request("PUT", f"/api/records/{record_id}/draft", json=payload)
        result: dict[str, Any] = response.json()
        return result

    def upload_file(self, record_id: str, path: Path, key: str | None = None) -> None:
        """
        Register, upload, and commit one file to a draft record.
        `key` names the file on the record and may contain `/` to convey directories;
        it defaults to the file's basename.
        """
        key = key or path.name
        self._request(
            "POST",
            f"/api/records/{record_id}/draft/files",
            json=[{"key": key}],
        )
        # Read fully into memory to send a Content-Length header:
        # generator content triggers chunked transfer encoding,
        # which Zenodo's file endpoint silently stores as zero bytes.
        self._request(
            "PUT",
            f"/api/records/{record_id}/draft/files/{key}/content",
            content=path.read_bytes(),
        )
        self._request("POST", f"/api/records/{record_id}/draft/files/{key}/commit")

    def publish_draft(self, record_id: str) -> dict[str, Any]:
        """Publish a draft record and return the published record JSON."""
        response = self._request(
            "POST", f"/api/records/{record_id}/draft/actions/publish"
        )
        result: dict[str, Any] = response.json()
        return result


def write_deposit_readme(directory: Path, commit_sha: str) -> Path:
    """Write the deposit README."""
    path = directory.joinpath("README.md")
    path.write_text(get_deposit_readme_markdown(commit_sha=commit_sha))
    return path


def get_commit_sha() -> str:
    """Return the commit hash of the repository HEAD."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=get_repo_directory(), text=True
    ).strip()


def build_code_zip(directory: Path, commit_sha: str) -> Path:
    """
    Archive the repository source at `commit_sha` via `git archive`.
    Contains only the files tracked at that commit:
    no git history, no untracked or gitignored files (`.env`, `data/`, caches),
    minus any paths marked `export-ignore` in `.gitattributes`.
    Git history is archived separately by Software Heritage.
    """
    path = directory.joinpath("code.zip")
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={path}", commit_sha],
        cwd=get_repo_directory(),
        check=True,
    )
    return path


def build_directory_zip(source: Path, destination: Path) -> Path:
    """
    Zip a directory's files with deflate, nested under the directory's name.
    Deflate over zstd for archival compatibility:
    stock extractors like Info-ZIP unzip skip zstd entries.
    """
    with zipfile.ZipFile(
        file=destination, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zip_file:
        for file in sorted(source.rglob("*")):
            if file.is_file():
                arcname = f"{source.name}/{file.relative_to(source).as_posix()}"
                zip_file.write(file, arcname=arcname)
    return destination


def gather_deposit_files(temp_dir: Path, commit_sha: str) -> dict[str, Path]:
    """
    Collect the deposit contents as a mapping of record file key to local path:
    all of `data/openskimap/`, all root-level `data/` files,
    and a zip per bundled directory.
    Record keys mirror the `data/` directory layout.
    """
    data_dir = get_data_directory()
    files = {
        "README.md": write_deposit_readme(temp_dir, commit_sha=commit_sha),
    }
    code_zip = build_code_zip(temp_dir, commit_sha=commit_sha)
    files[code_zip.name] = code_zip
    for path in sorted(data_dir.joinpath("openskimap").iterdir()):
        if path.is_file():
            files[f"openskimap/{path.name}"] = path
    for path in sorted(data_dir.iterdir()):
        if path.is_file():
            files[path.name] = path
    for name in ["webapp", "images"]:
        files[f"{name}.zip"] = build_directory_zip(
            source=data_dir.joinpath(name),
            destination=temp_dir.joinpath(f"{name}.zip"),
        )
    return files


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
    commit_sha = get_commit_sha()
    payload = get_deposit_payload(commit_sha=commit_sha)
    if record_id is None:
        draft = client.create_draft(payload=payload)
    else:
        draft = client.create_version_draft(record_id=record_id)
        draft = client.update_draft_metadata(record_id=draft["id"], payload=payload)
    draft_id = draft["id"]
    with tempfile.TemporaryDirectory() as temp_dir:
        files = gather_deposit_files(Path(temp_dir), commit_sha=commit_sha)
        for key, path in files.items():
            logging.info(f"Uploading {key} ({path.stat().st_size / 1024**2:.1f} MB)")
            client.upload_file(record_id=draft_id, path=path, key=key)
    record = client.publish_draft(record_id=draft_id) if publish else draft
    links = record.get("links", {})
    logging.info(f"Record {record['id']} status={record.get('status')}")
    logging.info(f"View at {links.get('self_html')}")
    return record
