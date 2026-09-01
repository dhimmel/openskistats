import zipfile
from pathlib import Path

from openskistats.zenodo import (
    DEPOSIT_TITLE,
    build_directory_zip,
    gather_deposit_files,
    get_commit_sha,
    get_deposit_creators,
    get_deposit_metadata,
    get_deposit_readme_markdown,
)

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"


def test_get_commit_sha_returns_head_hash() -> None:
    sha = get_commit_sha()
    assert len(sha) == 40
    int(sha, 16)


def test_get_deposit_creators_derive_from_manuscript_frontmatter() -> None:
    creators = get_deposit_creators()
    assert len(creators) >= 2
    first = creators[0]["person_or_org"]
    assert first["family_name"] == "Himmelstein"
    assert first["given_name"] == "Daniel"
    assert first["identifiers"] == [
        {"scheme": "orcid", "identifier": "0000-0002-3012-7446"}
    ]


def test_get_deposit_readme_markdown() -> None:
    readme = get_deposit_readme_markdown(commit_sha=FAKE_SHA)
    assert readme.startswith(f"# {DEPOSIT_TITLE}\n")
    assert f"@{FAKE_SHA[:7]}`" in readme
    assert f"tree/{FAKE_SHA}" in readme
    # single data date derived from the test data info.json
    assert "OpenSkiMap](https://openskimap.org) data of 2025-12-21" in readme


def test_get_deposit_metadata() -> None:
    metadata = get_deposit_metadata(commit_sha=FAKE_SHA)
    assert metadata["title"] == DEPOSIT_TITLE
    # the README title heading is stripped so the record page does not repeat it
    assert not metadata["description"].startswith("<h1>")
    assert "<p>" in metadata["description"]
    assert {right["id"] for right in metadata["rights"]} == {
        "odbl-1.0",
        "bsd-2-clause-patent",
        "cc-by-4.0",
    }
    related = [ri["identifier"] for ri in metadata["related_identifiers"]]
    assert f"https://github.com/dhimmel/openskistats/tree/{FAKE_SHA}" in related


def test_build_directory_zip(tmp_path: Path) -> None:
    source = tmp_path.joinpath("bundle")
    source.joinpath("nested").mkdir(parents=True)
    source.joinpath("a.txt").write_text("alpha")
    source.joinpath("nested", "b.txt").write_text("beta")
    zip_path = build_directory_zip(
        source=source, destination=tmp_path.joinpath("bundle.zip")
    )
    with zipfile.ZipFile(zip_path) as zip_file:
        assert zip_file.namelist() == ["bundle/a.txt", "bundle/nested/b.txt"]
        assert all(
            info.compress_type == zipfile.ZIP_DEFLATED for info in zip_file.infolist()
        )


def test_gather_deposit_files(tmp_path: Path) -> None:
    files = gather_deposit_files(tmp_path, commit_sha=get_commit_sha())
    expected_keys = {
        "README.md",
        "code.zip",
        "openskimap/info.json",
        "ski_area_metrics.parquet",
        "_variables.yaml",
        "webapp.zip",
        "images.zip",
    }
    assert expected_keys <= files.keys()
    for key, path in files.items():
        assert path.is_file(), key
        assert path.stat().st_size > 0, key
