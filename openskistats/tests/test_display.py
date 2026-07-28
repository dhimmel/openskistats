import htmltools
import reactable

from openskistats.display import _format_header, country_code_to_emoji


def test_country_code_to_emoji() -> None:
    assert country_code_to_emoji("US") == "🇺🇸"
    assert country_code_to_emoji("FR") == "🇫🇷"


def test_format_header_tooltip() -> None:
    header = _format_header(reactable.HeaderCellInfo(name="latitude", value="Latitude"))
    assert isinstance(header, htmltools.Tag)
    assert header.attrs["tabindex"] == "0"
    assert header.attrs["data-tippy-content"]
