"""
Variables stored during analysis for website templating usage.
"""

from pathlib import Path
from typing import Any

import yaml

from openskistats.utils import get_data_directory


def _get_variables_path() -> Path:
    variables_path = get_data_directory().joinpath("_variables.yaml")
    if not variables_path.exists():
        variables_path.touch()
    return variables_path


def _read_variables() -> dict[str, Any]:
    return yaml.safe_load(stream=_get_variables_path().read_text()) or {}


def read_variable(key: str, format_spec: str | None) -> Any:
    """key supports nested access with __ notation"""
    variable = _read_variables()
    for part in key.split("__"):
        variable = variable[part]
    if format_spec is None:
        return variable
    return format(variable, format_spec)


def set_variables(**kwargs: Any) -> None:
    """key supports nested access with __ notation"""
    variables = _read_variables()
    for key, value in kwargs.items():
        parts = key.split("__")
        variable = variables
        for part in parts[:-1]:
            variable = variable.setdefault(part, {})
        variable[parts[-1]] = value
    _get_variables_path().write_text(yaml.dump(variables))
