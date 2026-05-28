from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _installed_version() -> str:
    try:
        return version("anila")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _installed_version()
