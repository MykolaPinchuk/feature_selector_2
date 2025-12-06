"""Core package for the XGBoost feature-selection framework."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fs-xgb")
except PackageNotFoundError:  # pragma: no cover - fallback for local usage before install
    __version__ = "0.0.0"

__all__ = ["__version__"]
