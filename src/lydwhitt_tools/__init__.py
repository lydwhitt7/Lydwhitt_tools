# Public API
from .mahalanobis import geochemical_filter
from .KDE import KDE, MD, iqr_one_peak
from .formula_recalc import recalc
from .formula_recalc import recalc_Fe


try:
    from importlib.metadata import version as _v
except ImportError:  # Python <3.8 backport, but you require >=3.8 so this is mostly moot
    from importlib_metadata import version as _v  # pragma: no cover

try:
    __version__ = _v("lydwhitt-tools")  # use the PyPI/distribution name
except Exception:
    __version__ = "0"  # or leave undefined

__all__ = ["geochemical_filter", "__version__"]