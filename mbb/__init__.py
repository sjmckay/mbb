from .mbb import ModifiedBlackbody
from . import mbb_funcs
__all__ = ["ModifiedBlackbody", "mbb_funcs"]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mbb")
except PackageNotFoundError:
    __version__ = "0.0.0"