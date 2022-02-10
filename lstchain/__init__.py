import astropy
from astropy.time import TimeUnixTai, TimeFromEpoch

from . import calib
from . import data
from . import high_level
from . import image
from . import io
from . import mc
from . import reco
from . import spectra
from . import visualization
from .io import standard_config
from .version import __version__

__all__ = [
    "__version__",
    "data",
    "high_level",
    "calib",
    "image",
    "io",
    "mc",
    "reco",
    "spectra",
    "standard_config",
    "visualization",
]

if (
        astropy.version.major == 4
        and astropy.version.minor <= 2
        and astropy.version.bugfix <= 0
):
    # clear the cache to not depend on import orders
    TimeFromEpoch.__dict__["_epoch"]._cache.clear()
    # fix for astropy #11245, epoch was wrong by 8 seconds
    TimeUnixTai.epoch_val = "1970-01-01 00:00:00.0"
    TimeUnixTai.epoch_scale = "tai"
