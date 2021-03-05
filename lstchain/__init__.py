from astropy.time import TimeUnixTai, TimeFromEpoch
import astropy

from . import reco
from . import io
from . import visualization
from . import calib
from . import mc
from . import spectra
from . import image
from .io import standard_config
from .version import get_version

__all__ = [
    'reco', 'io', 'visualization', 'calib', 'mc', 'spectra', 'image',
    'standard_config', '__version__'
]

__version__ = get_version(pep440=False)


if astropy.version.major == 4 and astropy.version.minor <= 2 and astropy.version.bugfix <= 0:
    # clear the cache to not depend on import orders
    TimeFromEpoch.__dict__['_epoch']._cache.clear()
    # fix for astropy #11245, epoch was wrong by 8 seconds
    TimeUnixTai.epoch_val = '1970-01-01 00:00:00.0'
    TimeUnixTai.epoch_scale = 'tai'
