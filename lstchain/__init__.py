from astropy.time import TimeUnixTai
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
    # fix for astropy #11245
    TimeUnixTai.epoch_val = '1970-01-01 00:00:00.0'
    TimeUnixTai.epoch_scale = 'tai'
