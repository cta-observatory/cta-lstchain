from . import reco
from . import io
from . import visualization
from . import calib
from . import mc
from . import spectra
from . import image

from .io import standard_config

from . import version
__version__ = version.get_version(pep440=False)
