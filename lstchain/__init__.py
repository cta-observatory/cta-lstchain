from . import reco
from . import io
from . import visualization
from . import calib
from . import mc
from . import spectra
from . import image
from . import irf
from . import tools

from .io import standard_config

from .version import get_version
__version__ = get_version(pep440=False)
