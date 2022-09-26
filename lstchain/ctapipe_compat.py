"""
Module containing workarounds / wrappers for ctapipe code with fixes for lstchain.

The idea is that everything here should be fixed upstream and than an import
like `from lstchain.ctapipe_compat import Foo` can just be replaced by doing
`from ctapipe.<module> import Foo` when upgrading to a ctapipe version containing
the fix.
"""
from ctapipe.core import Component as UpstreamComponent
from ctapipe.calib.camera.pedestals import PedestalCalculator as UpstreamPedestalCalculator
from ctapipe.calib.camera.flatfield import FlatFieldCalculator as UpstreamFlatFieldCalculator


__all__ = [
    "Component",
    "PedestalCalculator",
    "FlatFieldCalculator",
]



# FIXME: Workaround for a logging issue in ctapipe < 0.15, remove when upgrading
# to 0.15
# issue: https://github.com/cta-observatory/ctapipe/issues/1882
# Release with fix: https://github.com/cta-observatory/ctapipe/releases/tag/v0.15.0
class Component(UpstreamComponent):
    """Wrapper for ctapipe.core.Component fixing a logging issue"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.parent is not None:
            self.log = self.parent.log.getChild(self.__class__.__name__)


class PedestalCalculator(UpstreamPedestalCalculator):
    """Wrapper for ctapipe.calib.camera.pedestal.PedestalCalculator fixing a logging issue"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.parent is not None:
            self.log = self.parent.log.getChild(self.__class__.__name__)


class FlatFieldCalculator(UpstreamFlatFieldCalculator):
    """Wrapper for ctapipe.calib.camera.flatfield.FlatFieldCalculator fixing a logging issue"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.parent is not None:
            self.log = self.parent.log.getChild(self.__class__.__name__)

