from ctapipe.containers import ArrayEventContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import Path


class CatBCalibrator(TelescopeComponent):
    """
    Component to apply the cat B calibrations on images and peak_times.
    """

    calibrations_path = Path(help="Path to cat-B calibrations file").tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

    def __call__(self, event: ArrayEventContainer):
        pass
