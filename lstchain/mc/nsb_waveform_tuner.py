from ctapipe.core import Component
from ctapipe.containers import ArrayEventContainer


__all__ = [
    "WaveformNSBTuner",
]


class WaveformNSBTuner(Component):
    """
    Class to apply waveform tuning.
    """

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Tune waveforms
        """
        pass
