from ctapipe.core import Component
from ctapipe.containers import ArrayEventContainer
from ctapipe.core.traits import Float, Path, Bool


__all__ = [
    "WaveformNSBTuner",
]


class WaveformNSBTuner(Component):
    """
    Class to apply waveform tuning.
    """

    apply_waveform_tuning = Bool(
        help="Adds NSB in waveforms",
        default_value=False,
    ).tag(config=True)

    nsb_tuning_ratio = Float(
        help="NSB tuning ratio",
        default_value=0.52,
    ).tag(config=True)

    spe_location = Path(
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
        help="Path to spe file",
    ).tag(config=True)

    target_data = Path(
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
        help="If nsb_tuning_ratio=None, calculate the"
        "tuning ratio based on target data.",
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Tune waveforms
        """
        pass
