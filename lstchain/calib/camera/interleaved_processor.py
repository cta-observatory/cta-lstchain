from ctapipe.containers import ArrayEventContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import Bool, Path
from ctapipe.io import DataWriter

from .calibration_calculator import LSTCalibrationCalculator


class LSTInterleavedProcessor(TelescopeComponent):
    """
    Class to handle interleaved pedestal and flatfield events.
    """

    write_interleaved = Bool(
        False, help="Write r1 waveforms of interleaved events to a .h5 file."
    ).tag(config=True)

    output_path = Path(
        default_value=None,
        allow_none=True,
        exists=False,
        directory_ok=False,
        help="Path to output file.",
    ).tag(config=True)

    def __init__(
        self,
        event_source,
        subarray,
        config=None,
        parent=None,
        write_only_mode=False,
        **kwargs,
    ):
        super().__init__(subarray, config, parent, **kwargs)

        self.interleaved_processor = LSTCalibrationCalculator(
            parent=self, subarray=subarray
        )

        self.write_only_mode = write_only_mode
        self.interleaved_writer = None
        if self.write_interleaved and self.output_path is not None:
            self.interleaved_writer = DataWriter(
                parent=self, event_source=event_source, output_path=self.output_path
            )

    def __call__(self, event: ArrayEventContainer) -> None:
        pass
