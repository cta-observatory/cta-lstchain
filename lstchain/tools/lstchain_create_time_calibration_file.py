"""
Create drs4 time correction coefficients.
"""
import numpy as np

from ctapipe.core import Provenance, traits
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSource
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.time_correction_calculate import TimeCorrectionCalculate


class TimeCalibrationHDF5Writer(Tool):

    name = "TimeCalibrationHDF5Writer"
    description = "Generate a HDF5 file with time calibration coefficients"

    output_file = traits.Unicode(
        default_value="time_calibration.hdf5",
        help="Path to the generated the generated HDF5 time calibration file",
    ).tag(config=True)

    aliases = {
        "input_file": "EventSource.input_url",
        "output_file": "TimeCorrectionCalculate.calib_file_path",
        "pedestal_file": "LSTR0Corrections.pedestal_path",
    }

    classes = [EventSource, LSTR0Corrections]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Tool that generates a HDF5 file with time calibration coefficients.

        For getting help run:
        python lstchain_create_time_calibration_file.py --help
        """

        self.eventsource = None
        self.timeCorr = None
        self.lst_r0 = None

    def setup(self):

        self.log.debug(f"Opening file")
        self.eventsource = EventSource.from_config(parent=self)
        self.lst_r0 = LSTR0Corrections(config=self.config)
        self.timeCorr = TimeCorrectionCalculate(
            config=self.config,
            subarray=self.eventsource.subarray,
        )

    def start(self):

        try:
            for i, event in enumerate(self.eventsource):
                if i % 5000 == 0:
                    self.log.debug(f"i = {i}, ev id = {event.index.event_id}")
                self.lst_r0.calibrate(event)

                # cut in signal to avoid cosmic events
                if event.r1.tel[self.timeCorr.tel_id].trigger_type == 4 or (
                        np.median(np.sum(event.r1.tel[self.timeCorr.tel_id].waveform[0], axis=1)) > 300
                ):
                    self.timeCorr.calibrate_peak_time(event)

        except Exception as e:
            self.log.error(e)

    def finish(self):
        # Provenance().add_output_file(
        #     self.output_file,
        #     role='mon.tel.calibration'
        # )
        # self.writer.close()
        pass


def main():
    exe = TimeCalibrationHDF5Writer()
    exe.run()


if __name__ == "__main__":
    main()
