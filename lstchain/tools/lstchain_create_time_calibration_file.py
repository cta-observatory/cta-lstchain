"""
Create drs4 time correction coefficients.
"""
import glob

import numpy as np
from ctapipe.core import Provenance, Tool, traits
from ctapipe_io_lst import LSTEventSource
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.time_correction_calculate import TimeCorrectionCalculate


class TimeCalibrationHDF5Writer(Tool):

    name = "TimeCalibrationHDF5Writer"
    description = "Generate a HDF5 file with time calibration coefficients"

    input = traits.Unicode(
        help="Path to the fits.fz input file containing events (wildcards allowed)",
    ).tag(config=True)

    output = traits.Unicode(
        help="Path to the hdf5 time calibration file",
    ).tag(config=True)

    pedestal = traits.Unicode(
        help="Path to drs4 fits pedestal file",
    ).tag(config=True)

    max_events = traits.Int(
        help="Maximum numbers of events to read. Default = 20000", default_value=20000
    ).tag(config=True)

    aliases = {
        "input": "TimeCalibrationHDF5Writer.input",
        "output": "TimeCalibrationHDF5Writer.output",
        "pedestal": "LSTR0Corrections.pedestal_path",
        "max_events": "TimeCalibrationHDF5Writer.max_events",
    }

    classes = [LSTEventSource, LSTR0Corrections, TimeCorrectionCalculate]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Tool that generates a HDF5 file with time calibration coefficients.

        For getting help run:
        lstchain_create_time_calibration_file --help
        """
        self.reader = None
        self.timeCorr = None
        self.path_list = None
        self.lst_r0 = None

    def setup(self):

        self.path_list = sorted(glob.glob(self.input))
        self.reader = self.add_component(
            LSTEventSource(
                input_url=self.path_list[0],
                max_events=self.max_events,
                config=self.config,
            )
        )
        self.lst_r0 = self.add_component(
            LSTR0Corrections(pedestal_path=self.pedestal, parent=self)
        )
        self.timeCorr = TimeCorrectionCalculate(
            calib_file_path=self.output,
            subarray=self.reader.subarray,
            config=self.config,
        )

    def start(self):

        try:
            for j, path in enumerate(self.path_list):
                self.reader.input_url = path
                self.log.info(f"File {j + 1} out of {len(self.path_list)}")
                self.log.info(f"Processing: {path}")
                for i, event in enumerate(self.reader):
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

        self.timeCorr.finalize()
        Provenance().add_output_file(
            self.output,
            role='mon.tel.calibration'
        )


def main():
    exe = TimeCalibrationHDF5Writer()
    exe.run()


if __name__ == "__main__":
    main()
