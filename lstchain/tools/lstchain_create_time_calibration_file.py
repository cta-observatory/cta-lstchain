"""
Create drs4 time correction coefficients.
"""
import glob

import numpy as np
from ctapipe.core import Provenance, Tool, traits
from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.calibration import LSTR0Corrections
from lstchain.calib.camera.time_correction_calculate import TimeCorrectionCalculate
from tqdm.autonotebook import tqdm


class TimeCalibrationHDF5Writer(Tool):

    name = "TimeCalibrationHDF5Writer"
    description = "Generate a HDF5 file with time calibration coefficients"

    input = traits.Path(
        help="Path to the fits.fz events file or directory to glob"
    ).tag(config=True)

    glob = traits.Unicode(
        help="Filename pattern to glob files in the directory",
        default_value="*"
    ).tag(config=True)

    max_events = traits.Int(
        help="Maximum numbers of events to read. Default = 20000", default_value=20000
    ).tag(config=True)

    progress_bar = traits.Bool(
        help="show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    aliases = {
        "input": "TimeCalibrationHDF5Writer.input",
        "i": "TimeCalibrationHDF5Writer.input",
        "glob": "TimeCalibrationHDF5Writer.glob",
        "output": "TimeCorrectionCalculate.calib_file_path",
        "o": "TimeCorrectionCalculate.calib_file_path",
        "pedestal": "LSTR0Corrections.drs4_pedestal_path",
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
        self.eventsource = None
        self.timeCorr = None
        self.path_list = None
        self.lst_r0 = None

    def setup(self):

        self.path_list = [str(self.input)]
        if self.input.is_dir():
            self.path_list = sorted(glob.glob(str(self.input/self.glob)))

        self.eventsource = self.add_component(
            LSTEventSource(
                input_url=self.path_list[0],
                max_events=self.max_events,
                parent=self,
            )
        )
        self.lst_r0 = self.add_component(
            LSTR0Corrections(parent=self)
        )
        self.timeCorr = TimeCorrectionCalculate(
            subarray=self.eventsource.subarray,
            config=self.config,
        )

    def start(self):

        for j, path in enumerate(self.path_list):
            self.eventsource.input_url = path
            self.log.info(f"File {j + 1} out of {len(self.path_list)}")
            self.log.info(f"Processing: {path}")
            for event in tqdm(
                    self.eventsource,
                    desc=self.eventsource.__class__.__name__,
                    total=self.eventsource.max_events,
                    unit="ev",
                    disable=not self.progress_bar,
            ):
                self.lst_r0.calibrate(event)
                # cut in signal to avoid cosmic events
                if event.r1.tel[self.timeCorr.tel_id].trigger_type == 4 or (
                    np.median(np.sum(event.r1.tel[self.timeCorr.tel_id].waveform[0], axis=1)) > 300
                ):
                    self.timeCorr.calibrate_peak_time(event)

    def finish(self):

        self.timeCorr.finalize()
        Provenance().add_output_file(
            self.timeCorr.calib_file_path,
            role='mon.tel.calibration'
        )


def main():
    exe = TimeCalibrationHDF5Writer()
    exe.run()


if __name__ == "__main__":
    main()
