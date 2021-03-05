"""
Create drs4 time correction coefficients.
"""
import glob

from ctapipe.core import Provenance, Tool, traits
from ctapipe.io import EventSource
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

    progress_bar = traits.Bool(
        help="show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "TimeCalibrationHDF5Writer.input",
        ("o", "output"): "TimeCorrectionCalculate.calib_file_path",
        "glob": "TimeCalibrationHDF5Writer.glob",
        "max-events": "EventSource.max_events",
        "pedestal": "LSTR0Corrections.drs4_pedestal_path",
        "dragon-reference-time": "EventTimeCalculator.dragon_reference_time",
        "dragon-reference-counter": "EventTimeCalculator.dragon_reference_counter",
    }

    classes = [TimeCorrectionCalculate] + traits.classes_with_traits(EventSource)

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        """
        Tool that generates a HDF5 file with time calibration coefficients.

        For getting help run:
        lstchain_create_time_calibration_file --help
        """
        self.eventsource = None
        self.path_list = None
        self.timeCorr = None

    def setup(self):

        self.path_list = [str(self.input)]
        if self.input.is_dir():
            self.path_list = sorted(glob.glob(str(self.input/self.glob)))

    def start(self):

        for j, path in enumerate(self.path_list):
            self.eventsource = EventSource(input_url=path, config=self.config)
            self.log.info(f"File {j + 1} out of {len(self.path_list)}")
            self.log.info(f"Processing: {path}")

            self.timeCorr = TimeCorrectionCalculate(
                subarray=self.eventsource.subarray,
                config=self.config,
            )

            for event in tqdm(
                    self.eventsource,
                    desc=self.eventsource.__class__.__name__,
                    total=self.eventsource.max_events,
                    unit="ev",
                    disable=not self.progress_bar,
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
