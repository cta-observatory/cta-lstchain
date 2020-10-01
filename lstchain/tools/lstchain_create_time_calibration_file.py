"""
Create drs4 time correction coefficients.
"""
from ctapipe.core import Provenance, traits
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSource
from lstchain.calib.camera.r0 import LSTR0Corrections


class TimeCalibrationHDF5Writer(Tool):

    name = "TimeCalibrationHDF5Writer"
    description = "Generate a HDF5 file with time calibration coefficients"

    output_file = traits.Path(
        help="Path to the generated the generated HDF5 time calibration file",
        directory_ok=False,
        exists=False,
    ).tag(config=True)

    aliases = {
        "input_file": "EventSource.input_url",
        "output_file": "TimeCalibrationHDF5Writer.output_file",
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
        self.writer = None

    def setup(self):

        self.log.debug(f"Open file")

    def start(self):

        try:
            self.log.debug(f"Start loop")
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
