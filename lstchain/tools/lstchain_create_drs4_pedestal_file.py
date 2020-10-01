"""
Create drs4 time pedestal fits file.
"""
import numpy as np
from astropy.io import fits

from ctapipe.core import Provenance, traits
from ctapipe.core import Tool
from ctapipe.io import EventSource
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.drs4 import DragonPedestal


class PedestalFITSWriter(Tool):

    name = "PedestalFITSWriter"
    description = "Generate a pedestal FITS file"

    output_file = traits.Path(
        help="Path to the generated FITS pedestal file",
        directory_ok=False,
        exist=False,
    ).tag(config=True)

    deltaT = traits.Bool(
        help="Flag to use deltaT correction. Default=True", default_value=True
    ).tag(config=True)

    aliases = {
        "input_file": "EventSource.input_url",
        "output_file": "PedestalFITSWriter.output_file",
    }

    flags = {
        "deltaT": ({"PedestalFITSWriter": {"deltaT": True}}, "Flag to use deltaT correction. Default is True")
    }

    classes = [EventSource, LSTR0Corrections, DragonPedestal]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Tool that generates a pedestal FITS file for low level calibration.

        For getting help run:
        python lstchain_create_drs4_pedestal_file.py --help
        """

        self.eventsource = None

    def setup(self):

        self.log.debug(f"Open file")
        self.eventsource = EventSource.from_config(parent=self)

    def start(self):

        try:
            pass
        except Exception as e:
            self.log.error(e)

    def finish(self):
        # Provenance().add_output_file(
        #     self.output_file,
        #     role='mon.tel.pedestal'
        # )
        pass


def main():
    exe = PedestalFITSWriter()
    exe.run()


if __name__ == "__main__":
    main()
