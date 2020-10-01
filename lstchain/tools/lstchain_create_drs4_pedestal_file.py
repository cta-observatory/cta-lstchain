"""
Create drs4 time pedestal fits file.
"""
import numpy as np
from astropy.io import fits

from ctapipe.core import Provenance, traits
from ctapipe.core import Tool
from ctapipe.io import EventSeeker, EventSource
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
        self.lst_r0 = None
        self.pedestal = None

    def setup(self):

        self.log.debug(f"Open file")
        self.eventsource = EventSource.from_config(parent=self)
        seeker = EventSeeker(self.eventsource)
        ev = seeker[0]
        tel_id = ev.r0.tels_with_data[0]
        n_modules = ev.lst.tel[tel_id].svc.num_modules
        self.lst_r0 = LSTR0Corrections(config=self.config)
        self.pedestal = DragonPedestal(tel_id=tel_id, n_module=n_modules)

    def start(self):

        try:
            if self.deltaT:
                self.log.info("DeltaT correction active")
                for i, event in enumerate(self.eventsource):
                    for tel_id in event.r0.tels_with_data:
                        self.lst_r0.time_lapse_corr(event, tel_id)
                        self.pedestal.fill_pedestal_event(event)
                        if i % 500 == 0:
                            self.log.debug("i = {}, ev id = {}".format(i, event.index.event_id))
            else:
                self.log.info("DeltaT correction no active")
                for i, event in enumerate(self.eventsource):
                    self.pedestal.fill_pedestal_event(event)
                    if i % 500 == 0:
                        self.log.debug("i = {}, ev id = {}".format(i, event.index.event_id))

            # Finalize pedestal and write to fits file
            self.pedestal.finalize_pedestal()

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
