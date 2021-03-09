"""
Create drs4 time pedestal fits file.
"""
import numpy as np
from astropy.io import fits
from tqdm.autonotebook import tqdm

from ctapipe.core import Provenance, traits
from ctapipe.core import Tool
from ctapipe.io import EventSource
from lstchain.calib.camera.drs4 import DragonPedestal


class PedestalFITSWriter(Tool):
    """
    Tool that generates a pedestal FITS file for low level calibration.

    For getting help run:
    lstchain_create_drs4_pedestal_file --help
    """

    name = "PedestalFITSWriter"
    description = "Generate a pedestal FITS file"

    output = traits.Path(
        default_value="pedestal.fits",
        help="Path to the generated fits pedestal file",
        directory_ok=False,
    ).tag(config=True)

    deltaT = traits.Bool(
        help="Use delta T correction", default_value=False
    ).tag(config=True)

    progress_bar = traits.Bool(
        help="show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("o", "output"): "PedestalFITSWriter.output",
        "max-events": "EventSource.max_events",
        "start-r0-waveform": "DragonPedestal.r0_sample_start",
    }
    flags = {
        "deltaT": (
            {"PedestalFITSWriter": {"deltaT": True}},
            "Activate delta T corrections",
        )
    }

    classes = [EventSource, DragonPedestal]

    def setup(self):

        self.log.debug("Opening file")
        self.eventsource = EventSource(parent=self)
        event = next(iter(self.eventsource))
        tel_id = event.trigger.tels_with_trigger[0]
        self.pixel_ids = event.lst.tel[tel_id].svc.pixel_ids
        self.pedestal = DragonPedestal(tel_id=tel_id, n_module=event.lst.tel[tel_id].svc.num_modules, parent=self)

    def start(self):

        for event in tqdm(
            self.eventsource,
            desc=self.eventsource.__class__.__name__,
            total=len(self.eventsource.multi_file),
            unit="ev",
            disable=not self.progress_bar,
        ):
            if self.deltaT:
                for tel_id in event.r0.tels_with_data:
                    self.eventsource.r0_r1_calibrator.update_first_capacitors(event)
                    self.eventsource.r0_r1_calibrator.time_lapse_corr(event, tel_id)
            self.pedestal.fill_pedestal_event(event)

    def finish(self):

        self.pedestal.complete_pedestal()

        expected_pixel_id = fits.PrimaryHDU(self.pixel_ids)
        pedestal_array = fits.ImageHDU(np.int16(self.pedestal.meanped), name="pedestal array")
        failing_pixels_col = fits.Column(name='failing pixels', array=self.pedestal.failing_pixels_array, format='K')
        failing_pixels = fits.BinTableHDU.from_columns([failing_pixels_col], name="failing pixels")
        hdulist = fits.HDUList([expected_pixel_id, pedestal_array, failing_pixels])
        hdulist.writeto(self.output)

        Provenance().add_output_file(self.output, role="mon.tel.pedestal")


def main():
    exe = PedestalFITSWriter()
    exe.run()


if __name__ == "__main__":
    main()
