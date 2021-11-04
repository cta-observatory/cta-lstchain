"""
Create drs4 pedestal fits file.
"""
import numpy as np
from astropy.io import fits
from tqdm.autonotebook import tqdm

from ctapipe.core import Provenance, Tool, traits
from ctapipe_io_lst import LSTEventSource
from lstchain.calib.camera.drs4 import DragonPedestal
from traitlets.config import Config


class PedestalFITSWriter(Tool):
    """
    Tool that generates a pedestal FITS file for low level calibration.

    For getting help run:
    lstchain_create_drs4_pedestal_file --help
    """

    name = "PedestalFITSWriter"
    description = "Generate a pedestal FITS file"

    input = traits.Path(
        help="Path to fitz.fz file to create pedestal file",
        directory_ok=False,
        exists=True,
    ).tag(config=True)

    output = traits.Path(
        default_value="pedestal.fits",
        help="Path to the generated fits pedestal file",
        directory_ok=False,
    ).tag(config=True)

    max_events = traits.Int(
        default_value=20000,
        help="Maximum numbers of events to read",
    ).tag(config=True)

    deltaT = traits.Bool(
        help="Use delta T correction", default_value=True
    ).tag(config=True)

    overwrite = traits.Bool(
        help="Overwrite output file", default_value=True
    ).tag(config=True)

    progress_bar = traits.Bool(
        help="Show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "PedestalFITSWriter.input",
        ("o", "output"): "PedestalFITSWriter.output",
        "start-r0-waveform": "DragonPedestal.r0_sample_start",
    }
    flags = {
        "no-delta-t": (
            {"PedestalFITSWriter": {"deltaT": False}},
            "Switch off delta T corrections",
        ),
        "no-overwrite": (
            {"PedestalFITSWriter": {"overwrite": False}},
            "Do not overwrite output file",
        ),
    }

    classes = [LSTEventSource, DragonPedestal]

    def setup(self):

        source_config = {
            "LSTEventSource": {
                "max_events": self.max_events,
            },
            "LSTR0Corrections": {
                "apply_drs4_pedestal_correction": False,
                "apply_timelapse_correction": self.deltaT,
                "apply_spike_correction": False,
            }
        }

        self.eventsource = LSTEventSource(input_url=self.input, config=Config(source_config))
        self.pixel_ids = self.eventsource.camera_config.expected_pixels_id
        self.pedestal = DragonPedestal(
            tel_id=self.eventsource.tel_id,
            n_module=self.eventsource.camera_config.lstcam.num_modules,
            parent=self
        )

    def start(self):

        for event in tqdm(
            self.eventsource,
            desc=self.eventsource.__class__.__name__,
            total=len(self.eventsource.multi_file),
            unit="ev",
            disable=not self.progress_bar,
        ):
            self.pedestal.fill_pedestal_event(event)

    def finish(self):

        self.pedestal.complete_pedestal()

        expected_pixel_id = fits.PrimaryHDU(self.pixel_ids)
        pedestal_array = fits.ImageHDU(
            self.pedestal.meanped.astype(np.int16),
            name="pedestal array")
        failing_pixels_col = fits.Column(
            name="failing pixels",
            array=self.pedestal.failing_pixels_array,
            format="K"
        )
        failing_pixels = fits.BinTableHDU.from_columns(
            [failing_pixels_col],
            name="failing pixels"
        )
        hdulist = fits.HDUList([expected_pixel_id, pedestal_array, failing_pixels])
        hdulist.writeto(self.output, overwrite=self.overwrite)

        Provenance().add_output_file(self.output, role="mon.tel.pedestal")


def main():
    exe = PedestalFITSWriter()
    exe.run()


if __name__ == "__main__":
    main()
