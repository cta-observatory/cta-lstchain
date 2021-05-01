import numpy as np
from astropy.io import fits
import h5py
from tqdm import tqdm
import os

from ctapipe.core import Provenance, traits
from ctapipe.core import Tool
from ctapipe.image.extractor import ImageExtractor

from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.calibration import LSTR0Corrections
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS, N_CAPACITORS_CHANNEL

from lstchain.calib.camera.sampling_interval_coefficient_calculate import SamplingIntervalCoefficientCalculate
from lstchain.paths import run_info_from_filename

class SamplingIntervalCoefficientHDFWriter(Tool):
    """
    Tool that generates a sampling interval coefficient hdf file.
    For getting help run:
    lstchain_create_sampling_interval_coefficient_file --help
    """

    name = "SamplingIntervalCoefficientHDFWriter"
    description = "Generate a sampling interval coefficient table"

    input_peak_count_fits = traits.Path(
        help="Path to the generated peak count fits files by lstchain_create_peak_count_file_for_sampling_interval_calib",
        directory_ok=True,
    ).tag(config=True)

    output = traits.Path(
        help="Path to the output fits file",
        directory_ok=False,
    ).tag(config=True)

    glob = traits.Unicode(
        help="Filename pattern to glob files in the directory",
        default_value="*.fits"
    ).tag(config=True)

    gain = traits.Int(
        help="Select gain channel (0: High Gain, 1: Low Gain)",
    ).tag(config=True)

    progress_bar = traits.Bool(
        help="show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("o", "output"): "SamplingIntervalCoefficientHDFWriter.output",
        ("g", "gain"): "SamplingIntervalCoefficientHDFWriter.gain",
        "input_peak_count_fits": "SamplingIntervalCoefficientHDFWriter.input_peak_count_fits",
        "pedestal": "LSTR0Corrections.drs4_pedestal_path",        
        "max-events": "EventSource.max_events",
        "r1_sample_start": "LSTR0Corrections.r1_sample_start",
        "r1_sample_end": "LSTR0Corrections.r1_sample_end",
        "progress_bar": "SamplingIntervalCoefficientHDFWriter.progress_bar",
    }

    classes = [LSTEventSource, LSTR0Corrections, SamplingIntervalCoefficientCalculate]

    def setup(self):
        
        self.sampling_interval_coefficient_calculate = SamplingIntervalCoefficientCalculate(self.gain)
        self.eventsource = LSTEventSource(parent=self)
        self.image_extractor = ImageExtractor.from_name('LocalPeakWindowSum', subarray = self.eventsource.subarray)

        self.path_list = [str(self.input_peak_count_fits)]

        if self.input_peak_count_fits.is_dir():
            self.path_list = sorted(self.input_peak_count_fits.glob(self.glob))

    def start(self):
        
        self.sampling_interval_coefficient_calculate.stack_peak_count_fits(self.path_list)
        self.sampling_interval_coefficient_calculate.convert_to_samp_interval_coefficient()
        self.sampling_interval_coefficient_calculate.set_charge_array()

        for i, event in enumerate(tqdm(
                self.eventsource,
                desc = self.eventsource.__class__.__name__,
                total = self.eventsource.max_events,
                unit = "event",
                disable = not self.progress_bar,
        )):

            if event.index.event_id % 500 == 0:
                self.log.debug(f'event id = {event.index.event_id}')

            # skip the first 1000 events which would not be calibrated correctly
            if i < 1000:
                continue

            self.sampling_interval_coefficient_calculate.calc_charge(event, tel_id = self.eventsource.tel_id,
                        r0_r1_calibrator = self.eventsource.r0_r1_calibrator, extractor = self.image_extractor)

        self.sampling_interval_coefficient_calculate.calc_charge_reso()
        self.sampling_interval_coefficient_calculate.verify()
        
    def finish(self):
        
        CHANNEL = ["HG", "LG"]

        output_abs_path = os.path.abspath(self.output)
        output_log_pdf_file = os.path.join(os.path.dirname(output_abs_path), "sampling_interval_coefficient_results_{}.pdf".format(CHANNEL[self.gain]))

        self.sampling_interval_coefficient_calculate.plot_results(self.eventsource.input_url, output_log_pdf_file)

        with h5py.File(self.output, 'a') as hf:

            if not 'sampling_interval_coefficient' in hf:
                hf.create_dataset('sampling_interval_coefficient', data = np.zeros([N_GAINS, N_PIXELS, N_CAPACITORS_CHANNEL]))
            hf['sampling_interval_coefficient'][self.gain] = self.sampling_interval_coefficient_calculate.sampling_interval_coefficient_final

            if not 'used_run' in hf:
                hf.create_dataset('used_run', data = np.zeros([N_GAINS, N_PIXELS]), dtype=np.uint16)
            hf['used_run'][self.gain] = self.sampling_interval_coefficient_calculate.used_run

            if not 'charge_resolution' in hf:
                hf.create_dataset('charge_resolution', data = np.zeros([N_GAINS, N_PIXELS]))
            hf['charge_resolution'][self.gain] = self.sampling_interval_coefficient_calculate.charge_reso_final

def main():
    exe = SamplingIntervalCoefficientHDFWriter()
    exe.run()


if __name__ == "__main__":
    main()
