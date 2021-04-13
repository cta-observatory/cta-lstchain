import numpy as np
from astropy.io import fits
import h5py

from ctapipe.core import Provenance, traits
from ctapipe.core import Tool
from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.calibration import LSTR0Corrections
from lstchain.calib.camera.sampling_interval_coefficient_calculate import SamplingIntervalCalculate
from lstchain.paths import run_info_from_filename

class SamplingIntervalCoefficientHDFWriter(Tool):
    """
    Tool that generates a sampling interval coefficient hdf file.
    For getting help run:
    lstchain_create_sampling_interval_coefficient_file --help
    """

    name = "SamplingIntervalCoefficientHDFWriter"
    description = "Generate a sampling interval coefficient table"

    input_fits = traits.Path(
        help="Path to the generated peak count fits files",
        directory_ok=True,
    ).tag(config=True)

    input_fits_hg = traits.Path(
        help="Path to the generated fits sampling interval coefficient file for high gain",
        directory_ok=False,
    ).tag(config=True)

    input_fits_lg = traits.Path(
        help="Path to the generated fits sampling interval coefficient file for low gain",
        directory_ok=False,
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

    count_flag = traits.Bool(
        default_value=False,
        help="First step: count peak on each capacitor"
    ).tag(config=True)

    stack_verify_flag = traits.Bool(
        default_value=False,
        help="Second step: Stack peak count files and verify with charge resolution"
    ).tag(config=True)

    merge_flag = traits.Bool(
        default_value=False,
        help="Third step: Merge sampling coefficient fits files for both gains."
    ).tag(config=True)

    progress_bar = traits.Bool(
        help="show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("o", "output"): "SamplingIntervalCoefficientHDFWriter.output",
        ("g", "gain"): "SamplingIntervalCoefficientHDFWriter.gain",
        "input_fits": "SamplingIntervalCoefficientHDFWriter.input_fits",
        "input_fits_hg": "SamplingIntervalCoefficientHDFWriter.input_fits_hg",
        "input_fits_lg": "SamplingIntervalCoefficientHDFWriter.input_fits_lg",
        "pedestal": "LSTR0Corrections.drs4_pedestal_path",        
        "max-events": "EventSource.max_events",
        "r1_sample_start": "LSTR0Corrections.r1_sample_start",
        "r1_sample_end": "LSTR0Corrections.r1_sample_end",
    }

    classes = [LSTEventSource, LSTR0Corrections, SamplingIntervalCalculate]

    flags = {
        'count_flag': (
            {'SamplingIntervalCoefficientHDFWriter': {'count_flag': True}},
            'First step: count peak on each capacitor',
        ),
        'stack_verify_flag': (
            {'SamplingIntervalCoefficientHDFWriter': {'stack_verify_flag': True}},
            'Second step: merge peak count files and verify with charge resolution',
        ),
        'merge_flag': (
            {'SamplingIntervalCoefficientHDFWriter': {'merge_flag': True}},
            'Third step: Merge sampling coefficient fits files for both gains',
        )
    }

    def setup(self):
        
        self.sampling_interval_calculate = SamplingIntervalCalculate()

        if self.count_flag:
            self.setup_count()
    
        if self.stack_verify_flag:
            self.setup_stack_verify()

        if self.merge_flag:
            self.setup_merge()


    def setup_count(self):
        self.log.debug('First step: count peak on each capacitor')
        self.eventsource = LSTEventSource(parent = self)
        self.run_id = self.eventsource.obs_ids[0]

    def setup_stack_verify(self):
        self.log.debug('Second step: stack and verify with charge resolution')
        self.path_list = [str(self.input_fits)]
        if self.input_fits.is_dir():
            self.path_list = sorted(self.input_fits.glob(self.glob))
        self.eventsource = LSTEventSource(parent=self)

    def setup_merge(self):
        self.log.debug('Third step: Merge sampling coefficients')



    def start(self):
        
        if self.count_flag:
            self.start_count()

        if self.stack_verify_flag:
            self.start_stack_verify()

        if self.merge_flag:
            self.start_merge()

    def start_count(self):
        self.log.debug('start peak counting')

        for i, event in enumerate(self.eventsource):
            if event.index.event_id % 500 == 0:
                self.log.debug(f'event id = {event.index.event_id}')

            # skip the first 1000 events which would not be calibrated correctly
            if i < 1000:
                continue

            #first_capacitor = self.eventsource.r0_r1_calibrator.first_cap
            self.sampling_interval_calculate.increment_peak_count(event, tel_id = self.eventsource.tel_id, 
                                                                  gain = self.gain, r0_r1_calibrator = self.eventsource.r0_r1_calibrator)

    def start_stack_verify(self):
        self.log.debug('stack peak count tables')
        self.sampling_interval_calculate.stack_single_sampling_interval(self.path_list, self.gain)

        self.log.debug('convert peak counts into sampling interval coefficients') 
        self.sampling_interval_calculate.convert_to_samp_interval_coefficient(gain = self.gain)
        self.sampling_interval_calculate.set_charge_array(gain = self.gain, self.eventsource.max_events)

        for i, event in enumerate(self.eventsource):
            if event.index.event_id % 500 == 0:
                self.log.debug(f'event id = {event.index.event_id}')

            # skip the first 1000 events which would not be calibrated correctly
            if i < 1000:
                continue

            self.sampling_interval_calculate.calc_charge(i, event, tel_id = self.eventsource.tel_id, 
                                                         gain = self.gain, r0_r1_calibrator = self.eventsource.r0_r1_calibrator)

        self.log.debug('calculate charge resolution using the sampling coefficients')
        self.sampling_interval_calculate.calc_charge_reso(gain = self.gain)
        self.sampling_interval_calculate.verify()
        

    def start_merge(self):
        # High Gain
        hdulist = fits.open(self.input_fits_hg)
        hdu = hdulist[0]
        self.sampling_interval_coefficient_hg = hdulist[1].data
        self.used_run_hg = hdulist[2].data
        self.charge_reso_hg = hdulist[3].data

        # Low Gain
        hdulist = fits.open(self.input_fits_lg)
        hdu = hdulist[0]
        self.sampling_interval_coefficient_lg = hdulist[1].data
        self.used_run_lg = hdulist[2].data
        self.charge_reso_lg = hdulist[3].data
        
        self.sampling_interval_coefficient_merge = np.array([self.sampling_interval_coefficient_hg, self.sampling_interval_coefficient_lg])
        self.used_run_merge = np.array([self.used_run_hg, self.used_run_lg])
        self.charge_reso_merge = np.array([self.charge_reso_hg, self.charge_reso_lg])
        

    def finish(self):
        
        if self.count_flag:
            self.finish_count()
        
        if self.stack_verify_flag:
            self.finish_stack_verify()

        if self.merge_flag:
            self.finish_merge()
            
    def finish_count(self):
        hdu_peak = fits.ImageHDU(self.sampling_interval_calculate.peak_count)
        hdu_fc = fits.ImageHDU(self.sampling_interval_calculate.fc_count)
        hdr = fits.Header()
        hdr['run_id'] = self.run_id
        hdr['gain'] = self.gain
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([primary_hdu, hdu_peak, hdu_fc])
        hdul.writeto(self.output)

    def finish_stack_verify(self):
        hdu_sampling_interval_coefficient = fits.ImageHDU(self.sampling_interval_calculate.sampling_interval_coefficient_final)
        hdu_used_run = fits.ImageHDU(self.sampling_interval_calculate.used_run)
        hdu_charge_reso_final = fits.ImageHDU(self.sampling_interval_calculate.charge_reso_final)

        hdr = fits.Header()
        hdr['gain'] = self.gain
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([primary_hdu, hdu_sampling_interval_coefficient, hdu_used_run, hdu_charge_reso_final])
        hdul.writeto(self.output)

    def finish_merge(self):

        with h5py.File(self.output, 'w') as hf:
            hf.create_dataset('sampling_interval_coefficient', data = self.sampling_interval_coefficient_merge)
            hf.create_dataset('used_run', data = self.used_run_merge)
            hf.create_dataset('charge_reso', data = self.charge_reso_merge)

def main():
    exe = SamplingIntervalCoefficientHDFWriter()
    exe.run()


if __name__ == "__main__":
    main()
