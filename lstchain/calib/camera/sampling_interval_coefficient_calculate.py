import numpy as np
from astropy.io import fits
from numba import jit, prange
import logging

from ctapipe.core import Component
from ctapipe_io_lst.constants import (
    N_PIXELS, N_SAMPLES, N_CAPACITORS_CHANNEL
)
import matplotlib.pyplot as plt
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

logger = logging.getLogger(__name__)

THRESHOLD_PULSE_ADC = 100
THRESHOLD_PULS_PIX_NUM = 1800
CHANNEL = ["HG", "LG"]
CRITERIA = [0.07, 0.05]


__all__ = ['SamplingIntervalCoefficientPeakCount', 'SamplingIntervalCoefficientCalculate']


class SamplingIntervalCoefficientPeakCount(Component):
    """
        The SamplingIntervalCoefficientPeakCount class to create a peak count table for DRS4 sampling interval calibration.
    """
    def __init__(self, gain):

        self.peak_count = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL], dtype=np.uint16)
        self.fc_count = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL], dtype=np.uint16)
        self.gain = gain

    def increment_peak_count(self, event, tel_id, r0_r1_calibrator):

        waveform = event.r1.tel[tel_id].waveform
        
        first_capacitors = r0_r1_calibrator.first_cap
        r1_sample_start = r0_r1_calibrator.r1_sample_start.tel[tel_id]

        # There are some events without test pulses for almost all modules.
        # In those events, a few modules have pulse signals of which pulse shapes are not expected...
        # For the moment, such events are just removed in the anlysis
        pulse_event_flag = np.sum(np.max(waveform[self.gain], axis=1) > THRESHOLD_PULSE_ADC) > THRESHOLD_PULS_PIX_NUM

        if pulse_event_flag :
            # find pulse peak position
            pulse_flag = np.max(waveform[self.gain], axis=1) > THRESHOLD_PULSE_ADC
            pulse_peak = np.argmax(waveform[self.gain], axis=1)
            fc = first_capacitors[tel_id][self.gain]
            pulse_peak_abs_pos = (fc + r1_sample_start + pulse_peak) % N_CAPACITORS_CHANNEL
            
            # increment peak count array in case that test pulses exist in the readout window
            self.peak_count[np.arange(N_PIXELS), pulse_peak_abs_pos] += pulse_flag
            self.fc_count[np.arange(N_PIXELS), fc % N_CAPACITORS_CHANNEL] += pulse_flag


class SamplingIntervalCoefficientCalculate(Component):
    """
        The SamplingIntervalCoefficientCalculate class to create a sampling interval coefficient table for LST readout system using chip DRS4.
    """
    def __init__(self, gain):

        self.peak_count_stack ={}
        self.fc_count_stack = {}
        self.sampling_interval_coefficient = {}
        self.gain = gain
            
    def stack_peak_count_fits(self, file_list):
        # stack peak count fits files, and sort stacked peak counts by gain, run_id

        for single_file in file_list:

            run_id, gain_in_file, peak_count, fc_count = load_peak_count_fits_file(single_file)

            # select only a given gain
            if gain_in_file == self.gain:
                if not run_id in self.peak_count_stack.keys():
                    self.peak_count_stack[run_id] = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL], dtype=np.uint16)
                    self.fc_count_stack[run_id] = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL], dtype=np.uint16)
                
                self.peak_count_stack[run_id] += peak_count
                self.fc_count_stack[run_id] += fc_count

    def convert_to_samp_interval_coefficient(self):
        # convert peak counts to sampling interval coefficient

        for run_id in self.peak_count_stack.keys():
            self.sampling_interval_coefficient[run_id] = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL + N_SAMPLES])

            stack_events_num = np.sum(self.peak_count_stack[run_id], axis=1)
            stack_events_num = stack_events_num.reshape(1, N_PIXELS).T

            self.sampling_interval_coefficient[run_id][:,:N_CAPACITORS_CHANNEL] = self.peak_count_stack[run_id] / stack_events_num * N_CAPACITORS_CHANNEL
            self.sampling_interval_coefficient[run_id][:,N_CAPACITORS_CHANNEL:] = self.sampling_interval_coefficient[run_id][:, :N_SAMPLES]
                
    def set_charge_array(self):
        # set charge array to compute charge resolutions using calculated coefficients with different observation runs  

        # before calibration
        self.charge_array_before_corr = []

        # after calibration
        self.charge_array_after_corr ={}

        for run_id in self.peak_count_stack.keys():
            self.charge_array_after_corr[run_id] = []
            
    def calc_charge(self, event, tel_id, r0_r1_calibrator, extractor):

        waveform = event.r1.tel[tel_id].waveform
        
        # There are some events without test pulses for almost all modules.
        # In those events, a few modules have pulse signals of which pulse shapes are not expected...
        # For the moment, such events are just removed in the anlysis
        pulse_event_flag = np.sum(np.max(waveform[self.gain], axis=1) > THRESHOLD_PULSE_ADC) > THRESHOLD_PULS_PIX_NUM

        if pulse_event_flag:

            first_capacitors = r0_r1_calibrator.first_cap
            r1_sample_start = r0_r1_calibrator.r1_sample_start.tel[tel_id]
            r1_sample_end = r0_r1_calibrator.r1_sample_end.tel[tel_id]

            # find pulse peak position
            pulse_peak = np.argmax(waveform[self.gain], axis=1)
            fc = first_capacitors[tel_id][self.gain]

            # calculate charge (before calibration)
            charge, peak_time = extractor(waveform[self.gain], tel_id, np.full(N_PIXELS, self.gain, dtype=int))
            self.charge_array_before_corr.append(charge.tolist())

            # calculate charge (after calibration with sampling interval coefficients obtained by each observation run)
            for run_id in self.peak_count_stack.keys():
                sampling_interval_coefficient_event = np.zeros([N_PIXELS, r1_sample_end - r1_sample_start])

                for pixel in range(N_PIXELS):
                    sampling_interval_coefficient_event[pixel] = (
                        self.sampling_interval_coefficient[run_id][pixel][fc[pixel]%N_CAPACITORS_CHANNEL + r1_sample_start:
                                                                          fc[pixel]%N_CAPACITORS_CHANNEL + r1_sample_end]
                    )

                charge, peak_time = extractor(waveform[self.gain] * sampling_interval_coefficient_event, tel_id, np.full(N_PIXELS, self.gain, dtype=int))

                self.charge_array_after_corr[run_id].append(charge.tolist())

    def calc_charge_reso(self):
        # calculate charge resolution to evaluate the sampling interval correction

        # before correction
        self.charge_reso_array_before_corr = (
            np.std(np.array(self.charge_array_before_corr).T, axis=1)/np.mean(np.array(self.charge_array_before_corr).T, axis=1)
        )

        # after correction
        self.charge_reso_array_after_corr ={}

        for run_id in self.peak_count_stack.keys():
            self.charge_reso_array_after_corr[run_id] = (
                np.std(np.array(self.charge_array_after_corr[run_id]).T, axis=1)/np.mean(np.array(self.charge_array_after_corr[run_id]).T, axis=1)
            )

    def verify(self):
        # verify the obtained sampling interval coefficients
        # Coefficients which give the best charge resolution are adopted as the final ones
        # If the charge resolution is worse than required criteria (<7% for high gain, <5% for low gain), coefficients are filled with ones.

        self.charge_reso_final = np.zeros(N_PIXELS)
        self.used_run = np.zeros(N_PIXELS, dtype=np.uint16)
        self.sampling_interval_coefficient_final = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL])

        n_runs = len(self.charge_reso_array_after_corr.keys())
        run_id_array = np.zeros(n_runs, dtype=np.uint16)
        charge_reso_array_after_corr_all = np.zeros([n_runs, N_PIXELS])
        
        for i, ikey in enumerate(self.charge_reso_array_after_corr.keys()):
            run_id_array[i] = ikey
            charge_reso_array_after_corr_all[i] = self.charge_reso_array_after_corr[ikey]
    
        for pixel in range(N_PIXELS):
            min_charge_reso_arg = np.argmin(charge_reso_array_after_corr_all.T[pixel])

            self.charge_reso_final[pixel] = charge_reso_array_after_corr_all[min_charge_reso_arg, pixel]                
            self.used_run[pixel] = run_id_array[min_charge_reso_arg]

            criteria_flag = self.charge_reso_final[pixel] < CRITERIA[self.gain]

            if criteria_flag:
                self.sampling_interval_coefficient_final[pixel] = self.sampling_interval_coefficient[self.used_run[pixel]][pixel, :N_CAPACITORS_CHANNEL]
            else:
                logger.info(f'[charge resolution] {CHANNEL[self.gain]} pixel {pixel}: {self.charge_reso_final[pixel]} > {CRITERIA[self.gain]}')
                self.sampling_interval_coefficient_final[pixel] = np.ones(N_CAPACITORS_CHANNEL)

    def plot_results(self, verify_data_path, output_file):

        plt.rcParams['font.size']=15
        camera = CameraGeometry.from_name('LSTCam-002')

        num_run = len(self.charge_reso_array_after_corr.keys())

        plt.figure(figsize=(20, 6*(num_run+1)))
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.suptitle("{} (data for the verification: {})".format(CHANNEL[self.gain], verify_data_path))

        for i, run_id in enumerate(self.charge_reso_array_after_corr.keys()):
    
            # charge resolution scatter plot (before correction vs after correction)
            plt.subplot(num_run + 1, 3, i*3+1)
            plt.scatter(self.charge_reso_array_before_corr, self.charge_reso_array_after_corr[run_id])
            plt.plot([0, 0.15], [0, 0.15], color='black', ls='--')
            plt.hlines(CRITERIA[self.gain], 0, 0.15, color='red', ls='--')
            plt.xlim(0,.15)
            plt.ylim(0,.15)
            plt.title('charge resolution (run {})'.format(run_id))
            plt.xlabel('before calibration')
            plt.ylabel('after calibration')
            plt.grid(color='gray', alpha=0.5)
            
            # charge resolution histogram
            plt.subplot(num_run + 1, 3, i*3+2)
            plt.hist(self.charge_reso_array_after_corr[run_id], range=[0, 0.1], bins=50)
            plt.vlines(CRITERIA[self.gain], 0.05, 500, color='red', ls='--')
            plt.title('charge resolution (after calib.) (run {})'.format(run_id))
            plt.xlabel('charge resolution')
            plt.yscale('log')
            plt.grid(color='gray', alpha=0.5)

            # camera map
            plt.subplot(num_run + 1, 3, i*3+3)
            disp = CameraDisplay(camera)
            disp.image = self.charge_reso_array_after_corr[run_id]
            
            plt.grid(color='gray', alpha=0.5)
            disp.set_limits_minmax(0, CRITERIA[self.gain])
            disp.highlight_pixels(self.charge_reso_array_after_corr[run_id]>CRITERIA[self.gain], color='red')
            disp.add_colorbar()
            
        # Final result
        plt.subplot(num_run + 1, 3, num_run*3+1)
        plt.scatter(self.charge_reso_array_before_corr, self.charge_reso_final)
        plt.plot([0, 0.15], [0, 0.15], color='black', ls='--')
        plt.hlines(CRITERIA[self.gain], 0, 0.15, color='red', ls='--')
        plt.xlim(0,.15)
        plt.ylim(0,.15)
        plt.title('charge resolution (Final)'.format(run_id))
        plt.xlabel('before calibration')
        plt.ylabel('after calibration')
        plt.grid(color='gray', alpha=0.5)

        plt.subplot(num_run + 1, 3, num_run*3+2)
        plt.hist(self.charge_reso_final, range=[0, 0.1], bins=50)
        plt.vlines(CRITERIA[self.gain], 0.05, 500, color='red', ls='--')
        plt.title('charge resolution (after calib.) (Final)'.format(run_id))
        plt.xlabel('charge resolution')
        plt.yscale('log')
        plt.grid(color='gray', alpha=0.5)
        
        plt.subplot(num_run + 1, 3, num_run*3+3)
        disp = CameraDisplay(camera)
        disp.image = self.charge_reso_final
        
        plt.grid(color='gray', alpha=0.5)
        disp.set_limits_minmax(0, CRITERIA[self.gain])
        disp.highlight_pixels(self.charge_reso_final>CRITERIA[self.gain], color='red')
        disp.add_colorbar()

        for i, bad_pixel in enumerate(np.where(self.charge_reso_final>CRITERIA[self.gain])[0]):
            plt.text(-1.2, 1.5-0.2*i, 'pix{}: {:.3f}'.format(bad_pixel, self.charge_reso_final[bad_pixel]), color='red')
 
            
        plt.savefig(output_file)

def load_peak_count_fits_file(input_file):
        
    hdulist = fits.open(input_file)
        
    run_id = hdulist[0].header['run_id']
    gain = hdulist[0].header['gain']
    peak_count = hdulist[1].data
    fc_count = hdulist[2].data
    
    return run_id, gain, peak_count, fc_count
