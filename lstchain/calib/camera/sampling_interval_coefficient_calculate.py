import numpy as np
from astropy.io import fits
from numba import jit, prange

from ctapipe.core import Component


__all__ = ['SamplingIntervalCalculate']

N_PIXELS = 1855
N_CAPACITORS_CHANNEL = 1024
N_SAMPLES = 40

class SamplingIntervalCalculate(Component):
    """
        The SamplingIntervalCalculate class to create a sampling interval coefficient table for LST readout system using chip DRS4.
    """
    def __init__(self):
        self.peak_count = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL], dtype=np.uint16)
        self.fc_count = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL], dtype=np.uint16)

        self.peak_count_stack ={}
        self.fc_count_stack = {}

        self.sampling_interval_coefficient = {}
        self.charge_array_after_corr ={}
        self.charge_reso_array_after_corr ={}

        self.charge_reso_final = np.zeros(N_PIXELS)
        self.used_run = np.zeros(N_PIXELS, dtype=np.uint16)
        self.sampling_interval_coefficient_final = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL])

    def increment_peak_count(self, event, tel_id, gain, r0_r1_calibrator):

        waveform = event.r1.tel[tel_id].waveform
        
        first_capacitors = r0_r1_calibrator.first_cap
        r1_sample_start = r0_r1_calibrator.r1_sample_start.tel[tel_id]

        # Check pulse
        pulse_event_flag = np.sum(np.max(waveform[gain], axis=1) > 100) > 1800
    
        if pulse_event_flag:

            # find pulse peak position
            pulse_peak = np.argmax(waveform[gain], axis=1)
            fc = first_capacitors[tel_id][gain]
            pulse_peak_abs_pos = (fc + r1_sample_start + pulse_peak) % N_CAPACITORS_CHANNEL
            
            # increment peak count array
            self.peak_count[np.arange(N_PIXELS), pulse_peak_abs_pos] += 1
            self.fc_count[np.arange(N_PIXELS), fc % N_CAPACITORS_CHANNEL] += 1
            

    def stack_single_sampling_interval(self, file_list, gain):

        for single_file in file_list:

            run_id, gain_in_file, peak_count, fc_count = load_single_fits_sampling_interval(single_file)
            
            if gain_in_file == gain:
                if not run_id in self.peak_count_stack.keys():
                    self.peak_count_stack[run_id] = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL], dtype=np.uint16)
                    self.fc_count_stack[run_id] = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL], dtype=np.uint16)
                
                self.peak_count_stack[run_id] += peak_count
                self.fc_count_stack[run_id] += fc_count


    def convert_to_samp_interval_coefficient(self, gain):
        # convert peak counts to sampling interval coefficient

        for run_id in self.peak_count_stack.keys():
            self.sampling_interval_coefficient[run_id] = np.zeros([N_PIXELS, N_CAPACITORS_CHANNEL + N_SAMPLES])

            for pixel in range(N_PIXELS):
                self.sampling_interval_coefficient[run_id][pixel, :N_CAPACITORS_CHANNEL] = \
                    self.peak_count_stack[run_id][pixel] / np.sum(self.peak_count_stack[run_id][pixel]) * N_CAPACITORS_CHANNEL
                
                self.sampling_interval_coefficient[run_id][pixel, N_CAPACITORS_CHANNEL:] = \
                    self.sampling_interval_coefficient[run_id][pixel, :N_SAMPLES]

                
    def set_charge_array(self, gain):
        self.charge_array_before_corr = np.zeros([N_PIXELS, 60000])
        self.charge_reso_array_before_corr = np.zeros(N_PIXELS)

        for run_id in self.peak_count_stack.keys():
            self.charge_array_after_corr[run_id] = np.zeros([N_PIXELS, 60000])
            self.charge_reso_array_after_corr[run_id] = np.zeros(N_PIXELS)
            

    def calc_charge(self, count, event, tel_id, gain, r0_r1_calibrator):

        waveform = event.r1.tel[tel_id].waveform
        
        first_capacitors = r0_r1_calibrator.first_cap
        r1_sample_start = r0_r1_calibrator.r1_sample_start.tel[tel_id]
        r1_sample_end = r0_r1_calibrator.r1_sample_end.tel[tel_id]

        # Check pulse
        pulse_event_flag = np.sum(np.max(waveform[gain], axis=1) > 100) > 1800

        if pulse_event_flag:

            # find pulse peak position
            pulse_peak = np.argmax(waveform[gain], axis=1)
            fc = first_capacitors[tel_id][gain]
            pulse_peak_abs_pos = (fc + r1_sample_start + pulse_peak) % N_CAPACITORS_CHANNEL
            integ_start = pulse_peak - 2
            integ_last = integ_start + 5

            integ_abs_start = (pulse_peak_abs_pos -2 ) % N_CAPACITORS_CHANNEL
            integ_abs_last = integ_abs_start + 5

            for pixel in range(N_PIXELS):

                if integ_start[pixel] < 0 or integ_last[pixel] > (r1_sample_end - r1_sample_start):
                    continue

                self.charge_array_before_corr[pixel][count] = np.sum(waveform[gain, pixel,integ_start[pixel]:integ_last[pixel]])

                for run_id in self.peak_count_stack.keys():
                    samp_interval_coefficient = self.sampling_interval_coefficient[run_id]
                    
                    self.charge_array_after_corr[run_id][pixel][count] = \
                        np.sum(waveform[gain,pixel,integ_start[pixel]:integ_last[pixel]] \
                               * samp_interval_coefficient[pixel, integ_abs_start[pixel]:integ_abs_last[pixel]])

    def calc_charge_reso(self, gain):
        
        # Before correction
        for pixel in range(N_PIXELS):
            charge_array_before_corr_final = self.charge_array_before_corr[pixel]
            charge_array_before_corr_final = charge_array_before_corr_final[charge_array_before_corr_final!=0]
            self.charge_reso_array_before_corr[pixel] = np.std(charge_array_before_corr_final)/np.mean(charge_array_before_corr_final)

            for run_id in self.peak_count_stack.keys():
                charge_array_after_corr_final = self.charge_array_after_corr[run_id][pixel]   
                charge_array_after_corr_final = charge_array_after_corr_final[charge_array_after_corr_final!=0] 
                self.charge_reso_array_after_corr[run_id][pixel] = np.std(charge_array_after_corr_final)/np.mean(charge_array_after_corr_final)

    def verify(self):
        n_keys=len(self.charge_reso_array_after_corr.keys())
        run_id_array = np.zeros(n_keys, dtype=np.uint16)
        charge_reso_array_after_corr_all = np.zeros([n_keys, 1855])
        
        for i, ikey in enumerate(self.charge_reso_array_after_corr.keys()):
            run_id_array[i] = ikey
            charge_reso_array_after_corr_all[i] = self.charge_reso_array_after_corr[ikey]
    
        for ipix in range(N_PIXELS):
            min_charge_reso_arg = np.argmin(charge_reso_array_after_corr_all.T[ipix])
            self.charge_reso_final[ipix] = charge_reso_array_after_corr_all[min_charge_reso_arg, ipix]
            self.used_run[ipix] = run_id_array[min_charge_reso_arg]
            self.sampling_interval_coefficient_final[ipix] = self.sampling_interval_coefficient[self.used_run[ipix]][ipix, :N_CAPACITORS_CHANNEL]



def load_single_fits_sampling_interval(input_file):
        
    hdulist = fits.open(input_file)
        
    run_id = hdulist[0].header['run_id']
    gain = hdulist[0].header['gain']
    peak_count = hdulist[1].data
    fc_count = hdulist[2].data
    
    return run_id, gain, peak_count, fc_count

