import h5py
import numpy as np
from numba import njit, prange

from ctapipe.core.traits import Int
from ctapipe.image.extractor import LocalPeakWindowSum


class PulseTimeCorrection:
    high_gain = 0
    low_gain = 1

    def __init__(self, n_harm, calib_file_path, tel_id=1, window_width=7, window_shift=3):
        self.fNumHarmonics = n_harm
        self.tel_id = tel_id
        self.n_modules = 265
        self.extractor = LocalPeakWindowSum(window_width=window_width,
                                            window_shift=window_shift)
        self.calib_file_path = calib_file_path
        self.fan_array = None
        self.fbn_array = None

        self.first_cap_array = np.zeros((265, 2, 7))

        self.load_calib_file()

    def load_calib_file(self):
        hf = h5py.File(self.calib_file_path, 'r')
        fan = hf.get('fan')
        self.fan_array = np.array(fan)
        fbn = hf.get('fbn')
        self.fbn_array = np.array(fbn)

    def get_corr_pulse(self, event, pulse):
        gain = 0
        pixel_ids = event.lst.tel[self.tel_id].svc.pixel_ids
        pulse_corr = np.zeros(1855)
        for nr in range(0, 265):
            for pix in range(0, 7):
                fc = self.get_first_capacitor(event, nr)[gain, pix]
                pixel = pixel_ids[nr * 7 + pix]
                pulse_corr[pixel] = pulse[pixel] - get_corr_time(fc%1024,
                                                                 self.fan_array[gain, pixel],
                                                                 self.fbn_array[gain, pixel],
                                                                 self.fNumHarmonics)
        return pulse_corr

    def call_corr_pulse_jit(self, event, pulse):
        gain = 0
        pixel_ids = event.lst.tel[self.tel_id].svc.pixel_ids
        pulse_corr = np.zeros(1855)
        for nr in prange(0, 265):
            self.first_cap_array[nr, :, :] = self.get_first_capacitor(event, nr)
        self.get_corr_pulse_jit(pulse, pulse_corr, pixel_ids, self.first_cap_array, self.fan_array, self.fbn_array, self.fNumHarmonics)
        return pulse_corr

    @staticmethod
    @njit(parallel=True)
    def get_corr_pulse_jit(pulse, pulse_corr, pixel_ids, first_capacitor, fan_array, fbn_array, fNumHarmonics):
        gain = 0
        for nr in prange(0, 265):
            for pix in prange(0, 7):
                fc = first_capacitor[nr, 0, pix]
                pixel = pixel_ids[nr * 7 + pix]
                pulse_corr[pixel] = pulse[pixel] - get_corr_time_jit(fc % 1024,
                                                               fan_array[gain, pixel],
                                                               fbn_array[gain, pixel],
                                                               fNumHarmonics)

    def get_first_capacitor(self, event, nr):
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[self.tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc


def get_corr_time(first_cap, fan, fbn, fNumHarmonics, fNumCap=1024):
    time = fan[0] / 2.
    for n in range(1, fNumHarmonics):
        time += fan[n] * np.cos((first_cap * n * 2 * np.pi) / fNumCap)
        time += fbn[n] * np.sin((first_cap * n * 2 * np.pi) / fNumCap)
    return time


@njit()
def get_corr_time_jit(first_cap, fan, fbn, fNumHarmonics, fNumCap=1024):
    time = fan[0] / 2.
    for n in prange(1, fNumHarmonics):
        time += fan[n] * np.cos((first_cap * n * 2 * np.pi) / fNumCap)
        time += fbn[n] * np.sin((first_cap * n * 2 * np.pi) / fNumCap)
    return time