import h5py
import numpy as np
from numba import njit, prange

from ctapipe.core import Component
from ctapipe.core.traits import Int, Unicode


class PulseTimeCorrection(Component):
    high_gain = 0
    low_gain = 1

    n_gain = 2
    n_modules = 265
    n_pixels = 1855

    tel_id = Int(1,
                 help='id of the telescope to calibrate'
                 ).tag(config=True)

    calib_file_path = Unicode('',
                            allow_none=True,
                            help='Path to the time calibration file'
                            ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_harm = None
        self.fan_array = None
        self.fbn_array = None

        self.first_cap_array = np.zeros((self.n_modules, self.n_gain, 7))

        self.load_calib_file()

    def load_calib_file(self):
        """
            Function to load calibration file.
        """
        if self.calib_file_path:
            with h5py.File(self.calib_file_path, 'r') as hf:
                self.n_harm = hf["/"].attrs['n_harm']
                fan = hf.get('fan')
                self.fan_array = np.array(fan)
                fbn = hf.get('fbn')
                self.fbn_array = np.array(fbn)

    def get_corr_pulse(self, event, pulse):
        """
        Interpolate Spike A & B.
        Change waveform array.
        Parameters
        ----------
        event : `ctapipe` event-container
        pulse : ndarray
            pulse time in each pixel.
            Stored in a numpy array of shape
            (2, 1855).
        """
        pixel_ids = event.lst.tel[self.tel_id].svc.pixel_ids
        pulse_corr = np.empty((self.n_gain, self.n_pixels))
        for nr in prange(0, self.n_modules):
            self.first_cap_array[nr, :, :] = self.get_first_capacitor(event, nr)
        self.get_corr_pulse_jit(pulse,
                                pulse_corr,
                                pixel_ids,
                                self.first_cap_array,
                                self.fan_array,
                                self.fbn_array,
                                self.n_harm)
        return pulse_corr

    @staticmethod
    @njit(parallel=True)
    def get_corr_pulse_jit(pulse, pulse_corr, pixel_ids, first_capacitor, fan_array, fbn_array, fNumHarmonics):
        """
        Interpolate Spike A & B.
        Change waveform array.
        Parameters
        ----------
        waveform : ndarray
            Waveform stored in a numpy array of shape
            (n_gain, n_pix, n_samples).
        pixel_ids: ndarray
            Array stored expected pixel id
            (n_pix*n_modules).
        first_capacitor : ndarray
            Value of first capacitor stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        fan_array : ndarray
            Value of first capacitor from previous event
            stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        n_modules : int
            Number of harmonics
        """
        for gain in prange(0, 2):
            for nr in prange(0, 265):
                for pix in prange(0, 7):
                    fc = first_capacitor[nr, gain, pix]
                    pixel = pixel_ids[nr * 7 + pix]
                    pulse_corr[gain, pixel] = pulse[gain, pixel] - get_corr_time_jit(fc % 1024,
                                                                                     fan_array[gain, pixel],
                                                                                     fbn_array[gain, pixel],
                                                                                     fNumHarmonics)

    def get_first_capacitor(self, event, nr):
        """
            Get first capacitor values from event for nr module.
            Parameters
            ----------
            event : `ctapipe` event-container
            nr_module : number of module
            tel_id : id of the telescope
        """
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[self.tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc

@njit()
def get_corr_time_jit(first_cap, fan, fbn, fNumHarmonics, fNumCap=1024):
    time = fan[0] / 2.
    for n in prange(1, fNumHarmonics):
        time += fan[n] * np.cos((first_cap * n * 2 * np.pi) / fNumCap)
        time += fbn[n] * np.sin((first_cap * n * 2 * np.pi) / fNumCap)
    return time