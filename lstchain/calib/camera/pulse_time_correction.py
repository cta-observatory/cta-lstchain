import h5py
import numpy as np
from numba import njit, prange

from ctapipe.core import Component
from ctapipe.core.traits import Int, Unicode

__all__ = [
    'PulseTimeCorrection',
    'get_corr_time_jit'
    ]


high_gain = 0
low_gain = 1
n_gain = 2
n_channel = 7
n_modules = 265
n_pixels = 1855


class PulseTimeCorrection(Component):
    """
        The PulseTimeCorrection class to correct time pulse
        using Fourier series expansion.
    """
    tel_id = Int(1,
                 help='id of the telescope to calibrate'
                 ).tag(config=True)

    n_capacitors = Int(1024,
                       help='number of capacitors (1024 or 4096)'
                       ).tag(config=True)

    calib_file_path = Unicode('',
                            allow_none=True,
                            help='Path to the time calibration file'
                            ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_harmonics = None
        self.fan_array = None # array to store cos coeff for Fourier series expansion
        self.fbn_array = None # array to store sin coeff for Fourier series expansion
        self.first_cap_array = np.zeros((n_modules, n_gain, n_channel))

        self.load_calib_file()

    def load_calib_file(self):
        """
            Function to load calibration file.
        """

        try:

            with h5py.File(self.calib_file_path, 'r') as hf:
                self.n_harmonics = hf["/"].attrs['n_harm']
                fan = hf.get('fan')
                self.fan_array = np.array(fan)
                fbn = hf.get('fbn')
                self.fbn_array = np.array(fbn)

        except:
            self.log.error(f"Problem in reading time from calibration file {self.calib_file_path}")

    def get_corr_pulse(self, event, pulse):
        """
        Return pulse time after time correction.
        Parameters
        ----------
        event : `ctapipe` event-container
        pulse : ndarray
            pulse time in each pixel.
            Stored in a numpy array of shape
            (2, 1855).
        """
        pixel_ids = event.lst.tel[self.tel_id].svc.pixel_ids
        n_modules_from_event = event.lst.tel[self.tel_id].svc.num_modules
        pulse_corr = np.empty((n_gain, n_pixels))
        for nr in prange(0, n_modules_from_event):
            self.first_cap_array[nr, :, :] = self.get_first_capacitor(event, nr)
        self.get_corr_pulse_jit(pulse,
                                pulse_corr,
                                pixel_ids,
                                self.first_cap_array,
                                self.fan_array,
                                self.fbn_array,
                                self.n_harmonics,
                                self.n_capacitors)
        return pulse_corr

    def get_pulse_time_corrections(self, event):
        """
        Return pulse time after time correction.
        Parameters
        ----------
        event : `ctapipe` event-container

        Return
        ------
        pulse_time_corrections (to be subtrated) : ndarray
            Pulse time stored in a numpy array of shape
            (n_gain, n_pixels).
        """
        pixel_ids = event.lst.tel[self.tel_id].svc.pixel_ids
        n_modules_from_event = event.lst.tel[self.tel_id].svc.num_modules
        pulse_time_corrections = np.empty((n_gain, n_pixels))
        for nr in prange(0, n_modules_from_event):
            self.first_cap_array[nr, :, :] = self.get_first_capacitor(event, nr)
        self.get_pulse_time_correction_jit(pulse_time_corrections,
                                pixel_ids,
                                self.first_cap_array,
                                self.fan_array,
                                self.fbn_array,
                                self.n_harmonics,
                                self.n_capacitors)
        return pulse_time_corrections



    @staticmethod
    @njit(parallel=True)
    def get_corr_pulse_jit(pulse, pulse_corr, pixel_ids, first_capacitor, fan_array, fbn_array, n_harmonics, n_cap):
        """
        Numba function for pulse time correction.
        Parameters
        ----------
        pulse : ndarray
            Pulse time stored in a numpy array of shape
            (n_gain, n_pixels).
        pulse_corr : ndarray
            Pulse correction time stored in a numpy array of shape
            (n_gain, n_pixels).
        pixel_ids: ndarray
            Array stored expected pixel id
            (n_pixels).
        first_capacitor : ndarray
            Value of first capacitor stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        fan_array : ndarray
            Array to store coeff for Fourier series expansion
            stored in a numpy array of shape
            (n_gain, n_pixels, n_harmonics).
        fbn_array : ndarray
            Array to store coeff for Fourier series expansion
            stored in a numpy array of shape
            (n_gain, n_pixels, n_harmonics).
        n_harmonics : int
            Number of harmonics
        """
        for gain in prange(0, n_gain):
            for nr in prange(0, n_modules):
                for pix in prange(0, n_channel):
                    fc = first_capacitor[nr, gain, pix]
                    pixel = pixel_ids[nr * 7 + pix]
                    pulse_corr[gain, pixel] = pulse[gain, pixel] - get_corr_time_jit(fc % n_cap,
                                                                                     fan_array[gain, pixel],
                                                                                     fbn_array[gain, pixel],
                                                                                     n_harmonics,
                                                                                     n_cap)
    @staticmethod
    @njit(parallel=True)
    def get_pulse_time_correction_jit(time_corrections, pixel_ids, first_capacitor, fan_array, fbn_array, n_harmonics, n_cap):
        """
        Numba function for pulse time correction.
        Parameters
        ----------

        time_corrections : ndarray
            Pulse correction time stored in a numpy array of shape
            (n_gain, n_pixels).
        pixel_ids: ndarray
            Array stored expected pixel id
            (n_pixels).
        first_capacitor : ndarray
            Value of first capacitor stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        fan_array : ndarray
            Array to store coeff for Fourier series expansion
            stored in a numpy array of shape
            (n_gain, n_pixels, n_harmonics).
        fbn_array : ndarray
            Array to store coeff for Fourier series expansion
            stored in a numpy array of shape
            (n_gain, n_pixels, n_harmonics).
        n_harmonics : int
            Number of harmonics
        """
        for gain in prange(0, n_gain):
            for nr in prange(0, n_modules):
                for pix in prange(0, n_channel):
                    fc = first_capacitor[nr, gain, pix]
                    pixel = pixel_ids[nr * 7 + pix]
                    time_corrections[gain, pixel] = get_corr_time_jit(fc % n_cap,
                                                                      fan_array[gain, pixel],
                                                                      fbn_array[gain, pixel],
                                                                      n_harmonics,
                                                                      n_cap)

    def get_first_capacitor(self, event, nr):
        """
            Get first capacitor values from event for nr module.
            Parameters
            ----------
            event : `ctapipe` event-container
            nr_module : number of module
            tel_id : id of the telescope
        """
        fc = np.zeros((n_gain, n_channel))
        first_cap = event.lst.tel[self.tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[low_gain, i] = first_cap[j]
        return fc

@njit()
def get_corr_time_jit(first_cap, fan, fbn, n_harmonics, n_cap):

    #time = fan[0] / 2. #commented because time flat-fielding is performed in charge calibrator
    time = 0
    for n in prange(1, n_harmonics):
        time += fan[n] * np.cos((first_cap * n * 2 * np.pi) / n_cap)
        time += fbn[n] * np.sin((first_cap * n * 2 * np.pi) / n_cap)
    return time
