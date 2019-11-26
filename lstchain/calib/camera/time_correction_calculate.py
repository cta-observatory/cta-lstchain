import numpy as np
import h5py

from ctapipe.core import Component
from ctapipe.core.traits import Int

from numba import jit, njit, prange

from ctapipe.image.extractor import LocalPeakWindowSum

__all__ = ['TimeCorrection']


class TimeCorrectionCalculate(Component):
    high_gain = 0
    low_gain = 1

    n_gain = 2
    n_modules = 265
    n_pixels = 1855

    tel_id = Int(1,
                 help='id of the telescope to calibrate'
                 ).tag(config=True)

    n_combine = Int(8,
                    help='Number of combine capacitor???'
                    ).tag(config=True)

    n_harm = Int(16,
                 help='Number of harmonic to fourier expansion'
                 ).tag(config=True)

    n_cap = Int(1024,
                help='Number of capacitor???'
                ).tag(config=True)

    window_width = Int(
        default_value=7,
        help='Define the width of the integration window'
    ).tag(config=True)

    window_shift = Int(
        default_value=3,
        help='Define the shift of the integration window'
             'from the peak_index (peak_index - shift)'
    ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n = int(self.n_cap/self.n_combine)
        self.fMeanVal = np.zeros((2, 1855, self.n))
        self.fNumMean = np.zeros((2, 1855, self.n))
        self.fNumCombine = self.n_combine
        self.fNumHarmonics = self.n_harm
        self.fNumCap = self.n_cap
        self.fNumPoints = int(self.fNumCap/self.fNumCombine)
        self.tel_id = self.tel_id
        self.first_cap_array = np.zeros((265, 2, 7))
        self.n_modules = 265
        self.extractor = LocalPeakWindowSum(window_width=self.window_width, window_shift=self.window_shift)

    def call_calib_pulse_time_jit(self, ev):
        if ev.r0.tel[self.tel_id].trigger_type == 1 and np.mean(ev.r1.tel[self.tel_id].waveform[0, :, 2:38]) > 100:
            for nr_module in prange(0, 265):
                self.first_cap_array[nr_module, :, :] = self.get_first_capacitor(ev, nr_module)

            pixel_ids = ev.lst.tel[self.tel_id].svc.pixel_ids
            charge, pulse_time = self.extractor(ev.r1.tel[self.tel_id].waveform[:, :, 2:38])
            self.calib_pulse_time_jit(charge,
                                      pulse_time,
                                      pixel_ids,
                                      self.first_cap_array,
                                      self.fMeanVal,
                                      self.fNumMean,
                                      fNumCap=self.fNumCap,
                                      fNumCombine=self.fNumCombine)

    @jit(parallel=True)
    def calib_pulse_time_jit(self,
                             charge,
                             pulse_time,
                             pixel_ids,
                             first_cap_array,
                             fMeanVal,
                             fNumMean,
                             fNumCap=1024,
                             fNumCombine=8):
        """
        Numba function for calib pulse
        """

        n_modules = 265
        n_gain = 2
        n_pix = 7
        for nr_module in prange(0, n_modules):
            for gain in prange(0, n_gain):
                for pix in prange(0, n_pix):
                    pixel = pixel_ids[nr_module * 7 + pix]
                    if charge[gain, pixel] > 200:
                        fc = first_cap_array[nr_module, :, :]
                        first_cap = (fc[gain, pix]) % fNumCap
                        fBin = int(first_cap / fNumCombine)
                        fMeanVal[gain, pixel, fBin] += pulse_time[gain, pixel]
                        fNumMean[gain, pixel, fBin] += 1

    def finalize(self):
        if np.sum(self.fNumMean == 0) > 0:
            raise RuntimeError("Not enough events to coverage all capacitor. "
                               "Please use more events to create pedestal file.")
        else:
            self.fMeanVal = self.fMeanVal / self.fNumMean

    def fit(self, pixel_id, gain=0):
        self.pos = np.zeros(self.fNumPoints)
        for i in range(0, self.fNumPoints):
            self.pos[i] = ( i +0.5 ) *self.fNumCombine

        self.fan = np.zeros(self.fNumHarmonics)
        self.fbn = np.zeros(self.fNumHarmonics)

        for n in range(0, self.fNumHarmonics):
            self.integrate_with_trig(self.pos, self.fMeanVal[gain, pixel_id], n, self.fan, self.fbn)

    def integrate_with_trig(self, x, y, n, an, bn):
        suma = 0
        sumb = 0

        for i in range(0, self.fNumPoints):
            suma += y[i] *self.fNumCombine *np.cos( 2 *np.pi * n *(x[i] /float(self.fNumCap)))
            sumb += y[i] *self.fNumCombine *np.sin( 2 *np.pi * n *(x[i] /float(self.fNumCap)))

        an[n] = suma *(2./(self.fNumPoints *self.fNumCombine))
        bn[n] = sumb *(2./(self.fNumPoints *self.fNumCombine))

    def get_first_capacitor(self, event, nr):
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[self.tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc

    def save_to_h5_file(self, path):
        fan_array = np.zeros((2, 1855, self.fNumHarmonics))
        fbn_array = np.zeros((2, 1855, self.fNumHarmonics))
        for pix_id in range(0, 1855):
            self.fit(pix_id, gain=0)
            fan_array[0, pix_id, :] = self.fan
            fbn_array[0, pix_id, :] = self.fbn

            self.fit(pix_id, gain=1)
            fan_array[1, pix_id, :] = self.fan
            fbn_array[1, pix_id, :] = self.fbn

        try:
            hf = h5py.File(path, 'w')
            hf.create_dataset('fan', data=fan_array)
            hf.create_dataset('fbn', data=fbn_array)
            hf.attrs['run id'] = 1625
            hf.attrs['n_harm'] = self.fNumHarmonics
        except Exception as err:
            print("FAILED!", err)
        hf.close()