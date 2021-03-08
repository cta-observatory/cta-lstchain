import h5py
import numpy as np

from numba import jit, njit, prange

from ctapipe.core import Component
from ctapipe.core.traits import Int, Float, Unicode
from ctapipe.image.extractor import ImageExtractor
from ctapipe.containers import EventType

__all__ = ['TimeCorrectionCalculate']

high_gain = 0
low_gain = 1

n_gain = 2
n_channel = 7
n_modules = 265
n_pixels = 1855


class TimeCorrectionCalculate(Component):
    """
        The TimeCorrectionCalculate class to create h5py
        file with coefficients for time correction curve
        of chip DRS4.
        Description of this method: "Analysis techniques
        and performance of the Domino Ring Sampler version 4
        based readout for the MAGIC telescopes [arxiv:1305.1007]
    """

    minimum_charge = Float(200,
                           help='Cut on charge. Default 200 ADC'
                          ).tag(config=True)

    tel_id = Int(1,
                 help='Id of the telescope to calibrate'
                 ).tag(config=True)

    n_combine = Int(8,
                    help='How many capacitors are combines in a single bin. Default 8'
                    ).tag(config=True)

    n_harmonics = Int(16,
                      help='Number of harmonic for Fourier series expansion. Default 16'
                      ).tag(config=True)

    n_capacitors = Int(1024,
                       help='Number of capacitors (1024 or 4096). Default 1024.'
                       ).tag(config=True)

    charge_product = Unicode(
        'LocalPeakWindowSum',
        help='Name of the charge extractor to be used'
    ).tag(config=True)

    calib_file_path = Unicode('',
                              allow_none=True,
                              help='Path to the time calibration file'
                              ).tag(config=True)

    def __init__(self, subarray, **kwargs):
        """
        The TimeCorrectionCalculate class to create h5py
        file with coefficients for time correction curve of chip DRS4.
        Description of this method: "Analysis techniques and performance
        of the Domino Ring Sampler version 4 based readout
        for the MAGIC telescopes [arxiv:1305.1007]

        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in charge extraction, such as reference
            pulse shape, sampling rate, neighboring pixels. Also required for
            configuring the TelescopeParameter traitlets.
        kwargs
        """
        super().__init__(**kwargs)

        self.n_bins = int(self.n_capacitors / self.n_combine)

        self.mean_values_per_bin = np.zeros((n_gain, n_pixels, self.n_bins))
        self.entries_per_bin = np.zeros((n_gain, n_pixels, self.n_bins))

        self.first_cap_array = np.zeros((n_modules, n_gain, n_channel))

        # load the waveform charge extractor
        self.extractor = ImageExtractor.from_name(
            self.charge_product,
            config=self.config,
            subarray=subarray
        )

        self.log.info(f"extractor {self.extractor}")
        self.sum_events = 0

    def calibrate_peak_time(self, event):
        """
        Fill bins using time pulse from LocalPeakWindowSum.
        Parameters
        ----------
        event : `ctapipe` event-container
        """

        if event.trigger.event_type == EventType.FLATFIELD:
            for nr_module in prange(0, n_modules):
                self.first_cap_array[nr_module, :, :] = self.get_first_capacitor(event, nr_module)

            pixel_ids = event.lst.tel[self.tel_id].svc.pixel_ids
            waveforms = event.r1.tel[self.tel_id].waveform
            no_gain_selection = np.zeros((waveforms.shape[0], waveforms.shape[1]), dtype=np.int64)
            # select both gain
            charge, peak_time = self.extractor(
                    event.r1.tel[self.tel_id].waveform[:, :, :],
                    self.tel_id,
                    no_gain_selection)
            self.calib_peak_time_jit(charge,
                                     peak_time,
                                     pixel_ids,
                                     self.first_cap_array,
                                     self.mean_values_per_bin,
                                     self.entries_per_bin,
                                     n_cap=self.n_capacitors,
                                     n_combine=self.n_combine,
                                     min_charge=self.minimum_charge)
            self.sum_events += 1

    @staticmethod
    @njit(parallel=True)
    def calib_peak_time_jit(charge,
                             peak_time,
                             pixel_ids,
                             first_cap_array,
                             mean_values_per_bin,
                             entries_per_bin,
                             n_cap,
                             n_combine,
                             min_charge):
        """
        Numba function for calibration pulse time.

        Parameters
        ----------
        pulse : ndarray
            Pulse time stored in a numpy array of shape
            (n_gain, n_pixels).
        charge : ndarray
            Charge in each pixel.
            (n_gain, n_pixels).
        pixel_ids: ndarray
            Array stored expected pixel id
            (n_pixels).
        first_cap_array : ndarray
            Value of first capacitor stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        mean_values_per_bin : ndarray
            Array to fill using pulse time
            stored in a numpy array of shape
            (n_gain, n_pixels, n_bins).
        entries_per_bin : ndarray
            Array to store number of entries per bin
            stored in a numpy array of shape
            (n_gain, n_pixels, n_bins).
        n_cap : int
            Number of capacitors
        n_combine : int
            Number of combine capacitors in a single bin

        """

        for nr_module in prange(0, n_modules):
            for gain in prange(0, n_gain):
                for pix in prange(0, n_channel):
                    pixel = pixel_ids[nr_module * 7 + pix]
                    if charge[gain, pixel] > min_charge: # cut change
                        fc = first_cap_array[nr_module, :, :]
                        first_cap = (fc[gain, pix]) % n_cap
                        bin = int(first_cap / n_combine)
                        mean_values_per_bin[gain, pixel, bin] += peak_time[gain, pixel]
                        entries_per_bin[gain, pixel, bin] += 1

    def finalize(self):
        if np.sum(self.entries_per_bin == 0) > 0:
            raise RuntimeError("Not enough events to coverage all capacitor. "
                               "Please use more events to time calibration file.")
        else:
            self.mean_values_per_bin = self.mean_values_per_bin / self.entries_per_bin
            self.save_to_hdf5_file()

    def fit(self, pixel_id, gain):
        """
            Fit data bins using Fourier series expansion
            Parameters
            ----------
            pixel_id : ndarray
            Array stored expected pixel id of shape
            (n_pixels).
            gain: int
            0 for high gain, 1 for low gain
        """
        self.pos = np.zeros(self.n_bins)
        for i in range(0, self.n_bins):
            self.pos[i] = ( i +0.5 ) *self.n_combine

        self.fan = np.zeros(self.n_harmonics) # cos coeff
        self.fbn = np.zeros(self.n_harmonics) # sin coeff

        for n in range(0, self.n_harmonics):
            self.integrate_with_trig(self.pos, self.mean_values_per_bin[gain, pixel_id], n, self.fan, self.fbn)

    def integrate_with_trig(self, x, y, n, an, bn):
        """
            Function to expanding into Fourier series
            Parameters
            ----------
            x : ndarray
            Array stored position in DRS4 ring of shape
            (n_bins).
            y: ndarray
            Array stored mean pulse time per bin of shape
            (n_bins)
            n : int
            n harmonic
            an: ndarray
            Array to fill with cos coeff of shape
            (n_harmonics)
            bn: ndarray
            Array to fill with sin coeff of shape
            (n_harmonics)
        """
        suma = 0
        sumb = 0

        for i in range(0, self.n_bins):
            suma += y[i] *self.n_combine * np.cos(2 * np.pi * n * (x[i] / float(self.n_capacitors)))
            sumb += y[i] *self.n_combine * np.sin(2 * np.pi * n * (x[i] / float(self.n_capacitors)))

        an[n] = suma *(2. / (self.n_bins * self.n_combine))
        bn[n] = sumb *(2. / (self.n_bins * self.n_combine))

    def get_first_capacitor(self, event, nr):
        fc = np.zeros((n_gain, n_channel))
        first_cap = event.lst.tel[self.tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[low_gain, i] = first_cap[j]
        return fc

    def save_to_hdf5_file(self):
        """
            Function to save Fourier series expansion coeff into hdf5 file
        """
        fan_array = np.zeros((n_gain, n_pixels, self.n_harmonics))
        fbn_array = np.zeros((n_gain, n_pixels, self.n_harmonics))
        for pix_id in range(0, n_pixels):
            self.fit(pix_id, gain=high_gain)
            fan_array[high_gain, pix_id, :] = self.fan
            fbn_array[high_gain, pix_id, :] = self.fbn

            self.fit(pix_id, gain=low_gain)
            fan_array[low_gain, pix_id, :] = self.fan
            fbn_array[low_gain, pix_id, :] = self.fbn

        try:
            with h5py.File(self.calib_file_path, 'w') as hf:
                hf.create_dataset('fan', data=fan_array)
                hf.create_dataset('fbn', data=fbn_array)
                hf.attrs['n_events'] = self.sum_events
                hf.attrs['n_harm'] = self.n_harmonics

        except Exception as err:
            print(f"FAILED to create the file {self.calib_file_path}", err)
