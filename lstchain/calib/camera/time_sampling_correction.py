import h5py
import numpy as np


from ctapipe.core import Component
from ctapipe.core.traits import Unicode

__all__ = [
    'TimeSamplingCorrection',
    ]



class TimeSamplingCorrection(Component):
    """
        The PulseTimeCorrection class to correct time pulse
        using Fourier series expansion.
    """

    time_sampling_correction_path = Unicode(
        '',
        help='Path to the waveform sampling correction file',
        allow_none = True,
    ).tag(config=True)


    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.time_sampling_coefficients = None

        self.load_sampling_coefficient_file()

    def load_sampling_coefficient_file(self):
        """
            Function to load sampling coefficient file.
        """

        try:
            with h5py.File(self.time_sampling_correction_path, 'r') as hf:
                self.time_sampling_coefficients = np.array(hf['sampling_interval_coefficient'])
        except:
            self.log.error(f"Problem in reading sampling coefficient file {self.time_sampling_correction_path}")


    def get_corrections(self, event, telid):
        """
        Get the time/charge sampling corrections for one event and one telescope

        Parameters
        ----------
        event: general event container
        telid: id of the telescope

        Return
        ------
        sampling_corrections: np.array (n_gains, n_pixels, n_samples) with the correction factors
        """

        n_gains = 2
        n_modules = 265  # number of modules in LST's camera.
        n_pixel_per_module = 7  # number of pixels per one module.
        size_drs4 = 1024
        n_pixels = 1855
        roi = 36

        # shift the first capacitor to the value used in the r1
        r1_roi_start = 3

        fc_all = event.lst.tel[telid].evt.first_capacitor_id

        # get the first capacitor per pixel and gain
        fc = np.zeros((n_gains, n_pixels), dtype=np.int16)

        for k in range(n_modules):
            for n in range(n_pixel_per_module):
                fc[0][k * 7 + n] = fc_all[8 * k + n // 2]
                fc[1][k * 7 + n] = fc_all[8 * k + n // 2 + 4]

        # reorder first capacitor as in waveform
        fc_ordered = np.zeros((n_gains, n_pixels), dtype=np.int16)
        fc_ordered[:, event.lst.tel[telid].svc.pixel_ids] = fc

        # shift the first capacitor to the value used in the r1
        fc_ordered = fc_ordered + r1_roi_start

        # first capacitor in the drs4
        fc_drs4 = (fc_ordered[:, :]) % size_drs4

        # how many slices to the end of buffer
        fc_to_last = size_drs4 - fc_drs4[:, :]

        # initialize the charge correction vector to one
        sampling_corrections = np.ones((n_gains, n_pixels, roi))

        # loop over the gains and pixels
        for gain in range(n_gains):
            for pix in range(n_pixels):

                # if I am at the end of the 1024
                if 0 < fc_to_last[gain, pix] < roi:
                    # I complete the buffer
                    sampling_corrections[gain, pix, :fc_to_last[gain, pix]] = (
                            sampling_corrections[gain, pix,:fc_to_last[gain, pix]] *
                            self.time_sampling_coefficients[gain, pix, fc_drs4[gain, pix]:])

                    # I start again from the beginning of the buffer
                    sampling_corrections[gain, pix, fc_to_last[gain, pix]:] = (
                            sampling_corrections[gain, pix,fc_to_last[gain, pix]:] *
                            self.time_sampling_coefficients[gain, pix,:roi - fc_to_last[gain, pix]])
                else:
                    sampling_corrections[gain, pix, :] = (
                            sampling_corrections[gain, pix] *
                            self.time_sampling_coefficients[gain, pix, fc_drs4[gain, pix]:fc_drs4[gain, pix] + roi])

        return sampling_corrections

