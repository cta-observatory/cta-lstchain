import numpy as np
from numba import jit, prange

from ctapipe.core import Component
from ctapipe.core.traits import Int


__all__ = ['DragonPedestal']

size4drs = 4096
# `size4drs` is number of capacitors in each pixel which using DRS4.
# "The sampler has 1024 capacitors per channel.
# Four channels of a chip are cascaded to obtain deeper sampling depth with 4096 capacitors.
# (https://arxiv.org/abs/1509.00548)

roi_size = 40
high_gain = 0
low_gain = 1
n_module_in_camera = 265
n_gain = 2
n_channel = 7


class DragonPedestal(Component):
    """
        The DragonPedestal class to create pedestal
        for LST readout system using chip DRS4.
    """

    r0_sample_start = Int(default_value=11,
                          help='Start sample for waveform'
                          ).tag(config=True)

    def __init__(self, tel_id, n_module, **kwargs):
        super().__init__(**kwargs)
        self.tel_id = tel_id
        self.n_module = n_module # This is number of module read from data

        # Readout system of LST has 265 modules.
        # Each module has 7 channels (pixels)
        self.n_pixels = n_module_in_camera*n_channel
        self.meanped = np.zeros((n_gain, self.n_pixels, size4drs))
        self.numped = np.zeros((n_gain, self.n_pixels, size4drs))
        self.first_cap_array = np.zeros((self.n_module, n_gain, n_channel))
        self.failing_pixels_array = np.full((self.n_pixels), False)

    def fill_pedestal_event(self, event):
        expected_pixel_id = event.lst.tel[self.tel_id].svc.pixel_ids
        waveform = event.r0.tel[self.tel_id].waveform
        for nr_module in prange(0, self.n_module):
            self.first_cap_array[nr_module, :, :] = self.get_first_capacitor(event, nr_module)

        self._fill_pedestal_event_jit(waveform,
                                      expected_pixel_id,
                                      self.first_cap_array,
                                      self.meanped,
                                      self.numped,
                                      self.n_module,
                                      self.r0_sample_start)

    @staticmethod
    @jit(parallel=True)
    def _fill_pedestal_event_jit(waveform, expected_pixel_id, first_cap_array,
                                 meanped, numped, n_module, start_sample_r0):
        for nr_module in prange(0, n_module):
            first_cap = first_cap_array[nr_module, :, :]
            for gain in prange(0, n_gain):
                for pix in prange(0, n_channel):
                    fc = first_cap[gain, pix]
                    pixel = expected_pixel_id[nr_module * 7 + pix]
                    posads0 = int((start_sample_r0+fc)%size4drs)
                    if posads0 + roi_size < size4drs:
                        # the first 9 samples have occasionally increased signal due to Tsutomu pattern,
                        # hence we skip them. Start sample might be set as script argument. Default = 11.
                        meanped[gain, pixel, posads0:((posads0-start_sample_r0) + roi_size-2)] += waveform[gain, pixel, start_sample_r0:roi_size-2]
                        numped[gain, pixel, posads0:((posads0-start_sample_r0) + roi_size-2)] += 1
                    else:
                        for k in prange(start_sample_r0, roi_size - 2):
                            # the first 9 samples have occasionally increased signal due to Tsutomu pattern,
                            # hence we skip them. Start sample might be set as script argument. Default = 11.
                            posads = int((k + fc) % size4drs)
                            val = waveform[gain, pixel, k]
                            meanped[gain, pixel, posads] += val
                            numped[gain, pixel, posads] += 1

    def finalize_pedestal(self):
        self.meanped = self.meanped / self.numped
        pixels_with_nan_value = np.where(np.isnan(self.meanped).any(axis=0))
        if len(pixels_with_nan_value[0]) > 0:
            # Find failing pixels id
            index_failing_pixels = np.unique(pixels_with_nan_value[0])
            self.failing_pixels_array[index_failing_pixels] = True
            print("Failing pixels:")
            print(index_failing_pixels)

    def get_first_capacitor(self, event, nr):
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[self.tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[low_gain, i] = first_cap[j]
        return fc
