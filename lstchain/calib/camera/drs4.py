import numpy as np
from numba import jit, prange


__all__ = ['DragonPedestal']

size4drs = 4096
# `size4drs` is number of capacitors in each pixel which using DRS4.
# "The sampler has 1024 capacitors per channel.
# Four channels of a chip are cascaded to obtain deeper sampling depth with 4096 capacitors.
# (https://arxiv.org/abs/1509.00548)

roi_size = 40
high_gain = 0
low_gain = 1
n_gain = 2
n_channel = 7


class DragonPedestal:
    """
        The DragonPedestal class to create pedestal
        for LST readout system using chip DRS4.
    """

    def __init__(self, tel_id, n_module):
        self.tel_id = tel_id
        self.n_module = n_module
        self.n_pixels = n_module*n_channel # Each module has 7 channels (pixels)
        self.meanped = np.zeros((n_gain, self.n_pixels, size4drs))
        self.numped = np.zeros((n_gain, self.n_pixels, size4drs))
        self.first_cap_array = np.zeros((n_module, n_gain, n_channel))

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
                                      self.n_module)

    @staticmethod
    @jit(parallel=True)
    def _fill_pedestal_event_jit(waveform, expected_pixel_id, first_cap_array,
                                 meanped, numped, n_module):
        for nr_module in prange(0, n_module):
            first_cap = first_cap_array[nr_module, :, :]
            for gain in prange(0, n_gain):
                for pix in prange(0, n_channel):
                    fc = first_cap[gain, pix]
                    pixel = expected_pixel_id[nr_module * 7 + pix]

                    posads0 = int((2 + fc) % size4drs)
                    if posads0 + 40 < size4drs:
                        meanped[gain, pixel, posads0:(posads0 + 36)] += waveform[gain, pixel, 2:38]
                        numped[gain, pixel, posads0:(posads0 + 36)] += 1

                    else:
                        for k in prange(2, roi_size - 2):
                            posads = int((k + fc) % size4drs)
                            val = waveform[gain, pixel, k]
                            meanped[gain, pixel, posads] += val
                            numped[gain, pixel, posads] += 1

    def finalize_pedestal(self):
        if np.sum(self.numped==0) > 0:
            raise RuntimeError("Not enough events to coverage all capacitor. "
                               "Please use more events to create pedestal file.")
        else:
            self.meanped = self.meanped / self.numped

    def get_first_capacitor(self, event, nr):
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[self.tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[low_gain, i] = first_cap[j]
        return fc