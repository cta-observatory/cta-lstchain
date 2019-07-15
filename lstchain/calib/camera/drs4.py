import numpy as np
from numba import jit, prange


__all__ = ['DragonPedestal']


class DragonPedestal:
    size4drs = 4096

    def __init__(self, tel_id, n_module):
        self.tel_id = tel_id
        self.n_module = n_module
        self.n_pixels = n_module*7
        self.meanped = np.zeros((2, self.n_pixels, self.size4drs))
        self.numped = np.zeros((2, self.n_pixels, self.size4drs))
        self.first_cap_array = np.zeros((self.n_module, 2, 7))

    def fill_pedestal_event(self, event):
        expected_pixel_id = event.lst.tel[self.tel_id].svc.pixel_ids
        waveform = event.r0.tel[self.tel_id].waveform
        for nr_module in prange(0, self.n_module):
            self.first_cap_array[nr_module, :, :] = self.get_first_capacitor(event, nr_module)

        self._fill_pedestal_event_jit(waveform,
                                      expected_pixel_id,
                                      self.first_cap_array,
                                      self.meanped,
                                      self.numped)

    @staticmethod
    @jit(parallel=True)
    def _fill_pedestal_event_jit(waveform, expected_pixel_id, first_cap_array,
                                 meanped, numped):
        size4drs = 4096
        roisize = 40
        for nr_module in prange(0, 265):
            first_cap = first_cap_array[nr_module, :, :]
            for gain in prange(0, 2):
                for pix in prange(0, 7):
                    fc = first_cap[gain, pix]
                    pixel = expected_pixel_id[nr_module * 7 + pix]

                    posads0 = int((2 + fc) % size4drs)
                    if posads0 + 40 < 4096:
                        meanped[gain, pixel, posads0:(posads0 + 36)] += waveform[gain, pixel, 2:38]
                        numped[gain, pixel, posads0:(posads0 + 36)] += 1

                    else:
                        for k in prange(2, roisize - 2):
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
            fc[0, i] = first_cap[j] # high gain
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[1, i] = first_cap[j] # low gain
        return fc