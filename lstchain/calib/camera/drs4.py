import numpy as np
from numba import jit


class DragonPedestal:
    n_pixels = 7
    roisize = 40
    size4drs = 4*1024
    high_gain = 0
    low_gain = 1

    def __init__(self, telid=0):
        self.telid = telid
        self.first_capacitor = np.zeros((2, 8))
        self.meanped = np.zeros((2, self.n_pixels, self.size4drs))
        self.numped = np.zeros((2, self.n_pixels, self.size4drs))

    def fill_pedestal_event(self, event, nr):
        first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]

        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            self.first_capacitor[self.high_gain, i] = first_cap[j]

        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            self.first_capacitor[self.low_gain, i] = first_cap[j]

        waveform = event.r0.tel[0].waveform[:, :, :]
        expected_pixel_id = event.lst.tel[self.telid].svc.pixel_ids
        self._fill_pedestal_event_jit(nr, waveform, expected_pixel_id, self.first_capacitor, self.meanped, self.numped)

    @staticmethod
    @jit(parallel=True)
    def _fill_pedestal_event_jit(nr, waveform, expected_pixel_id, first_cap, meanped, numped):
        size4drs = 4096
        roisize = 40
        for i in range(0, 2):
            for j in range(0, 7):
                fc = int(first_cap[i, j])
                pixel =  expected_pixel_id[nr*7 + j]
                posads0 = int((2+fc)%size4drs)
                if posads0 + 40 < 4096:
                    meanped[i, j, posads0:(posads0+36)] += waveform[i, pixel, 2:38]
                    numped[i, j, posads0:(posads0 + 36)] += 1
                else:
                    for k in range(2, roisize-2):
                        posads = int((k+fc)%size4drs)
                        val = waveform[i, pixel, k]
                        meanped[i, j, posads] += val
                        numped[i, j, posads] += 1

    def finalize_pedestal(self):
        try:
            self.meanped = self.meanped/self.numped
        except Exception as err:
            print("Not enough events to coverage all capacitor. Please use more events to create pedestal file.")
            print(err)

    def get_first_capacitor(self, event, nr):
        fc = np.zeros((2, 8))
        first_cap = event.lst.tel[self.telid].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc