import numpy as np
from astropy.io import fits
from ctapipe.core import Component, Factory
from ctapipe.core.traits import Unicode

from numba import jit, prange

class CameraR0Calibrator(Component):
    """
    The base R0-level calibrator. Change the r0 container.
    The R0 calibrator performs the camera-specific R0 calibration that is
    usually performed on the raw data by the camera server. This calibrator
    exists in lstchain for testing and prototyping purposes.
    Parameters
    ----------
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        Set to None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        Passes the correct logger to the component.
        Set to None if no Tool to pass.
    kwargs
    """

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Parent class for the r0 calibrators. Change the r0 container.
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)



class LSTR0Corrections(CameraR0Calibrator):
    """
    The R0 calibrator class for LST Camera.
    """

    pedestal_path = Unicode(
        '',
        allow_none=True,
        help='Path to the LST pedestal binary file'
    ).tag(config=True)

    def __init__(self, config=None, tool=None, offset=300, **kwargs):
        """
        The R0 calibrator for LST data.
        Change the r0 container.
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)
        self.telid = 0
        self.n_module = 265
        self.n_gain = 2
        self.n_pix = 7
        self.size4drs = 4 * 1024
        self.roisize = 40
        self.offset = offset
        self.high_gain = 0
        self.low_gain = 1

        self.pedestal_value_array = None
        self.first_cap_array = np.zeros((self.n_module, self.n_gain, self.n_pix))

        self.first_cap_time_lapse_array = np.zeros((self.n_module, self.n_gain, self.n_pix))
        self.last_reading_time_array = np.zeros((self.n_module, self.n_gain, self.n_pix, self.size4drs))

        self.first_cap_array_spike = np.zeros((self.n_module, self.n_gain, self.n_pix))
        self.first_cap_old_array = np.zeros((self.n_module, self.n_gain, self.n_pix))

        self._load_calib()

    def subtract_pedestal(self, event):
        """
        Subtracts cell offset using pedestal file.
        Parameters
        ----------
        event : `ctapipe` event-container
        """
        n_modules = event.lst.tel[0].svc.num_modules

        for nr_module in range(0, n_modules):
            self.first_cap_array[nr_module, :, :] = self._get_first_capacitor(event, nr_module)

        expected_pixel_id = event.lst.tel[0].svc.pixel_ids
        event.r0.tel[self.telid].waveform[:, :, :] = subtract_pedestal_jit(
            event.r0.tel[self.telid].waveform,
            expected_pixel_id,
            self.first_cap_array,
            self.pedestal_value_array,
            n_modules)
        event.r0.tel[self.telid].waveform.astype(np.uint16)

    def time_lapse_corr(self, event):
        """
        Perform time lapse corrections.
        Parameters
        ----------
        event : `ctapipe` event-container
        """
        expected_pixel_id = event.lst.tel[0].svc.pixel_ids
        EVB = event.lst.tel[0].evt.counters
        n_modules = event.lst.tel[0].svc.num_modules
        for nr_clus in range(0, n_modules):
            self.first_cap_time_lapse_array[nr_clus, :, :] = self._get_first_capacitor(event, nr_clus)

        do_time_lapse_corr(event.r0.tel[0].waveform, expected_pixel_id, EVB,
                           self.first_cap_time_lapse_array, self.last_reading_time_array, n_modules)

    def interpolate_spikes(self, event):
        """
        Interpolates spike A & B and change the R0 container.

        Parameters
        ----------
        event : `ctapipe` event-container
        """
        self.first_cap_old_array[:, :, :] = self.first_cap_array_spike[:, :, :]
        n_modules = event.lst.tel[0].svc.num_modules
        for nr_module in range(0, n_modules):
            self.first_cap_array_spike[nr_module, :, :] = self._get_first_capacitor(event, nr_module)

        waveform = event.r0.tel[0].waveform[:, :, :]
        expected_pixel_id = event.lst.tel[0].svc.pixel_ids
        wf = waveform.copy()
        wf = wf.astype('int16')
        event.r0.tel[0].waveform = self.interpolate_pseudo_pulses(wf,
                                                                  expected_pixel_id,
                                                                  self.first_cap_array_spike,
                                                                  self.first_cap_old_array,
                                                                  n_modules)

    @staticmethod
    @jit(parallel=True)
    def interpolate_pseudo_pulses(waveform, expected_pixel_id, fc, fc_old, n_modules):
        """
        Interpolate Spike A & B and change the R0 container.

        Parameters
        ----------
        waveform : ndarray
            Waveform stored in a numpy array of shape
            (n_gain, n_pix, n_samples).
        expected_pixel_id: ndarray
            Array stored expected pixel id
            (n_pix*n_modules).
        fc : ndarray
            Value of first capacitor stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        fc_old : ndarray
            Value of first capacitor from previous event
            stored in a numpy array of shape
            (n_clus, n_gain, n_pix).
        n_modules : int
            Number of modules
        """

        roisize = 40
        size4drs = 4096
        n_gain = 2
        n_pix = 7
        for nr_clus in prange(0, n_modules):
            for gain in prange(0, n_gain):
                for pix in prange(0, n_pix):
                    for k in prange(0, 4):
                        # looking for spike A first case
                        abspos = int(1024 - roisize - 2 - fc_old[nr_clus, gain, pix] + k * 1024 + size4drs)
                        spike_A_position = int((abspos - fc[nr_clus, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < 38):
                            pixel = expected_pixel_id[nr_clus*7 + pix]
                            interpolate_spike_A(waveform, gain, spike_A_position, pixel)
                        abspos = int(roisize - 2 + fc_old[nr_clus, gain, pix] + k * 1024 + size4drs)
                        spike_A_position = int((abspos - fc[nr_clus, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < 38):
                            pixel = expected_pixel_id[nr_clus*7 + pix]
                            interpolate_spike_A(waveform, gain, spike_A_position, pixel)

                    spike_b_position = int(
                        (fc_old[nr_clus, gain, pix] - 1 - fc[nr_clus, gain, pix] + 2 * size4drs) % size4drs)
                    if spike_b_position < roisize - 1:
                        pixel = expected_pixel_id[nr_clus*7 + pix]
                        interpolate_spike_B(waveform, gain, spike_b_position, pixel)

        return waveform

    def _load_calib(self):
        """event.r0.tel[0].waveform
        If a pedestal file has been supplied, create a array with
        pedestal value . If it hasn't then point calibrate to
        fake_calibrate, where nothing is done to the waveform.
        """

        if self.pedestal_path:
            with fits.open(self.pedestal_path) as f:
                n_modules = f[1].header['NAXIS4']
                self.pedestal_value_array = np.zeros((n_modules, 2, 7, 4136), dtype=np.int16)
                pedestal_data = np.int16(f[1].data)
                self.pedestal_value_array[:, :, :, :self.size4drs] = pedestal_data - self.offset
                self.pedestal_value_array[:, :, :, self.size4drs:self.size4drs + 40] \
                    = pedestal_data[:, :, :, 0:40] - self.offset

    def _get_first_capacitor(self, event, nr_module):
        """
        Get first capacitor values from event for nr module.
        Parameters
        ----------
        event : `ctapipe` event-container
        nr_module : number of module
        """
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[0].evt.first_capacitor_id[nr_module * 8:
                                                            (nr_module + 1) * 8]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc

@jit(parallel=True)
def subtract_pedestal_jit(event_waveform, expected_pixel_id, fc_cap, pedestal_value_array, n_modules):
    waveform = np.zeros(event_waveform.shape)
    size4drs = 4096
    n_gain = 2
    n_pix = 7
    for nr in prange(0, n_modules):
        for gain in prange(0, n_gain):
            for pix in prange(0, n_pix):
                pixel = expected_pixel_id[nr*7 + pix]
                position = int((fc_cap[nr, gain, pix]) % size4drs)
                waveform[gain, pixel, :] = \
                    (event_waveform[gain, pixel, :] -
                    pedestal_value_array[nr, gain, pix, position:position + 40])
    return waveform

@jit(parallel=True)
def do_time_lapse_corr(waveform, expected_pixel_id, EVB, fc, last_time_array, number_of_modules):
    size4drs = 4096
    for nr_clus in prange(0, number_of_modules):
        time_now = int64(EVB[14 + (nr_clus * 22): 22 + (nr_clus * 22)])
        for gain in prange(0, 2):
            for pix in prange(0, 7):
                pixel = expected_pixel_id[nr_clus*7 + pix]
                for k in prange(0, 40):
                    posads = int((k + fc[nr_clus, gain, pix]) % size4drs)
                    if last_time_array[nr_clus, gain, pix, posads] > 0:
                        time_diff = time_now - last_time_array[nr_clus, gain, pix, posads]
                        val = waveform[gain, pixel, k] - ped_time(time_diff / (133.e3))
                        waveform[gain, pixel, k] = val
                    if (k < 39):
                        last_time_array[nr_clus, gain, pix, posads] = time_now

@jit
def ped_time(timediff):
    return 29.3 * np.power(timediff, -0.2262) - 12.4

@jit
def int64(x):
    return x[0] + x[1] * 256 + x[2] * 256 ** 2 + x[3] * 256 ** 3 + x[4] * 256 ** 4 + x[5] * 256 ** 5 + x[
            6] * 256 ** 6 + x[7] * 256 ** 7

@jit
def interpolate_spike_A(waveform, gain, pos, pixel):
    samples = waveform[gain, pixel, :]
    a = int(samples[pos - 1])
    b = int(samples[pos + 2])
    waveform[gain, pixel, pos] = (samples[pos - 1]) + (0.33 * (b - a))
    waveform[gain, pixel, pos + 1] = (samples[pos - 1]) + (0.66 * (b - a))

@jit
def interpolate_spike_B(waveform, gain, pos, pixel):
    samples = waveform[gain, pixel, :]
    waveform[gain, pixel, pos] = 0.5 * (samples[pos - 1] + samples[pos + 1])