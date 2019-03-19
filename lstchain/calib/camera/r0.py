import numpy as np
from astropy.io import fits
from ctapipe.core import Component
from ctapipe.core.traits import Unicode, Int
from abc import abstractmethod
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


    @abstractmethod
    def calibrate(self, event):
        """
        Abstract method to be defined in child class.

        Perform the conversion from raw R0 data to R1 data
        (ADC Samples -> PE Samples), and fill the r1 container.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """

class LSTR0Corrections(CameraR0Calibrator):
    """
    The R0 calibrator class for LST Camera.
    """

    pedestal_path = Unicode(
        '',
        allow_none=True,
        help='Path to the LST pedestal binary file'
    ).tag(config=True)
    telid = Int(
        0,
        help='id of the telescope to calibrate'
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

        self.n_module = 265
        self.n_gain = 2
        self.n_pix = 7
        self.size4drs = 4 * 1024
        self.roisize = 40
        self.offset = offset
        self.high_gain = 0
        self.low_gain = 1

        self.pedestal_value_array = np.zeros((self.n_gain, self.n_pix*self.n_module, self.size4drs+40), dtype=np.int16)
        self.first_cap_array = np.zeros((self.n_module, self.n_gain, self.n_pix))

        self.first_cap_time_lapse_array = np.zeros((self.n_module, self.n_gain, self.n_pix))
        self.last_reading_time_array = np.zeros((self.n_module, self.n_gain, self.n_pix, self.size4drs))

        self.first_cap_array_spike = np.zeros((self.n_module, self.n_gain, self.n_pix))
        self.first_cap_old_array = np.zeros((self.n_module, self.n_gain, self.n_pix))

        self._load_calib()

    def calibrate(self, event):
        for telid in event.r0.tels_with_data:
            self.subtract_pedestal(event)
            self.time_lapse_corr(event)
            self.interpolate_spikes(event)

            samples = event.r0.tel[telid].waveform[:,:,2:38]
            event.r1.tel[telid].waveform = samples.astype('float32')


    def subtract_pedestal(self, event):
        """
        Subtract cell offset using pedestal file.
        Change the R0 container.

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
        Perform time lapse baseline corrections.
        Change the R0 container.

        Parameters
        ----------
        event : `ctapipe` event-container
        """
        expected_pixel_id = event.lst.tel[0].svc.pixel_ids
        local_clock_list = event.lst.tel[0].evt.local_clock_counter
        n_modules = event.lst.tel[0].svc.num_modules
        for nr_module in range(0, n_modules):
            self.first_cap_time_lapse_array[nr_module, :, :] = self._get_first_capacitor(event, nr_module)

        do_time_lapse_corr(event.r0.tel[0].waveform, expected_pixel_id, local_clock_list,
                           self.first_cap_time_lapse_array, self.last_reading_time_array, n_modules)
        event.r0.tel[self.telid].waveform.astype(np.uint16)

    def interpolate_spikes(self, event):
        """
        Interpolates spike A & B.
        Change the R0 container.

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
        event.r0.tel[self.telid].waveform.astype(np.uint16)

    @staticmethod
    @jit(parallel=True)
    def interpolate_pseudo_pulses(waveform, expected_pixel_id, fc, fc_old, n_modules):
        """
        Interpolate Spike A & B.
        Change waveform array.

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
        for nr_module in prange(0, n_modules):
            for gain in prange(0, n_gain):
                for pix in prange(0, n_pix):
                    for k in prange(0, 4):
                        # looking for spike A first case
                        abspos = int(1024 - roisize - 2 - fc_old[nr_module, gain, pix] + k * 1024 + size4drs)
                        spike_A_position = int((abspos - fc[nr_module, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < 38):
                            pixel = expected_pixel_id[nr_module*7 + pix]
                            interpolate_spike_A(waveform, gain, spike_A_position, pixel)
                        # looking for spike A second case
                        abspos = int(roisize - 2 + fc_old[nr_module, gain, pix] + k * 1024 + size4drs)
                        spike_A_position = int((abspos - fc[nr_module, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < 38):
                            pixel = expected_pixel_id[nr_module*7 + pix]
                            interpolate_spike_A(waveform, gain, spike_A_position, pixel)

                    # looking for spike B
                    spike_b_position = int(
                        (fc_old[nr_module, gain, pix] - 1 - fc[nr_module, gain, pix] + 2 * size4drs) % size4drs)
                    if spike_b_position < roisize - 1:
                        pixel = expected_pixel_id[nr_module*7 + pix]
                        interpolate_spike_B(waveform, gain, spike_b_position, pixel)
        return waveform

    def _load_calib(self):
        """
        Function to load pedestal file.
        """
        if self.pedestal_path:
            with fits.open(self.pedestal_path) as f:
                pedestal_data = np.int16(f[1].data)
                self.pedestal_value_array[:, :, :self.size4drs] = pedestal_data - self.offset
                self.pedestal_value_array[:, :, self.size4drs:self.size4drs + 40] \
                    = pedestal_data[:, :, 0:40] - self.offset

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

        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc

@jit(parallel=True)
def subtract_pedestal_jit(event_waveform, expected_pixel_id, fc_cap, pedestal_value_array, n_modules):
    """
    Numba function for subtract pedestal.
    Change waveform array.
    """
    waveform = np.zeros(event_waveform.shape)
    size4drs = 4096
    n_gain = 2
    n_pix = 7
    for nr_module in prange(0, n_modules):
        for gain in prange(0, n_gain):
            for pix in prange(0, n_pix):
                pixel = expected_pixel_id[nr_module*7 + pix]
                position = int((fc_cap[nr_module, gain, pix]) % size4drs)
                waveform[gain, pixel, :] = \
                    (event_waveform[gain, pixel, :] -
                    pedestal_value_array[gain, pixel, position:position + 40])
    return waveform

@jit(parallel=True)
def do_time_lapse_corr(waveform, expected_pixel_id, local_clock_list, fc, last_time_array, number_of_modules):
    """
    Numba function for time lapse baseline correction.
    Change waveform array.
    """
    size4drs = 4096
    for nr_module in prange(0, number_of_modules):
        time_now = local_clock_list[nr_module]
        for gain in prange(0, 2):
            for pix in prange(0, 7):
                pixel = expected_pixel_id[nr_module*7 + pix]
                for k in prange(0, 40):
                    posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                    if last_time_array[nr_module, gain, pix, posads] > 0:
                        time_diff = time_now - last_time_array[nr_module, gain, pix, posads]
                        val = waveform[gain, pixel, k] - ped_time(time_diff / (133.e3))
                        waveform[gain, pixel, k] = val

                posads0 = int((0 + fc[nr_module, gain, pix]) % size4drs)
                if posads0+40 < 4096:
                    last_time_array[nr_module, gain, pix, posads0:(posads0+39)] = time_now
                else:
                    for k in prange(0, 39):
                        posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                        last_time_array[nr_module, gain, pix, posads] = time_now

@jit
def ped_time(timediff):
    """
    Power law function for time lapse baseline correction.
    Coefficients from curve fitting.
    """
    return 29.3 * np.power(timediff, -0.2262) - 12.4

@jit
def interpolate_spike_A(waveform, gain, position, pixel):
    """
    Numba function for interpolation spike type A.
    Change waveform array.
    """
    samples = waveform[gain, pixel, :]
    a = int(samples[position - 1])
    b = int(samples[position + 2])
    waveform[gain, pixel, position] = (samples[position - 1]) + (0.33 * (b - a))
    waveform[gain, pixel, position + 1] = (samples[position - 1]) + (0.66 * (b - a))

@jit
def interpolate_spike_B(waveform, gain, position, pixel):
    """
    Numba function for interpolation spike type B.
    Change waveform array.
    """
    samples = waveform[gain, pixel, :]
    waveform[gain, pixel, position] = 0.5 * (samples[position - 1] + samples[position + 1])


class NullR0Calibrator(CameraR0Calibrator):
    """
    A dummy R1 calibrator that simply fills the r1 container with the samples
    from the r0 container.

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

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)
        self.log.info("Using NullR0Calibrator, if event source is at "
                      "the R0 level, then r1 samples will equal r0 samples")

    def calibrate(self, event):
        for telid in event.r0.tels_with_data:
            samples = event.r0.tel[telid].waveform[:,:,2:38]
            event.r1.tel[telid].waveform = samples.astype('float32')

