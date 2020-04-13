import numpy as np
from astropy.io import fits
from ctapipe.core import Component
from ctapipe.core.traits import Unicode, Int
from abc import abstractmethod
from numba import jit, prange

__all__ = [
    'CameraR0Calibrator',
    'LSTR0Corrections',
    'NullR0Calibrator',
]


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
    tel_id = Int(1,
                 help='id of the telescope to calibrate'
                 ).tag(config=True)

    offset = Int(default_value=400,
                 help='Define the offset of the baseline').tag(config=True)

    r1_sample_start = Int(default_value=2,
                          help='Start sample for r1 waveform',
                          allow_none=True).tag(config=True)

    r1_sample_end = Int(default_value=38,
                        help='End sample for r1 waveform',
                        allow_none=True).tag(config=True)

    def __init__(self, **kwargs):
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
        super().__init__(**kwargs)

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

    pedestal_path = Unicode('',
                            allow_none=True,
                            help='Path to the LST pedestal binary file'
                            ).tag(config=True)

    def __init__(self, **kwargs):
        """
        The R0 calibrator for LST data.
        Fill the r1 container.
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
        super().__init__(**kwargs)
        self.n_module = 265
        self.n_gain = 2
        self.n_pix = 7
        self.size4drs = 4 * 1024
        self.roisize = 40
        self.high_gain = 0
        self.low_gain = 1
        self.last_run_with_old_firmware = 1574
        self.pedestal_value_array = np.zeros((self.n_gain,
                                              self.n_pix*self.n_module,
                                              self.size4drs+40),
                                              dtype=np.int16)

        self.first_cap_array = np.zeros((self.n_module,
                                         self.n_gain,
                                         self.n_pix))

        self.first_cap_time_lapse_array = np.zeros((self.n_module,
                                                    self.n_gain,
                                                    self.n_pix))

        self.last_reading_time_array = np.zeros((self.n_module,
                                                 self.n_gain,
                                                 self.n_pix,
                                                 self.size4drs))

        self.first_cap_array_spike = np.zeros((self.n_module,
                                               self.n_gain,
                                               self.n_pix))

        self.first_cap_old_array = np.zeros((self.n_module,
                                             self.n_gain,
                                             self.n_pix))

        self._load_calib()

    def calibrate(self, event):
        for tel_id in event.r0.tels_with_data:
            self.subtract_pedestal(event, tel_id)
            self.time_lapse_corr(event, tel_id)
            self.interpolate_spikes(event, tel_id)

            event.r1.tel[tel_id].trigger_type = event.r0.tel[tel_id].trigger_type

            event.r1.tel[tel_id].trigger_time = event.r0.tel[tel_id].trigger_time

            samples = event.r1.tel[tel_id].waveform[:, :, self.r1_sample_start:self.r1_sample_end]

            event.r1.tel[tel_id].waveform = samples.astype('int16') - self.offset


    def subtract_pedestal(self, event, tel_id):
        """
        Subtract cell offset using pedestal file.
        Fill the R1 container.
        Parameters
        ----------
        event : `ctapipe` event-container
        tel_id : id of the telescope
        """
        n_modules = event.lst.tel[tel_id].svc.num_modules

        for nr_module in range(0, n_modules):
            self.first_cap_array[nr_module, :, :] = self._get_first_capacitor(event, nr_module, tel_id)

        expected_pixel_id = event.lst.tel[tel_id].svc.pixel_ids
        samples = np.copy(event.r0.tel[tel_id].waveform)
        samples.astype('int16')

        samples = subtract_pedestal_jit(samples,
                                        expected_pixel_id,
                                        self.first_cap_array,
                                        self.pedestal_value_array,
                                        n_modules)

        event.r1.tel[self.tel_id].trigger_type = event.r0.tel[self.tel_id].trigger_type
        event.r1.tel[self.tel_id].trigger_time = event.r1.tel[self.tel_id].trigger_time
        event.r1.tel[self.tel_id].waveform = samples[:, :, :]


    def time_lapse_corr(self, event, tel_id):
        """
        Perform time lapse baseline corrections.
        Fill the R1 container or modifies R0 container.
        Parameters
        ----------
        event : `ctapipe` event-container
        tel_id : id of the telescope
        """

        run_id = event.lst.tel[tel_id].svc.configuration_id

        expected_pixel_id = event.lst.tel[tel_id].svc.pixel_ids
        local_clock_list = event.lst.tel[tel_id].evt.local_clock_counter
        n_modules = event.lst.tel[tel_id].svc.num_modules
        for nr_module in range(0, n_modules):
            self.first_cap_time_lapse_array[nr_module, :, :] = self._get_first_capacitor(event, nr_module, tel_id)

        #If R1 container exist modifies it
        if isinstance(event.r1.tel[self.tel_id].waveform, np.ndarray):
            samples = event.r1.tel[self.tel_id].waveform

            # We have 2 functions: one for data from 2018/10/10 to 2019/11/04 and
            # one for data from 2019/11/05 (from Run 1574) after update firmware.
            # The old readout (before 2019/11/05) is shifted by 1 cell.
            if run_id > self.last_run_with_old_firmware:
                do_time_lapse_corr(samples,
                                   expected_pixel_id,
                                   local_clock_list,
                                   self.first_cap_time_lapse_array,
                                   self.last_reading_time_array,
                                   n_modules)
            else:
                do_time_lapse_corr_data_from_20181010_to_20191104(samples,
                                                                  expected_pixel_id,
                                                                  local_clock_list,
                                                                  self.first_cap_time_lapse_array,
                                                                  self.last_reading_time_array,
                                                                  n_modules)

            event.r1.tel[self.tel_id].trigger_type = event.r0.tel[self.tel_id].trigger_type
            event.r1.tel[self.tel_id].trigger_time = event.r0.tel[self.tel_id].trigger_time
            event.r1.tel[self.tel_id].waveform = samples[:, :, :]

        else: # Modifies R0 container. This is for create pedestal file.
            samples = np.copy(event.r0.tel[self.tel_id].waveform)

            if run_id > self.last_run_with_old_firmware:
                do_time_lapse_corr(samples,
                                   expected_pixel_id,
                                   local_clock_list,
                                   self.first_cap_time_lapse_array,
                                   self.last_reading_time_array,
                                   n_modules)
            else:
                do_time_lapse_corr_data_from_20181010_to_20191104(samples,
                                                                  expected_pixel_id,
                                                                  local_clock_list,
                                                                  self.first_cap_time_lapse_array,
                                                                  self.last_reading_time_array,
                                                                  n_modules)

            event.r0.tel[self.tel_id].waveform = samples[:, :, :]


    def interpolate_spikes(self, event, tel_id):
        """
        Interpolates spike A & B.
        Fill the R1 container.
        Parameters
        ----------
        event : `ctapipe` event-container
        tel_id : id of the telescope
        """
        run_id = event.lst.tel[tel_id].svc.configuration_id

        self.first_cap_old_array[:, :, :] = self.first_cap_array_spike[:, :, :]
        n_modules = event.lst.tel[tel_id].svc.num_modules
        for nr_module in range(0, n_modules):
            self.first_cap_array_spike[nr_module, :, :] = self._get_first_capacitor(event, nr_module, tel_id)

        # Interpolate spikes should be done after pedestal subtraction and time lapse correction.
        if isinstance(event.r1.tel[tel_id].waveform, np.ndarray):
            waveform = event.r1.tel[tel_id].waveform[:, :, :]
            expected_pixel_id = event.lst.tel[tel_id].svc.pixel_ids
            samples = waveform.copy()

            # We have 2 functions: one for data from 2018/10/10 to 2019/11/04 and
            # one for data from 2019/11/05 (from Run 1574) after update firmware.
            # The old readout (before 2019/11/05) is shifted by 1 cell.
            if run_id > self.last_run_with_old_firmware:
                event.r1.tel[self.tel_id].waveform = self.interpolate_pseudo_pulses(samples,
                                                                                    expected_pixel_id,
                                                                                    self.first_cap_array_spike,
                                                                                    self.first_cap_old_array,
                                                                                    n_modules)
            else:
                event.r1.tel[self.tel_id].waveform = \
                    self.interpolate_pseudo_pulses_data_from_20181010_to_20191104(samples,
                                                                                  expected_pixel_id,
                                                                                  self.first_cap_array_spike,
                                                                                  self.first_cap_old_array,
                                                                                  n_modules)

            event.r1.tel[self.tel_id].trigger_type = event.r0.tel[self.tel_id].trigger_type
            event.r1.tel[self.tel_id].trigger_time = event.r0.tel[self.tel_id].trigger_time

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
        roi_size = 40
        size1drs = 1024
        size4drs = 4096
        n_gain = 2
        n_pix = 7
        for nr_module in prange(0, n_modules):
            for gain in prange(0, n_gain):
                for pix in prange(0, n_pix):
                    for k in prange(0, 4):
                        # looking for spike A first case
                        abspos = int(size1drs + 1 - roi_size - 2 - fc_old[nr_module, gain, pix] + k * size1drs + size4drs)
                        spike_A_position = int((abspos - fc[nr_module, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < roi_size-2):
                            # The correction is only needed for even
                            # last capacitor (lc) in the first half of the
                            # DRS4 ring
                            if ((fc_old[nr_module, gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[nr_module, gain, pix] + (roi_size-1)) % size1drs <= size1drs//2-1):
                                pixel = expected_pixel_id[nr_module * 7 + pix]
                                interpolate_spike_A(waveform, gain, spike_A_position, pixel)

                        # looking for spike A second case
                        abspos = int(roi_size - 1 + fc_old[nr_module, gain, pix] + k * size1drs)
                        spike_A_position = int((abspos - fc[nr_module, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < (roi_size-2)):
                            # The correction is only needed for even last capacitor (lc) in the
                            # first half of the DRS4 ring
                            if ((fc_old[nr_module, gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[nr_module, gain, pix] + (roi_size-1)) % size1drs <= size1drs//2-1):
                                pixel = expected_pixel_id[nr_module * 7 + pix]
                                interpolate_spike_A(waveform, gain, spike_A_position, pixel)
        return waveform

    @staticmethod
    @jit(parallel=True)
    def interpolate_pseudo_pulses_data_from_20181010_to_20191104(waveform, expected_pixel_id, fc, fc_old, n_modules):
        """
        Interpolate Spike A & B.
        This is function for data from 2018/10/10 to 2019/11/04 with old firmware.
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
        roi_size = 40
        size1drs = 1024
        size4drs = 4096
        n_gain = 2
        n_pix = 7
        for nr_module in prange(0, n_modules):
            for gain in prange(0, n_gain):
                for pix in prange(0, n_pix):
                    for k in prange(0, 4):
                        # looking for spike A first case
                        abspos = int(size1drs - roi_size - 2 -fc_old[nr_module, gain, pix]+ k * size1drs + size4drs)
                        spike_A_position = int((abspos - fc[nr_module, gain, pix]+ size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < roi_size-2):
                            # The correction is only needed for even
                            # last capacitor (lc) in the first half of the
                            # DRS4 ring
                            if ((fc_old[nr_module, gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[nr_module, gain, pix]+ (roi_size-1)) % size1drs <= size1drs//2-2):
                                pixel = expected_pixel_id[nr_module*7 + pix]
                                interpolate_spike_A(waveform, gain, spike_A_position, pixel)

                        # looking for spike A second case
                        abspos = int(roi_size - 2 + fc_old[nr_module, gain, pix]+ k * size1drs)
                        spike_A_position = int((abspos -fc[nr_module, gain, pix] + size4drs) % size4drs)
                        if (spike_A_position > 2 and spike_A_position < (roi_size-2)):
                            # The correction is only needed for even last capacitor (lc) in the
                            # first half of the DRS4 ring
                            if ((fc_old[nr_module, gain, pix] + (roi_size-1)) % 2 == 0 and (fc_old[nr_module, gain, pix] + (roi_size-1)) % size1drs <= size1drs//2-2):
                                pixel = expected_pixel_id[nr_module*7 + pix]
                                interpolate_spike_A(waveform, gain, spike_A_position, pixel)
        return waveform

    def _load_calib(self):
        """
        Function to load pedestal file.
        """
        if self.pedestal_path:
            with fits.open(self.pedestal_path) as f:
                pedestal_data = np.int16(f[1].data)
                self.pedestal_value_array[:, :, :self.size4drs] = \
                                                    pedestal_data - self.offset
                self.pedestal_value_array[:, :, self.size4drs:self.size4drs + 40] \
                    = pedestal_data[:, :, 0:40] - self.offset

    def _get_first_capacitor(self, event, nr_module, tel_id):
        """
        Get first capacitor values from event for nr module.
        Parameters
        ----------
        event : `ctapipe` event-container
        nr_module : number of module
        tel_id : id of the telescope
        """
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[tel_id].evt.first_capacitor_id[nr_module * 8:
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
    roi_size = 40
    n_gain = 2
    n_pix = 7
    for nr_module in prange(0, n_modules):
        for gain in prange(0, n_gain):
            for pix in prange(0, n_pix):
                pixel = expected_pixel_id[nr_module*7 + pix]
                position = int((fc_cap[nr_module, gain, pix]) % size4drs)
                waveform[gain, pixel, :] = \
                    (event_waveform[gain, pixel, :] -
                    pedestal_value_array[gain, pixel, position:position + roi_size])
    return waveform

@jit(parallel=True)
def do_time_lapse_corr(waveform, expected_pixel_id, local_clock_list,
                       fc, last_time_array, number_of_modules):
    """
    Numba function for time lapse baseline correction.
    Change waveform array.
    """
    size4drs = 4096
    size1drs = 1024
    roi_size = 40
    n_gain = 2
    n_pix = 7
    for nr_module in prange(0, number_of_modules):
        time_now = local_clock_list[nr_module]
        for gain in prange(0, n_gain):
            for pix in prange(0, n_pix):
                pixel = expected_pixel_id[nr_module*7 + pix]
                for k in prange(0, roi_size):
                    posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                    if last_time_array[nr_module, gain, pix, posads] > 0:
                        time_diff = time_now - last_time_array[nr_module, gain, pix, posads]
                        time_diff_ms = time_diff / (133.e3)
                        if time_diff_ms < 100:
                            val =(waveform[gain, pixel, k] - ped_time(time_diff_ms))
                            waveform[gain, pixel, k] = val

                posads0 = int((0 + fc[nr_module, gain, pix]) % size4drs)
                if posads0+roi_size < size4drs:
                    last_time_array[nr_module, gain, pix, (posads0):(posads0+roi_size)] = time_now
                else:
                    for k in prange(0, roi_size):
                        posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                        last_time_array[nr_module, gain, pix, posads] = time_now

                # now the magic of Dragon,
                # extra conditions on the number of capacitor times being updated
                # if the ROI is in the last quarter of each DRS4
                # for even channel numbers extra 12 slices are read in a different place
                # code from Takayuki & Julian
                if pix % 2 == 0:
                    first_cap = fc[nr_module, gain, pix]
                    if first_cap % size1drs > 767 and first_cap % size1drs < 1013:
                        start = int(first_cap) + size1drs
                        end = int(first_cap) + size1drs + 12
                        last_time_array[nr_module, gain, pix, (start%size4drs):(end%size4drs)] = time_now
                    elif first_cap % size1drs >= 1013:
                        channel = int(first_cap / size1drs)
                        for kk in range(first_cap + size1drs, ((channel + 2) * size1drs)):
                            last_time_array[nr_module, gain, pix, int(kk) % size4drs] = time_now

@jit(parallel=True)
def do_time_lapse_corr_data_from_20181010_to_20191104(waveform, expected_pixel_id, local_clock_list,
                                                      fc, last_time_array, number_of_modules):
    """
    Numba function for time lapse baseline correction.
    This is function for data from 2018/10/10 to 2019/11/04 with old firmware.
    Change waveform array.
    """
    size4drs = 4096
    size1drs = 1024
    roi_size = 40
    n_gain = 2
    n_pix = 7

    for nr_module in prange(0, number_of_modules):
        time_now = local_clock_list[nr_module]
        for gain in prange(0, n_gain):
            for pix in prange(0, n_pix):
                pixel = expected_pixel_id[nr_module * 7 + pix]
                for k in prange(0, roi_size):
                    posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                    if last_time_array[nr_module, gain, pix, posads] > 0:
                        time_diff = time_now - last_time_array[nr_module, gain, pix, posads]
                        time_diff_ms = time_diff / (133.e3)
                        if time_diff_ms < 100:
                            val = waveform[gain, pixel, k] - ped_time(time_diff_ms)
                            waveform[gain, pixel, k] = val

                posads0 = int((0 + fc[nr_module, gain, pix]) % size4drs)
                if posads0 + roi_size < size4drs and (posads0-1) > 1:
                    last_time_array[nr_module, gain, pix, (posads0-1):(posads0 + (roi_size-1))] = time_now
                else:
                    # Old firmware issue: readout shifted by 1 cell
                    for k in prange(-1, roi_size-1):
                        posads = int((k + fc[nr_module, gain, pix]) % size4drs)
                        last_time_array[nr_module, gain, pix, posads] = time_now

                # now the magic of Dragon,
                # if the ROI is in the last quarter of each DRS4
                # for even channel numbers extra 12 slices are read in a different place
                # code from Takayuki & Julian
                if pix % 2 == 0:
                    first_cap = fc[nr_module, gain, pix]
                    if first_cap % size1drs > 766 and first_cap % size1drs < 1013:
                        start = int(first_cap) + size1drs - 1
                        end = int(first_cap) + size1drs + 11
                        last_time_array[nr_module, gain, pix, (start % size4drs):(end % size4drs)] = time_now
                    elif first_cap % size1drs >= 1013:
                        channel = int(first_cap / size1drs)
                        for kk in range(first_cap + size1drs, (channel + 2) * size1drs):
                            last_time_array[nr_module, gain, pix, int(kk) % size4drs] = time_now

@jit
def ped_time(timediff):
    """
    Power law function for time lapse baseline correction.
    Coefficients from curve fitting to dragon test data
    at temperature 20 degC
    """
    # old values at 30 degC (used till release v0.4.5)
    #return 27.33 * np.power(timediff, -0.24) - 10.4

    # new values at 20 degC, provided by Yokiho Kobayashi 2/3/2020
    # see also Yokiho's talk in https://indico.cta-observatory.org/event/2664/
    return 32.99 * np.power(timediff, -0.22) - 11.9


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


class NullR0Calibrator(CameraR0Calibrator):
    """
    A dummy R0 calibrator that simply fills the r1 container with the samples
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log.info("Using NullR0Calibrator, if event source is at "
                      "the R0 level, then r1 samples will equal r0 samples")

    def calibrate(self, event):
        for tel_id in event.r0.tels_with_data:
            event.r1.tel[tel_id].trigger_type = event.r0.tel[tel_id].trigger_type
            event.r1.tel[tel_id].trigger_time = event.r0.tel[tel_id].trigger_time
            samples = event.r0.tel[tel_id].waveform[:, :, self.r1_sample_start:self.r1_sample_end]
            event.r1.tel[tel_id].waveform = samples.astype('int16') - self.offset
