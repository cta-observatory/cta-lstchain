"""
Calibration functions
"""

import numpy as np
from astropy.io import fits
from ctapipe.core import Component, Factory
from ctapipe.core.traits import Unicode
from ctapipe.image.charge_extractors import LocalPeakIntegrator
from ctapipe.calib.camera.gainselection import ThresholdGainSelector

def lst_calibration(event, telescope_id):
    """
    Custom lst calibration.
    Update event.dl1.tel[telescope_id] with calibrated image and peakpos

    Parameters
    ----------
    event: ctapipe event container
    telescope_id: int
    """

    data = event.r0.tel[telescope_id].waveform

    ped = event.mc.tel[telescope_id].pedestal  # the pedestal is the
    # average (for pedestal events) of the *sum* of all samples,
    # from sim_telarray


    nsamples = data.shape[2]  # total number of samples

    # Subtract pedestal baseline. atleast_3d converts 2D to 3D matrix

    pedcorrectedsamples = data - np.atleast_3d(ped) / nsamples

    integrator = LocalPeakIntegrator(None, None)
    integration, peakpos, window = integrator.extract_charge(
        pedcorrectedsamples)  # these are 2D matrices num_gains * num_pixels

    signals = integration.astype(float)

    dc2pe = event.mc.tel[telescope_id].dc_to_pe  # numgains * numpixels
    signals *= dc2pe

    event.dl1.tel[telescope_id].image = signals
    event.dl1.tel[telescope_id].peakpos = peakpos

def gain_selection(waveform, signals, peakpos, cam_id, threshold):

    """
    Custom lst calibration.
    Update event.dl1.tel[telescope_id] with calibrated image and peakpos

    Parameters
    ----------
    waveform: array of waveforms of the events
    signals: array of calibrated pixel charges
    peakpos: array of pixel peak positions
    cam_id: str
    threshold: int threshold to change form high gain to low gain
    """
    '''
    combined = signals[0].copy() 
    peaks = peakpos[0].copy()
    for pixel in range(0, combined.size):
            if np.any(waveform[0][pixel] >= threshold):
                combined[pixel] = signals[1][pixel]
                peaks[pixel] = peakpos[1][pixel]
    '''
    ###Gain Selection using ctapipe GainSelector###
    gainsel = ThresholdGainSelector(select_by_sample=True)
    gainsel.thresholds[cam_id] = threshold
    
    waveform, gainmask = gainsel.select_gains(cam_id,
                                              waveform)
    signalmask = np.zeros(waveform.shape[0],
                          dtype=bool)

    for i in range(signalmask.size):
        signalmask[i] = gainmask[i].any()==True

    combined = signals[0].copy()
    combined[signalmask] = signals[1][signalmask]
    peaks = peakpos[0].copy()
    peaks[signalmask] = peakpos[1][signalmask]
                
    return combined, peaks


class CameraR0Calibrator(Component):
    """
    The base R0-level calibrator. Change the r0 container.
    The R0 calibrator performs the camera-specific R0 calibration that is
    usually performed on the raw data by the camera server. This calibrator
    exists in ctapipe for testing and prototyping purposes.
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

        self._load_calib()

    def subtract_pedestal(self, event):
        """
        Subtracts cell offset using pedestal file.
        Parameters
        ----------
        event : `ctapipe` event-container
        """
        n_modules = event.lst.tel[0].svc.num_modules
        expected_pixel_id = event.lst.tel[0].svc.pixel_ids
        for nr in range(0, n_modules):
            fc_cap = self._get_first_capacitor(event, nr)
            for gain in range(0, self.n_gain):
                for pix in range(0, self.n_pix):
                    pixel =  expected_pixel_id[nr*7 + pix]
                    position = int((fc_cap[gain, pix]) % self.size4drs)
                    event.r0.tel[0].waveform[gain, pixel, :] = \
                    (event.r0.tel[0].waveform[gain, pixel, :] -
                    self.pedestal_value_array[nr, gain, pix, position:position + 40])

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
