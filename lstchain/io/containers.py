"""
Container structures for data that should be read or written to disk
"""

from astropy import units as u
#from astropy.time import Time
#from numpy import nan
#import numpy as np

from ctapipe.core import Container, Field, Map
from ctapipe.io.containers import *

__all__ = [
    'FlatFieldCameraContainer',
    'FlatFieldContainer',
    'PedestalCameraContainer',
    'PedestalContainer',
    'PixelStatusCameraContainer',
    'PixelStatusContainer',
    'LSTCameraContainer',
    'MonitoringDataContainer',
    'LSTDataContainer'
]


class R0CameraContainer(Container):
    """
    Storage of raw data from a single telescope
    """
    trigger_time = Field(None, "Telescope trigger time, start of waveform "
                               "readout, None for MCs")
    trigger_type = Field(0o0, "camera's event trigger type if applicable")
    num_trig_pix = Field(0, "Number of trigger groups (sectors) listed")
    trig_pix_id = Field(None, "pixels involved in the camera trigger")
    image = Field(None, (
        "numpy array containing integrated ADC data "
        "(n_channels x n_pixels) DEPRECATED"
    ))  # to be removed, since this doesn't exist in real data and useless in mc
    waveform = Field(None, (
        "numpy array containing ADC samples"
        "(n_channels x n_pixels, n_samples)"
    ))
    num_samples = Field(None, "number of time samples for telescope")
    pixel_status = Field(0o0, "status of the pixels")


class FlatFieldCameraContainer(Container):
    """
    Container for relative camera flat-field coefficients

    """

    time_mean = Field(0, 'Mean time, seconds since reference', unit=u.s)
    time_range = Field(
        [],
        'Range of time of the calibration data [t_min, t_max]',
        unit=u.s
    )
    n_events = Field(0, 'Number of events used for statistics')
    relative_gain_mean = Field(
        None,
        "np array of the relative flat-field coefficient mean (n_chan X N_pix)"
    )
    relative_gain_median = Field(
        None,
        "np array of the relative flat-field coefficient  median (n_chan X N_pix)"
    )
    relative_gain_rms = Field(
        None,
        "np array of the relative flat-field coefficient rms (n_chan X N_pix)"
    )
    relative_time_mean = Field(
        None,
        "np array of the relative time mean (n_chan X N_pix)",
        unit=u.ns
    )
    relative_time_median = Field(
        None,
        "np array of the relative time  median (n_chan X N_pix)",
        unit=u.ns)


class FlatFieldContainer(Container):
    """
    Container for relative flat field coefficients
    """
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(
        Map(FlatFieldCameraContainer),
        "map of tel_id to FlatFiledCameraContainer")


class PedestalCameraContainer(Container):
    """
    Container for pedestals per camera
    """
    time_mean = Field(0, 'Mean time, seconds since reference', unit=u.s)
    time_range = Field(
        [],
        'Range of time of the calibration data [t_min, t_max]',
        unit=u.s
    )
    n_events = Field(0, 'Number of events used for statistics')
    pedestal_mean = Field(
        None,
        "np array of pedestal average (n_chan X N_pix)"
    )
    pedestal_median = Field(
        None,
        "np array of the pedestal  median (n_chan X N_pix)"
    )
    pedestal_rms = Field(
        None,
        "np array of the pedestal rms (n_chan X N_pix)"
    )
    relative_pedestal_mean = Field(
        None,
        "np array of relative pedestal average (n_chan X N_pix)"
    )
    relative_pedestal_median = Field(
        None,
        "np array of the relative pedestal  median (n_chan X N_pix)"
    )
    relative_pedestal_rms = Field(
        None,
        "np array of the relative pedestal rms (n_chan X N_pix)"
    )


class PedestalContainer(Container):
    """
    Container for pedestal data
    """
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(
        Map(PedestalCameraContainer),
        "map of tel_id to PedestalCameraContainer")


class PixelStatusCameraContainer(Container):
    hardware_mask = Field(
        None,
        "Mask from the hardware pixel status data (N_pix)"
    )

    pedestal_mask = Field(
        None,
        "Mask from the pedestal data analysis (N_pix)"
    )

    flatfield_mask = Field(
        None,
        "Mask from the flat-flield data analysis (N_pix)"
    )


class PixelStatusContainer(Container):
    """
    Container for pedestal data
    """
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(
        Map(PixelStatusCameraContainer),
        "map of tel_id to PixelStatusContainer")


class MonitoringDataContainer(Container):
    """
    Root container for MON data
    """
    flatfield = Field(FlatFieldContainer(), "Relative flat field data")
    pedestal = Field(PedestalContainer(), "Pedestal data")
    pixel_status = Field(PixelStatusContainer(), "Container of masks with pixel status")



class LSTServiceContainer(Container):
    """
    Container for Fields that are specific to each LST camera configuration
    """

    # Data from the CameraConfig table
    telescope_id = Field(-1, "telescope id")
    cs_serial = Field(None, "serial number of the camera server")
    configuration_id = Field(None, "id of the CameraConfiguration")
    date = Field(None, "NTP start of run date")
    num_pixels = Field(-1, "number of pixels")
    num_samples = Field(-1, "num samples")
    pixel_ids = Field([], "id of the pixels in the waveform array")
    data_model_version = Field(None, "data model version")

    idaq_version = Field(0o0, "idaq version")
    cdhs_version = Field(0o0, "cdhs version")
    algorithms = Field(None, "algorithms")
    pre_proc_algorithms = Field(None, "pre processing algorithms")
    module_ids = Field([], "module ids")
    num_modules = Field(-1, "number of modules")


class LSTEventContainer(Container):
    """
    Container for Fields that are specific to each LST event
    """

    # Data from the CameraEvent table
    configuration_id = Field(None, "id of the CameraConfiguration")
    event_id = Field(None, "local id of the event")
    tel_event_id = Field(None, "global id of the event")
    pixel_status = Field([], "status of the pixels (n_pixels)")
    ped_id = Field(None, "tel_event_id of the event used for pedestal substraction")
    module_status = Field([], "status of the modules (n_modules)")
    extdevices_presence = Field(None, "presence of data for external devices")

    tib_event_counter = Field(None, "TIB event counter")
    tib_pps_counter = Field(None, "TIB pps counter")
    tib_tenMHz_counter = Field(None, "TIB 10 MHz counter")
    tib_stereo_pattern = Field(None, "TIB stereo pattern")
    tib_masked_trigger = Field(None, "TIB trigger mask")

    ucts_event_counter =  Field(None, "UCTS event counter")
    ucts_pps_counter = Field(None, "UCTS pps counter")
    ucts_clock_counter = Field(None, "UCTS clock counter")
    ucts_timestamp = Field(None, "UCTS timestamp")
    ucts_camera_timestamp = Field(None, "UCTS camera timestamp")
    ucts_trigger_type = Field(None, "UCTS trigger type")
    ucts_white_rabbit_status = Field(None, "UCTS whiteRabbit status")

    #cdts_data = Field([], "CDTS data array")
    swat_data = Field([], "SWAT data array")

    pps_counter= Field([], "Dragon pulse per second counter (n_modules)")
    tenMHz_counter = Field([], "Dragon 10 MHz counter (n_modules)")
    event_counter = Field([], "Dragon event counter (n_modules)")
    trigger_counter = Field([], "Dragon trigger counter (n_modules)")
    local_clock_counter = Field([], "Dragon local 133 MHz counter (n_modules)")

    chips_flags = Field([], "chips flags")
    first_capacitor_id = Field([], "first capacitor id")
    drs_tag_status = Field([], "DRS tag status")
    drs_tag = Field([], "DRS tag")


class LSTCameraContainer(Container):
    """
    Container for Fields that are specific to each LST camera
    """
    evt = Field(LSTEventContainer(), "LST specific event Information")
    svc = Field(LSTServiceContainer(), "LST specific camera_config Information")


class LSTContainer(Container):
    """
    Storage for the LSTCameraContainer for each telescope
    """
    tels_with_data = Field([], "list of telescopes with data")

    # create the camera container
    tel = Field(
        Map(LSTCameraContainer),
        "map of tel_id to LSTTelContainer")


class LSTDataContainer(DataContainer):
    """
    Data container including LST and monitoring information
    """
    lst = Field(LSTContainer(), "LST specific Information")
    mon = Field(MonitoringDataContainer(), "container for MON data")
