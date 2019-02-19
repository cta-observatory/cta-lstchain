"""
Container structures for data that should be read or written to disk
"""

from astropy import units as u

from ctapipe.core import Container, Field, Map
from ctapipe.io.containers import *

__all__ = [
    'FlatFieldContainer',
    'PedestalContainer',
    'PixelStatusContainer',
    'MonitoringCameraContainer',
    'MonitoringContainer',
    'LSTCameraContainer',
    'LSTDataContainer'
]


class FlatFieldContainer(Container):
    """
    Container for flat-field parameters obtained from a set of
    [n_events] flat-field events
    """

    time = Field(0, 'Time associated to the flat-field event set ', unit=u.s)
    time_range = Field(
        [],
        'Range of time of the flat-field events [t_min, t_max]',
        unit=u.s
    )
    n_events = Field(0, 'Number of events used for statistics')
    relative_gain_mean = Field(
        None,
        "np array of the relative flat-field coefficient mean (n_chan X N_pix)"
    )
    charge_mean = Field(
        None,
        "np array of signal charge mean (n_chan X N_pix)"
    )
    charge_median = Field(
        None,
        "np array of signal charge median (n_chan X N_pix)"
    )
    charge_rms = Field(
        None,
        "np array of signal charge rms (n_chan X N_pix)"
    )
    time_mean = Field(
        None,
        "np array of signal time mean (n_chan X N_pix)",
        unit=u.ns,
    )
    time_median = Field(
        None,
        "np array of signal time median (n_chan X N_pix)",
        unit=u.ns
    )
    time_rms = Field(
        None,
        "np array of signal time rms (n_chan X N_pix)"
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


class PedestalContainer(Container):
    """
    Container for pedestal parameters obtained from a set of
    [n_pedestal] pedestal events
    """

    time = Field(0, 'Time associated to the pedestal event set', unit=u.s)
    time_range = Field(
        [],
        'Range of time of the pedestal events [t_min, t_max]',
        unit=u.s
    )
    n_event = Field(0, 'Number of events used for statistics')
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


class PixelStatusContainer(Container):
    """
    Container for pixel status information
    It contains masks obtained by several data analysis steps
    At r0 level only the hardware_mask is initialized
    """
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


class MonitoringCameraContainer(Container):
    """
    Container for camera monitoring data
    """

    flatfield = Field(FlatFieldContainer(), "Relative flat field data")
    pedestal = Field(PedestalContainer(), "Pedestal data")
    pixel_status = Field(PixelStatusContainer(), "Container of masks with pixel status")


class MonitoringContainer(Container):
    """
    Root container for monitoring data (MON)
    """

    tels_with_data = Field([], "list of telescopes with data")

    # create the camera container
    tel = Field(
        Map(MonitoringCameraContainer),
        "map of tel_id to MonitoringCameraContainer")


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
    mon = Field(MonitoringContainer(), "container for monitoring data (MON)")
