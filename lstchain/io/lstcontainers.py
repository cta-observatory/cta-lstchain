"""
Functions to handle custom containers for the mono reconstruction of LST1
"""

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from ctapipe.core import Container, Field
from ctapipe.image import leakage_parameters as leakage
from ctapipe.image import concentration_parameters as concentration
from ctapipe.image import timing_parameters
from ctapipe.image.morphology import number_of_islands
from numpy import nan

from ..reco.disp import disp_parameters_event

__all__ = [
    'DL1MonitoringEventIndexContainer',
    'DL1ParametersContainer',
    'DispContainer',
    'ExtraImageInfo',
    'ExtraMCInfo',
    'ExtraMCInfo',
    'LSTEventType',
    'MetaData',
    'ThrownEventsHistogram',
]


class DL1ParametersContainer(Container):
    """
    TODO: maybe fields could be inherited from ctapipe containers definition
        For now I have not found an elegant way to do so
    """
    intensity = Field(np.float64(np.nan), 'total intensity (size)')
    log_intensity = Field(np.float64(np.nan), 'log of total intensity (size)')

    x = Field(u.Quantity(np.nan, u.m), 'centroid x coordinate', unit=u.m)
    y = Field(u.Quantity(np.nan, u.m), 'centroid x coordinate', unit=u.m)
    r = Field(u.Quantity(np.nan, u.m), 'radial coordinate of centroid', unit=u.m)
    phi = Field(Angle(np.nan, u.rad), 'polar coordinate of centroid',
                unit=u.rad)
    length = Field(u.Quantity(np.nan, u.deg), 'RMS spread along the major-axis',
                   unit=u.deg)
    width = Field(u.Quantity(np.nan, u.deg), 'RMS spread along the minor-axis',
                             unit=u.deg)
    psi = Field(Angle(np.nan, u.rad), 'rotation angle of ellipse', unit=u.rad)

    skewness = Field(np.nan, 'measure of the asymmetry')
    kurtosis = Field(np.nan, 'measure of the tailedness')
    disp_norm = Field(None, 'disp_norm [m]', unit=u.m)
    disp_dx = Field(None, 'disp_dx [m]', unit=u.m)
    disp_dy = Field(None, 'disp_dy [m]', unit=u.m)
    disp_angle = Field(None, 'disp_angle [rad]', unit=u.rad)
    disp_sign = Field(None, 'disp_sign')
    disp_miss = Field(None, 'disp_miss [m]', unit=u.m)
    src_x = Field(None, 'source x coordinate in camera frame', unit=u.m)
    src_y = Field(None, 'source y coordinate in camera frame', unit=u.m)
    time_gradient = Field(np.nan, 'Time gradient in the camera')
    intercept = Field(np.nan, 'Intercept')
    leakage_intensity_width_1 = \
        Field(np.float32(np.nan), 'Fraction of intensity in outermost pixels',
              dtype=np.float32)
    leakage_intensity_width_2 = \
        Field(np.float32(np.nan), 'Fraction of intensity in two outermost '
                                  'rings of pixels', dtype=np.float32)
    leakage_pixels_width_1 = Field(np.nan, 'Fraction of signal pixels that are '
                                           'border pixels')
    leakage_pixels_width_2 = Field(np.nan, 'Fraction of signal pixels that are '
                                           'in the two outermost rings of pixels')
    n_pixels = Field(-1, 'Number of pixels after cleaning')
    concentration_cog = Field(np.nan, 'Fraction of intensity in three pixels '
                                      'closest to the cog')
    concentration_core = Field(np.nan, 'Fraction of intensity inside hillas '
                                       'ellipse')
    concentration_pixel = Field(np.nan, 'Fraction of intensity in brightest '
                                        'pixel')
    n_islands = Field(-1, 'Number of Islands')
    alt_tel = Field(None, 'Telescope altitude pointing',
                    unit=u.rad)
    az_tel = Field(None, 'Telescope azimuth pointing',
                   unit=u.rad)

    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    calibration_id = Field(-1, 'ID of the employed calibration event')
    dragon_time = Field(None, 'Dragon time event trigger')
    ucts_time = Field(None, 'UCTS time event trigger')
    tib_time = Field(None, 'TIB time event trigger')

    mc_energy = Field(None, 'Simulated Energy', unit=u.TeV)
    log_mc_energy = Field(None, 'log of simulated energy/TeV')
    mc_alt = Field(None, 'Simulated altitude', unit=u.rad)
    mc_az = Field(None, 'Simulated azimuth', unit=u.rad)
    mc_core_x = Field(None, 'Simulated impact point x position', unit=u.m)
    mc_core_y = Field(None, 'Simulated impact point y position', unit=u.m)
    mc_h_first_int = Field(None, 'Simulated first interaction height', unit=u.m)
    mc_type = Field(-9999, "MC shower primary ID 0 (gamma), 1(e-),"
                           "2(mu-), 100*A+Z for nucleons and nuclei,"
                           "negative for antimatter.")
    mc_az_tel = Field(None, 'Telescope MC azimuth pointing', unit=u.rad)
    mc_alt_tel = Field(None, 'Telescope MC altitude pointing', unit=u.rad)
    mc_x_max = Field(None, "MC Xmax value", unit=u.g / u.cm**2)
    mc_core_distance = Field(None, "Distance from the impact point to the telescope", unit=u.m)

    hadroness = Field(None, "Hadroness")
    wl = Field(u.Quantity(np.nan), "width/length")

    tel_id = Field(None, "Telescope Id")
    tel_pos_x = Field(None, "Telescope x position in the ground")
    tel_pos_y = Field(None, "Telescope y position in the ground")
    tel_pos_z = Field(None, "Telescope z position in the ground")

    trigger_type = Field(None, "trigger type")
    ucts_trigger_type = Field(None, "UCTS trigger type")
    trigger_time = Field(None, "trigger time")
    event_type = Field(None, "event type")

    # info not available in data
    #num_trig_pix = Field(None, "Number of trigger groups (sectors) listed")
    #trig_pix_id = Field(None, "pixels involved in the camera trigger")

    def fill_hillas(self, hillas):
        """
        fill Hillas parameters

        hillas: HillasParametersContainer
        # TODO : parameters should not be simply copied but inherited
        (e.g. conserving unit definition)
        """
        for key in hillas.keys():
            if key in self.keys():
                self[key] = hillas[key]

    def fill_mc(self, event, tel_pos):
        """
        fill from mc
        """
        shower = event.simulation.shower
        try:
            self.mc_energy = shower.energy
            self.log_mc_energy = np.log10(self.mc_energy.to_value(u.TeV))
            self.mc_alt = shower.alt
            self.mc_az = shower.az
            self.mc_core_x = shower.core_x
            self.mc_core_y = shower.core_y
            self.mc_h_first_int = shower.h_first_int
            self.mc_x_max = shower.x_max
            self.mc_alt_tel = event.pointing.array_altitude
            self.mc_az_tel = event.pointing.array_azimuth
            self.mc_type = shower.shower_primary_id
            distance = np.sqrt(
                (shower.core_x - tel_pos[0]) ** 2 +
                (shower.core_y - tel_pos[1]) ** 2
            )
            if np.isfinite(distance):
                self.mc_core_distance = distance

        except IndexError:
            print("mc information not filled")

    def fill_event_info(self, event):
        self.obs_id = event.index.obs_id
        self.event_id = event.index.event_id

    def get_features(self, features_names):
        return np.array([
            self[k].value
            if isinstance(self[k], Quantity)
            else self[k]
            for k in features_names
        ])

    def set_disp(self, source_pos, hillas):
        disp = disp_parameters_event(hillas, source_pos[0], source_pos[1])
        self.disp_norm = disp.norm
        self.disp_dx = disp.dx
        self.disp_dy = disp.dy
        self.disp_angle = disp.angle
        self.disp_sign = disp.sign
        self.disp_miss = disp.miss

    def set_timing_features(self, geom, image, peak_time, hillas):
        try:    # if np.polyfit fails (e.g. len(image) < deg + 1)
            timepars = timing_parameters(geom, image, peak_time, hillas)
            self.time_gradient = timepars.slope.value
            self.intercept = timepars.intercept
        except ValueError:
            self.time_gradient = np.nan
            self.intercept = np.nan

    def set_leakage(self, geom, image, clean):
        leakage_c = leakage(geom, image, clean)
        self.leakage_intensity_width_1 = leakage_c.intensity_width_1
        self.leakage_intensity_width_2 = leakage_c.intensity_width_2
        self.leakage_pixels_width_1 = leakage_c.pixels_width_1
        self.leakage_pixels_width_2 = leakage_c.pixels_width_2

    def set_concentration(self, geom, image, hillas_parameters):
        conc = concentration(geom, image, hillas_parameters)
        self.concentration_cog = conc.cog
        self.concentration_core = conc.core
        self.concentration_pixel = conc.pixel

    def set_n_islands(self, geom, clean):
        n_islands, islands_mask = number_of_islands(geom, clean)
        self.n_islands = n_islands

    def set_telescope_info(self, subarray, telescope_id):
        self.tel_id = telescope_id
        tel_pos = subarray.positions[telescope_id]
        self.tel_pos_x = tel_pos[0]
        self.tel_pos_y = tel_pos[1]
        self.tel_pos_z = tel_pos[2]

    def set_source_camera_position(self, event, telescope):
        source_pos = utils.get_event_pos_in_camera(event, telescope)
        self.src_x = source_pos[0]
        self.src_y = source_pos[1]




class DispContainer(Container):
    """
    Disp vector container
    """
    dx = Field(nan, 'x coordinate of the disp_norm vector')
    dy = Field(nan, 'y coordinate of the disp_norm vector')

    angle = Field(nan, 'Angle between the X axis and the disp_norm vector')
    norm = Field(nan, 'Norm of the disp_norm vector')
    sign = Field(nan, 'Sign of the disp_norm')
    miss = Field(nan, 'miss parameter norm')


class ExtraMCInfo(Container):
    obs_id = Field(0, "MC Run Identifier")

class ExtraImageInfo(Container):
    """ attach the tel_id """
    tel_id = Field(0, "Telescope ID")
    selected_gain_channel = Field(None, "Selected gain channel")


class ThrownEventsHistogram(Container):
    """ 2D histogram from SimTel files """
    obs_id = Field(-1, 'MC run ID')
    hist_id = Field(-1, 'Histogram ID')
    num_entries = Field(-1, 'Number of entries in the histogram')
    bins_energy = Field(None, 'array of energy bin lower edges, as in np.histogram')
    bins_core_dist = Field(None, 'array of core-distance bin lower edges, as in np.histogram')
    histogram = Field(None, "array of histogram entries, size (n_bins_x, n_bins_y)")

    def fill_from_simtel(self, hist):
        """ fill from a SimTel Histogram entry"""
        self.hist_id = hist['id']
        self.num_entries = hist['entries']
        xbins = np.linspace(hist['lower_x'], hist['upper_x'], hist['n_bins_x'] + 1)
        ybins = np.linspace(hist['lower_y'], hist['upper_y'], hist['n_bins_y'] + 1)
        self.bins_core_dist = xbins
        self.bins_energy = 10 ** ybins
        self.histogram = hist['data']
        self.meta['hist_title'] = hist['title']
        self.meta['x_label'] = 'Log10 E (TeV)'
        self.meta['y_label'] = '3D Core Distance (m)'


class MetaData(Container):
    """
    Some metadata
    """
    SOURCE_FILENAMES = Field([], "filename of the source file")
    LSTCHAIN_VERSION = Field(None, "version of lstchain")
    CTAPIPE_VERSION = Field(None, "version of ctapipe")
    CONTACT = Field(None, "Person or institution responsible for this data product")


class DL1MonitoringEventIndexContainer(Container):
    """
    Container with the calibration coefficients
    """
    tel_id = Field(1, 'Index of telescope')
    calibration_id = Field(-1, 'Index of calibration event for DL1 file')
    pedestal_id = Field(-1, 'Index of pedestal event for DL1 file')
    flatfield_id = Field(-1, 'Index of flat-field event for DL1 file')


class LSTEventType:
    """
    Class to recognize event type from trigger bits
    bit 0: Mono
    bit 1: stereo
    bit 2: Calibration
    bit 3: Single Phe
    bit 4: Softrig(from the UCTS)
    bit 5: Pedestal
    bit 6: slow control
    bit 7: busy
    """

    @staticmethod
    def is_mono(trigger_type):

        return trigger_type >> 0 & 1 and trigger_type != -1

    @staticmethod
    def is_stereo(trigger_type):
        return trigger_type >> 1 & 1 and trigger_type != -1

    @staticmethod
    def is_calibration(trigger_type):
        return trigger_type >> 2 & 1 and trigger_type != -1

    @staticmethod
    def is_single_pe(trigger_type):
        return trigger_type >> 3 & 1  and trigger_type != -1

    @staticmethod
    def is_soft_trig(trigger_type):
        return trigger_type >> 4 & 1  and trigger_type != -1

    @staticmethod
    def is_pedestal(trigger_type):
        return trigger_type >> 5 & 1 and trigger_type != -1

    @staticmethod
    def is_slow_control(trigger_type):
        return trigger_type >> 6 & 1  and trigger_type != -1

    @staticmethod
    def is_busy(trigger_type):
        return trigger_type >> 7 & 1  and trigger_type != -1

    @staticmethod
    def is_unknown(trigger_type):
        return trigger_type == -1
