"""
Functions to handle custom containers for the mono reconstruction of LST1
"""

import astropy.units as u
from astropy.units import Quantity
import numpy as np
from ctapipe.core import Container, Field
from ctapipe.image import timing_parameters as time
from ..reco.utils import disp, get_event_pos_in_camera


class DL1ParametersContainer(Container):
    """
    TODO: maybe fields could be inherited from ctapipe containers definition
        For now I have not found an elegant way to do so
    """
    intensity = Field(None, 'total intensity (size)')

    x = Field(None, 'centroid x coordinate', unit=u.m)
    y = Field(None, 'centroid x coordinate', unit=u.m)
    r = Field(None, 'radial coordinate of centroid', unit=u.m)
    phi = Field(None, 'polar coordinate of centroid', unit=u.rad)
    length = Field(None, 'RMS spread along the major-axis', unit=u.m)
    width = Field(None, 'RMS spread along the minor-axis', unit=u.m)
    psi = Field(None, 'rotation angle of ellipse', unit=u.rad)
    skewness = Field(None, 'measure of the asymmetry')
    kurtosis = Field(None, 'measure of the tailedness')
    disp = Field(None, 'disp [m]', unit=u.m)
    hadroness = Field(None, 'hadroness')
    src_x = Field(None, 'source x coordinate in camera frame', unit=u.m)
    src_y = Field(None, 'source y coordinate in camera frame', unit=u.m)
    time_gradient = Field(None, 'Time gradient in the camera')
    intercept = Field(None, 'Intercept')

    obs_id = Field(None, 'Observation ID')
    event_id = Field(None, 'Event ID')
    gps_time = Field(None, 'GPS time event trigger')

    mc_energy = Field(None, 'Simulated Energy', unit=u.TeV)
    mc_alt = Field(None, 'Simulated altitude', unit=u.rad)
    mc_az = Field(None, 'Simulated azimuth', unit=u.rad)
    mc_core_x = Field(None, 'Simulated impact point x position', unit=u.m)
    mc_core_y = Field(None, 'Simulated impact point y position', unit=u.m)
    mc_h_first_int = Field(None, 'Simulated first interaction height', unit=u.m)
    mc_type = Field(None, 'Simulated particle type')
    mc_az_tel = Field(None, 'Telescope altitude pointing', unit=u.rad)
    mc_alt_tel = Field(None, 'Telescope azimuth pointing', unit=u.rad)
    mc_x_max = Field(None, "MC Xmax value", unit=u.g / (u.cm ** 2))
    mc_core_distance = Field(None, "Distance from the impact point to the telescope", unit=u.m)
    mc_shower_primary_id = Field(None, "MC shower primary ID 0 (gamma), 1(e-),"
                                    "2(mu-), 100*A+Z for nucleons and nuclei,"
                                    "negative for antimatter.")

    hadroness = Field(None, "Hadroness")
    wl = Field(None, "width/length")

    def fill_hillas(self, hillas):
        """
        fill Hillas parameters

        hillas: HillasContainer
        # TODO : parameters should not be simply copied but inherited
        (e.g. conserving unit definition)
        """
        for k in hillas.keys():
            try:
                self[k] = hillas[k]
            except:
                print("{} cannot be copied in container".format(k))
                pass

    def fill_mc(self, event):
        """
        fill from mc
        """
        try:
            self.mc_energy = event.mc.energy
            self.mc_alt = event.mc.alt
            self.mc_az = event.mc.az
            self.mc_core_x = event.mc.core_x
            self.mc_core_y = event.mc.core_y
            self.mc_h_first_int = event.mc.h_first_int
            # mcType = event.mc. # TODO: find type in event
            self.mc_x_max = event.mc.x_max
            self.mc_alt_tel = event.mcheader.run_array_direction[1]
            self.mc_az_tel = event.mcheader.run_array_direction[0]
        except:
            print("mc information not filled")

    def fill_event_info(self, event):
        try:
            self.gps_time = event.trig.gps_time
            self.obs_id = event.r0.obs_id
            self.event_id = event.r0.event_id
        except:
            print("event information not filled")

    def get_features(self, features_names):
        return np.array([self[k].value if isinstance(self[k], Quantity) else self[k]
                         for k in features_names])

    def set_mc_core_distance(self, event, telescope_id):
        tel_pos = event.inst.subarray.positions[telescope_id]
        distance = np.sqrt((event.mc.core_x - tel_pos[0]) ** 2 + (event.mc.core_y - tel_pos[1]) ** 2)
        self.mc_core_distance = distance

    def set_disp(self, source_pos, hillas):
        self.disp = disp(source_pos, hillas)

    def set_timing_features(self, geom, image, peakpos, hillas):
        try:
            peak_time = Quantity(peakpos) * u.Unit("ns")
            timepars = time.timing_parameters(geom, image, peak_time, hillas)
            self.time_gradient = timepars['slope'].value
            self.intercept = timepars['intercept']
        except:
            pass

    def set_source_camera_position(self, event, telescope_id):
        # sourcepos = utils.cal_cam_source_pos(mc_alt, mc_az,
        #                                      mc_alt_tel, mc_az_tel,
        #                                      focal_length)
        # self.src_x = sourcepos[0]
        # self.src_y = sourcepos[1]
        tel = event.inst.subarray.tel[telescope_id]
        source_pos = get_event_pos_in_camera(event, tel)
        self.src_x = source_pos[0]
        self.src_y = source_pos[1]


