#!/usr/bin/env python3
"""Module with auxiliar functions:
Transform AltAz coordinates into Camera coordinates (This should be
implemented already in ctapipe but I haven't managed to find how to
do it)
Calculate source position from disp_norm distance.
Calculate disp_ distance from source position.

Usage:

"import utils"
"""

import numpy as np
from ctapipe.coordinates import CameraFrame
import astropy.units as u
from astropy.utils import deprecated
from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.time import Time
from . import disp
from warnings import warn
import pandas as pd
import logging

__all__ = [
    'alt_to_theta',
    'az_to_phi',
    'cal_cam_source_pos',
    'get_event_pos_in_camera',
    'reco_source_position_sky',
    'camera_to_altaz',
    'sky_to_camera',
    'radec_to_camera',
    'source_side',
    'source_dx_dy',
    'polar_to_cartesian',
    'cartesian_to_polar',
    'predict_source_position_in_camera',
    'expand_tel_list',
    'filter_events',
    'linear_imputer',
    'impute_pointing',
    'clip_alt',
    'unix_tai_to_time',
]

# position of the LST1
location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)
obstime = Time('2018-11-01T02:00')
horizon_frame = AltAz(location=location, obstime=obstime)
UCTS_EPOCH = Time('1970-01-01T00:00:00', scale='tai', format='isot')
INVALID_TIME = UCTS_EPOCH


def alt_to_theta(alt):
    """Transforms altitude (angle from the horizon upwards) to theta
    (angle from z-axis) for simtel array coordinate systems
    Parameters:
    -----------
    alt: float

    Returns:
    --------
    float: theta

    """

    return (90 * u.deg - alt).to(alt.unit)


def az_to_phi(az):
    """Transforms azimuth (angle from north towards east)
    to phi (angle from x-axis towards y-axis)
    for simtel array coordinate systems
    Parameters:
    -----------
    az: float

    Returns:
    --------
    az: float
    """
    return -az


@deprecated("09/07/2019", message="This is a custom implementation. Use `sky_to_camera` that relies on astropy")
def cal_cam_source_pos(mc_alt, mc_az, mc_alt_tel, mc_az_tel, focal_length):
    """Transform Alt-Az source position into Camera(x,y) coordinates
    source position.

    Parameters:
    -----------
    mc_alt: float
    Alt coordinate of the event

    mc_az: float
    Az coordinate of the event

    mc_alt_tel: float
    Alt coordinate of the telescope pointing

    mc_az_tel: float
    Az coordinate of the telescope pointing

    focal_length: float
    Focal length of the telescope

    Returns:
    --------
    float: source_x1,

    float: source_x2
    """

    mc_alt = alt_to_theta(mc_alt*u.rad).value
    mc_az = az_to_phi(mc_az*u.rad).value
    mc_alt_tel = alt_to_theta(mc_alt_tel*u.rad).value
    mc_az_tel = az_to_phi(mc_az_tel*u.rad).value

    # Sines and cosines of direction angles
    cp = np.cos(mc_az)
    sp = np.sin(mc_az)
    ct = np.cos(mc_alt)
    st = np.sin(mc_alt)

    # Shower direction coordinates
    sourcex = st*cp
    sourcey = st*sp
    sourcez = ct

    source = np.array([sourcex, sourcey, sourcez])
    source = source.T

    # Rotation matrices towars the camera frame
    rot_Matrix = np.empty((0, 3, 3))

    alttel = mc_alt_tel
    aztel = mc_az_tel
    mat_Y = np.array([[np.cos(alttel), 0, np.sin(alttel)],
                      [0, 1, 0],
                      [-np.sin(alttel), 0, np.cos(alttel)]]).T

    mat_Z = np.array([[np.cos(aztel), -np.sin(aztel), 0],
                      [np.sin(aztel), np.cos(aztel), 0],
                      [0, 0, 1]]).T

    rot_Matrix = np.matmul(mat_Y, mat_Z)

    res = np.einsum("...ji,...i", rot_Matrix, source)
    res = res.T

    source_x = -focal_length*res[0]/res[2]
    source_y = -focal_length*res[1]/res[2]
    return source_x, source_y


def get_event_pos_in_camera(event, tel):
    """
    Return the position of the source in the camera frame
    Parameters
    ----------
    event: `ctapipe.io.containers.DataContainer`
    tel: `ctapipe.instruement.telescope.TelescopeDescription`

    Returns
    -------
    (x, y) (float, float): position in the camera
    """

    array_pointing = SkyCoord(alt=clip_alt(event.mcheader.run_array_direction[1]),
                              az=event.mcheader.run_array_direction[0],
                              frame=horizon_frame)

    event_direction = SkyCoord(alt=clip_alt(event.mc.alt),
                               az=event.mc.az,
                               frame=horizon_frame)

    focal = tel.optics.equivalent_focal_length

    camera_frame = CameraFrame(focal_length=focal,
                               telescope_pointing=array_pointing)

    camera_pos = event_direction.transform_to(camera_frame)
    return camera_pos.x, camera_pos.y


def reco_source_position_sky(cog_x, cog_y, disp_dx, disp_dy, focal_length, pointing_alt, pointing_az):
    """
    Compute the reconstructed source position in the sky

    Parameters
    ----------
    cog_x: `astropy.units.Quantity`
    cog_y: `astropy.units.Quantity`
    disp_dx: `astropy.units.Quantity`
    disp_dy: `astropy.units.Quantity`
    focal_length: `astropy.units.Quantity`
    pointing_alt: `astropy.units.Quantity`
    pointing_az: `astropy.units.Quantity`

    Returns
    -------
    sky frame: `astropy.coordinates.sky_coordinate.SkyCoord`
    """
    src_x, src_y = disp.disp_to_pos(disp_dx, disp_dy, cog_x, cog_y)
    return camera_to_altaz(src_x, src_y, focal_length, pointing_alt, pointing_az)


def camera_to_altaz(pos_x, pos_y, focal, pointing_alt, pointing_az, obstime = None):
    """
    Compute camera to Horizontal frame (Altitude-Azimuth system). For MC assume the default ObsTime.

    Parameters
    ----------
    pos_x: `~astropy.units.Quantity`
        X coordinate in camera (distance)
    pos_y: `~astropy.units.Quantity`
        Y coordinate in camera (distance)
    focal: `~astropy.units.Quantity`
        telescope focal (distance)
    pointing_alt: `~astropy.units.Quantity`
        pointing altitude in angle unit
    pointing_az: `~astropy.units.Quantity`
        pointing altitude in angle unit
    obstime: `~astropy.time.Time`


    Returns
    -------
    sky frame: `astropy.coordinates.SkyCoord`
       in AltAz frame
    Example:
    --------
    import astropy.units as u
    import numpy as np
    pos_x = np.array([0, 0]) * u.m
    pos_y = np.array([0, 0]) * u.m
    focal = 28*u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    sky_coords = utils.camera_to_altaz(pos_x, pos_y, focal, pointing_alt, pointing_az)

    """
    if not obstime:
        logging.info("No time given. To be use only for MC data.")
    horizon_frame = AltAz(location=location, obstime=obstime)

    pointing_direction = SkyCoord(alt=clip_alt(pointing_alt), az=pointing_az, frame=horizon_frame)

    camera_frame = CameraFrame(focal_length=focal, telescope_pointing=pointing_direction)

    camera_coord = SkyCoord(pos_x, pos_y, frame=camera_frame)

    horizon = camera_coord.transform_to(horizon_frame)

    return horizon


def sky_to_camera(alt, az, focal, pointing_alt, pointing_az):
    """
    Coordinate transform from aky position (alt, az) (in angles) to camera coordinates (x, y) in distance
    Parameters
    ----------
    alt: astropy Quantity
    az: astropy Quantity
    focal: astropy Quantity
    pointing_alt: pointing altitude in angle unit
    pointing_az: pointing altitude in angle unit

    Returns
    -------
    camera frame: `astropy.coordinates.sky_coordinate.SkyCoord`
    """
    pointing_direction = SkyCoord(alt=clip_alt(pointing_alt), az=pointing_az, frame=horizon_frame)

    camera_frame = CameraFrame(focal_length=focal, telescope_pointing=pointing_direction)

    event_direction = SkyCoord(alt=clip_alt(alt), az=az, frame=horizon_frame)

    camera_pos = event_direction.transform_to(camera_frame)

    return camera_pos

def radec_to_camera(sky_coordinate, obstime, pointing_alt, pointing_az, focal):
    """
    Coordinate transform from sky coordinate to camera coordinates (x, y) in distance
    Parameters
    ----------
    sky_coordinate: astropy.coordinates.sky_coordinate.SkyCoord
    obstime: astropy.time.Time
    pointing_alt: pointing altitude in angle unit
    pointing_az: pointing altitude in angle unit
    focal: astropy Quantity
    
    Returns
    -------
    camera frame: `astropy.coordinates.sky_coordinate.SkyCoord`
    """   
    
    horizon_frame = AltAz(location=location, obstime=obstime)

    pointing_direction = SkyCoord(alt=clip_alt(pointing_alt), az=pointing_az, frame=horizon_frame)

    camera_frame = CameraFrame(focal_length=focal, telescope_pointing=pointing_direction, obstime=obstime, location=location)

    camera_pos = sky_coordinate.transform_to(camera_frame)

    return camera_pos

def source_side(source_pos_x, cog_x):
    """
    Compute on what side of the center of gravity the source is in the camera.

    Parameters
    ----------
    source_pos_x: X coordinate of the source in the camera, float
    cog_x: X coordinate of the center of gravity, float

    Returns
    -------
    float: -1 or +1
    """
    return np.sign(source_pos_x - cog_x)


def source_dx_dy(source_pos_x, source_pos_y, cog_x, cog_y):
    """
    Compute the coordinates of the vector (dx, dy) from the center of gravity to the source position

    Parameters
    ----------
    source_pos_x: X coordinate of the source in the camera
    source_pos_y: Y coordinate of the source in the camera
    cog_x: X coordinate of the center of gravity in the camera
    cog_y: Y coordinate of the center of gravity in the camera

    Returns
    -------
    (dx, dy)
    """
    return source_pos_x - cog_x, source_pos_y - cog_y


def polar_to_cartesian(norm, angle, sign):
    """
    Polar to cartesian transformation.
    As a convention, angle should be in [-pi/2:pi/2].

    Parameters
    ----------
    norm: float or `numpy.ndarray`
    angle: float or `numpy.ndarray`
    sign: float or `numpy.ndarray`

    Returns
    -------

    """
    assert np.isfinite([norm, angle, sign]).all()
    x = norm * sign * np.cos(angle)
    y = norm * sign * np.sin(angle)
    return x, y


def cartesian_to_polar(x, y):
    """
    Cartesian to polar transformation
    As a convention, angle is always included in [-pi/2:pi/2].
    When the angle should be in [pi/2:3*pi/2], angle = -1

    Parameters
    ----------
    x: float or `numpy.ndarray`
    y: float or `numpy.ndarray`

    Returns
    -------
    norm, angle, sign
    """
    norm = np.sqrt(x**2 + y**2)
    if x == 0:
        angle = np.pi/2. * np.sign(y)
    else:
        angle = np.arctan(y/x)
    sign = np.sign(x)
    return norm, angle, sign


def predict_source_position_in_camera(cog_x, cog_y, disp_dx, disp_dy):
    """
    Compute the source position in the camera frame

    Parameters
    ----------
    cog_x: float or `numpy.ndarray` - x coordinate of the center of gravity (hillas.x)
    cog_y: float or `numpy.ndarray` - y coordinate of the center of gravity (hillas.y)
    disp_dx: float or `numpy.ndarray`
    disp_dy: float or `numpy.ndarray`

    Returns
    -------
    source_pos_x, source_pos_y
    """
    reco_src_x = cog_x + disp_dx
    reco_src_y = cog_y + disp_dy
    return reco_src_x, reco_src_y


def expand_tel_list(tel_list, max_tels):
    """
    transform for the telescope list (to turn it into a telescope pattern)
    un-pack var-length list of tel_ids into
    fixed-width bit pattern by tel_index
    """
    pattern = np.zeros(max_tels).astype(bool)
    pattern[tel_list] = 1
    return pattern


def filter_events(events,
                  filters=dict(intensity=[0, np.inf],
                                 width=[0, np.inf],
                                 length=[0, np.inf],
                                 wl=[0, np.inf],
                                 r=[0, np.inf],
                                 leakage2_intensity=[0, 1],
                                 ),
                  dropna=True,
                  ):
    """
    Apply data filtering to a pandas dataframe.
    Each filtering range is applied if the column name exists in the DataFrame so that
    `(events >= range[0]) & (events <= range[1])`
    If the column name does not exist, the filtering is simply not applied

    Parameters
    ----------
    events: `pandas.DataFrame`
    filters: dict containing events features names and their filtering range
    dropna: bool
        if True (default), `dropna()` is applied to the dataframe.

    Returns
    -------
    `pandas.DataFrame`
    """

    filter = np.ones(len(events), dtype=bool)

    for k in filters.keys():
        if k in events.columns:
            filter = filter & (events[k] >= filters[k][0]) & (events[k] <= filters[k][1])

    if dropna:
        with pd.option_context('mode.use_inf_as_null', True):
            return events[filter].dropna()
    else:
        return events[filter]


def linear_imputer(y, missing_values=np.nan, copy=True):
    """
    Replace missing values in y with values from a linear interpolation on their position in the array.
    Parameters
    ----------
    y: list or `numpy.array`
    missing_values: number, string, np.nan or None, default=`np.nan`
        The placeholder for the missing values. All occurrences of `missing_values` will be imputed.
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will be done in-place whenever possible.
    Returns
    -------
    `numpy.array` : array with `missing_values` imputed
    """
    x = np.arange(len(y))
    if missing_values is np.nan:
        mask_missing = np.isnan(y)
    else:
        mask_missing = y == missing_values
    imputed_values = np.interp(x[mask_missing], x[~mask_missing], y[~mask_missing])
    if copy:
        yy = np.copy(y)
        yy[mask_missing] = imputed_values
        return yy
    else:
        y[mask_missing] = imputed_values
        return y


def impute_pointing(dl1_data, missing_values=np.nan):
    """
    Impute missing pointing values using `linear_imputer` and replace them inplace
    Parameters
    ----------
    dl1_data: `pandas.DataFrame`
    missing_values: number, string, np.nan or None, default=`np.nan`
        The placeholder for the missing values. All occurrences of `missing_values` will be imputed.
    """
    if len(set(dl1_data.event_id)) != len(dl1_data.event_id):
        warn("Beware, the data has been resorted by `event_id` to interpolate invalid pointing values but there are "
             "several events with the same `event_id` in the data, thus probably leading to unexpected behaviour",
             UserWarning)
    dl1_data = dl1_data.sort_values(by='event_id')
    for k in ['alt_tel', 'az_tel']:
        dl1_data[k] = linear_imputer(dl1_data[k].values, missing_values=missing_values)
    return dl1_data


def clip_alt(alt):
    """
    Make sure altitude is not larger than 90 deg (it happens in some MC files for zenith=0),
    to keep astropy happy
    """
    return np.clip(alt, -90.*u.deg, 90.*u.deg)


def unix_tai_to_time(timestamp):
    """
    Create an astropy.Time object for timestamps in unix tai format.
    Unix tai format mean seconds since 1970-01-01T00:00 TAI as opposed
    to 1970-01-01T00:00 UTC for the usual unix timestamps.
    """
    scalar = np.isscalar(timestamp)

    timestamp = u.Quantity(timestamp, u.s, ndmin=1)
    invalid = ~np.isfinite(timestamp)
    timestamp[invalid] = 0

    t = UCTS_EPOCH + timestamp

    if scalar:
        return t[0]

    return t
