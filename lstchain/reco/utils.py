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

import logging
from warnings import warn

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.time import Time
from astropy.utils import deprecated
from ctapipe.coordinates import CameraFrame

from . import disp

__all__ = [
    'add_delta_t_key',
    'alt_to_theta',
    'az_to_phi',
    'cal_cam_source_pos',
    'camera_to_altaz',
    'cartesian_to_polar',
    'clip_alt',
    'compute_alpha',
    'compute_theta2',
    'expand_tel_list',
    'extract_source_position',
    'filter_events',
    'get_effective_time',
    'get_event_pos_in_camera',
    'impute_pointing',
    'linear_imputer',
    'polar_to_cartesian',
    'predict_source_position_in_camera',
    'radec_to_camera',
    'reco_source_position_sky',
    'rotate',
    'sky_to_camera',
    'source_dx_dy',
    'source_side',
]

# position of the LST1
location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)
obstime = Time('2018-11-01T02:00')
horizon_frame = AltAz(location=location, obstime=obstime)


log = logging.getLogger(__name__)
def rotate(flat_object, degree=0, origin=(0, 0)):
    """
    Rotate 2D object around given axle

    Parameters:
    -----------
    array-like flat_object: 2D object to rotate
    tuple origin: rotation axle coordinates
    int degree: rotation angle in degrees

    Returns:
    --------
    NDArray with new coordinates
    """
    angle = np.deg2rad(degree)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotation_axis_coordinates = np.asarray(origin)
    res = [(rotation_matrix @ (point.T-rotation_axis_coordinates.T) +
                       rotation_axis_coordinates.T).T for point in np.atleast_2d(flat_object)]
    return res


def extract_source_position(data, observed_source_name, equivalent_focal_length = 28*u.m):
    """
    Extract source position from data

    Parameters:
    -----------
    pandas.DataFrame data: input data
    str observed_source_name: Name of the observed source
    astropy.units.m equivalent_focal_length: Equivalent focal length of a telescope

    Returns:
    --------
    2D array of coordinates of the source in form [(x),(y)] in astropy.units.m
    """
    observed_source = SkyCoord.from_name(observed_source_name)
    obstime = pd.to_datetime(data['dragon_time'], unit='s')
    pointing_alt = u.Quantity(data['alt_tel'], u.rad, copy=False)
    pointing_az = u.Quantity(data['az_tel'], u.rad, copy=False)
    source_pos_camera = radec_to_camera(observed_source, obstime,
                                        pointing_alt, pointing_az, focal=equivalent_focal_length)
    source_position = [source_pos_camera.x, source_pos_camera.y]
    return source_position


def compute_theta2(data, source_position, conversion_factor=2.0):
    """
    Computes a square of theta (angle from z-axis) from camera frame coordinates

    Parameters:
    -----------
    pandas.DataFrame data: Input data
    2D array (x,y) source_position: Observed source position in astropy.units.m
    float conversion_factor: Conversion factor (default 0.1/0.05 deg/m)

    Returns:
    -------
    Array with `theta2` values
    """
    reco_src_x = np.array(data['reco_src_x']) * u.m
    reco_src_y = np.array(data['reco_src_y']) * u.m
    return conversion_factor**2 * ((source_position[0] - reco_src_x)**2 +
                                   (source_position[1] - reco_src_y)**2)


def compute_alpha(data):
    """
    Computes the angle between the shower major axis and polar angle of the shower centroid

    Parameters:
    -----------
    pandas.DataFrame data: Input data

    Returns:
    --------
    Array with `alpha` values
    """
    # phi and psi range [-np.pi, +np.pi]
    alpha = np.mod(data['phi'] - data['psi'], np.pi)  # alpha in [0, np.pi]
    alpha = np.minimum(np.pi - alpha, alpha)  # put alpha in [0, np.pi/2]

    return np.rad2deg(alpha)


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
    event: `ctapipe.containers.ArrayEventContainer`
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
                               leakage_intensity_width_2=[0, 1],
                               ),
                  finite_params=None,
                  ):
    """
    Apply data filtering to a pandas dataframe or astropy Table.
    The Table object will be converted to pandas dataframe and used.
    Each filtering range is applied if the column name exists in the DataFrame so that
    `(events >= range[0]) & (events <= range[1])`
    Returning filter is converted to a numpy object so that it can be used by both dataframe
    and table inputs

    Parameters
    ----------
    events: `pandas.DataFrame` or 'astropy.table.Table'
    filters: dict containing events features names and their filtering range
    finite_params: optional, None or list of strings
        extra filter to ensure finite parameters

    Returns
    -------
    `pandas.DataFrame` or 'astropy.table.Table'
    """
    from astropy.table import Table

    if isinstance(events, Table):
        events_df = events.to_pandas()
    else:
        events_df = events

    filter = np.ones(len(events_df), dtype=bool)

    for col, (lower_limit, upper_limit) in filters.items():
        filter &= (events_df[col] >= lower_limit) & (events_df[col] <= upper_limit)
        
    if finite_params is not None:
        _finite_params = list(set(finite_params).intersection(list(events_df.columns)))
        with pd.option_context('mode.use_inf_as_null', True):
            not_finite_mask = events_df[_finite_params].isnull()
        filter &= ~(not_finite_mask.any(axis=1))

        not_finite_counts = (not_finite_mask).sum(axis=0)[_finite_params]
        if (not_finite_counts > 0).any():
            log.warning('Data contains not-predictable events.')
            log.warning('Column | Number of non finite values')
            for k, v in not_finite_counts.items():
                if v > 0:
                    log.warning(f'{k} : {v}')

    return events[filter.to_numpy()]



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

def add_delta_t_key(events):
    """
    Adds the time difference with the previous event to a real data
    dataframe.
    Should be only used only with non-filtered data frames,
    so events are consecutive.
    Parameters
    ----------
    events: pandas DataFrame of dl1 events

    Returns
    -------
    events: pandas DataFrame of dl1 events with delta_t
    """

    # Get delta t of real data and add it to the data frame
    if 'dragon_time' in events.columns:
        time= np.array(events.dragon_time)
        delta_t = np.insert(np.diff(time),0,0)
        events['delta_t'] = delta_t
    return events

def get_effective_time(events):
    """
    Calculate the effective observation time of a set of real data events
    from a sky observation. delta_t (s) must be the time elapsed from the
    previous *triggered* event, regardless of whether the list of events
    contains all triggered events or not. It can be a list only of events
    which e.g. have valid image parameters. Besides delta_t, each event must
    have dragon_time, a timestamp (s)

    Parameters
    ----------
    events: pandas DataFrame or astropy.table.QTable
    If a dataframe, units are assumed to be seconds

    Returns
    -------
    t_eff: astropy Quantity (in seconds, if input has no units)
    t_elapsed: astropy Quantity (ditto)
    """
    timestamp = np.array(events['dragon_time'])
    delta_t = np.array(events['delta_t'])

    if not isinstance(timestamp, u.Quantity):
        timestamp *= u.s
    if not isinstance(delta_t, u.Quantity):
        delta_t *= u.s

    # time differences between the events in the table (which in general are
    # NOT all triggered events):
    time_diff = np.diff(timestamp)

    # elapsed time: sum of those time differences, excluding large ones which
    # might indicate the DAQ was stopped (e.g. if the table contains more
    # than one run). We set 0.1 s as limit to decide a "break" occurred:
    t_elapsed = np.sum(time_diff[time_diff<0.1*u.s])

    # delta_t is the time elapsed since the previous triggered event.
    # We exclude the null values that might be set for the first even in a file.
    delta_t = delta_t[delta_t>0.]

    # dead time per event (minimum observed delta_t, ):
    dead_time = np.amin(delta_t)

    # Estimate the "true external rate", i.e. what we would see in absence of
    # dead time. For a Poisson process with fixed dead time per event,
    # it can be shown that the expected value of delta_t is
    # <delta_t> = dead_time + 1/rate
    # Note that the formula is not strictly correct if we have interleaved
    # events (pedestal and flatfield) at regular intervals, because the
    # delta_t will never be larger than the time between interleaved
    # events. But this truncation would hardly be noticeable for the typical
    # cosmics rates, and 200 Hz of interleaved events.

    rate = 1/(np.mean(delta_t) - dead_time)

    t_eff = t_elapsed / (1 + rate*dead_time)

    return t_eff, t_elapsed
