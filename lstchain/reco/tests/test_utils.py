from lstchain.reco import utils
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd


def test_camera_to_altaz():
    pos_x = np.array([0, 0]) * u.m
    pos_y = np.array([0, 0]) * u.m
    focal = 28*u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    sky_coords = utils.camera_to_altaz(pos_x, pos_y, focal, pointing_alt, pointing_az)
    np.testing.assert_allclose(sky_coords.alt, pointing_alt, rtol=1e-4)
    np.testing.assert_allclose(sky_coords.az, pointing_az, rtol=1e-4)

    # Test for real event with a time
    obs_time = Time('2018-11-01T02:00', '2018-11-01T02:00')
    sky_coords = utils.camera_to_altaz(pos_x, pos_y, focal, pointing_alt, pointing_az, obstime = obs_time)
    np.testing.assert_allclose(sky_coords.alt, pointing_alt, rtol=1e-4)
    np.testing.assert_allclose(sky_coords.az, pointing_az, rtol=1e-4)

def test_radec_to_camera():
    pointing_radec = SkyCoord.from_name('Crab')
    obstime = Time('2020-01-27T23:00', scale='utc')
    pointing_alt  = u.Quantity(1.3748, u.rad, copy=False)
    pointing_az = u.Quantity(4.0975, u.rad, copy=False)
    focal = 28*u.m
    expected_source_pos_camera = np.array([0.0, 0.0]) * u.m
    pointing_pos_camera = utils.radec_to_camera(pointing_radec, obstime, pointing_alt, pointing_az, focal)
    np.testing.assert_allclose(pointing_pos_camera.x.to_value(), expected_source_pos_camera[0].to_value(), atol=0.1)
    np.testing.assert_allclose(pointing_pos_camera.y.to_value(), expected_source_pos_camera[1].to_value(), atol=0.1)

def test_reco_source_position_sky():
    cog_x = np.array([2, 1]) * u.m
    cog_y = np.array([-1, 1]) * u.m
    disp_dx = np.array([-2, -1]) * u.m
    disp_dy = np.array([1, -1]) * u.m
    focal_length = 28 *u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    sky_coords = utils.reco_source_position_sky(cog_x, cog_y, disp_dx, disp_dy, focal_length, pointing_alt, pointing_az)
    np.testing.assert_allclose(sky_coords.alt, pointing_alt, rtol=1e-4)
    np.testing.assert_allclose(sky_coords.az, pointing_az, rtol=1e-4)


def test_sky_to_camera():
    alt = np.array([1, 1]) * u.rad
    az = np.array([0.2, 0.5]) * u.rad
    focal = 28*u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    camera_coords = utils.sky_to_camera(alt, az, focal, pointing_alt, pointing_az)
    np.testing.assert_allclose(camera_coords.x.value, np.array([0, 0]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(camera_coords.y.value, np.array([0, 0]), rtol=1e-4, atol=1e-4)


def test_linear_imputer():
    a = np.array([0.2, 0.3, np.nan, np.nan, np.nan, 0.7, np.nan, 0.8, np.nan])
    utils.linear_imputer(a, missing_values=np.nan, copy=False)
    np.testing.assert_allclose(a, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.8])


def test_impute_pointing():
    event_id = np.arange(5, 14, dtype=int)
    a = np.array([0.2, 0.3, np.nan, np.nan, np.nan, 0.7, np.nan, 0.8, np.nan])
    df = pd.DataFrame(np.transpose([event_id, a, a]), columns=['event_id', 'alt_tel', 'az_tel'])
    df = utils.impute_pointing(df)
    np.testing.assert_allclose(df.alt_tel, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.8])
    np.testing.assert_allclose(df.az_tel, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.8])


def test_unix_tai_to_utc():
    from lstchain.reco.utils import unix_tai_to_time, INVALID_TIME

    timestamp_tai = 1579376359.3225002
    leap_seconds = 37
    utc_time = unix_tai_to_time(timestamp_tai)

    assert np.isclose(utc_time.unix, timestamp_tai - leap_seconds)

    # test nan values
    assert unix_tai_to_time(np.nan) == INVALID_TIME

    # test multiple values including nans
    timestamps = np.array([timestamp_tai, np.nan])
    assert np.isclose(unix_tai_to_time(timestamps)[0].unix, timestamp_tai - leap_seconds)
    assert unix_tai_to_time(timestamps)[1] == INVALID_TIME
