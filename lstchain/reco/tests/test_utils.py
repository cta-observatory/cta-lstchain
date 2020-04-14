from lstchain.reco import utils
import astropy.units as u
from astropy.time import Time
import numpy as np
import pandas as pd


def test_camera_to_sky():
    pos_x = np.array([0, 0]) * u.m
    pos_y = np.array([0, 0]) * u.m
    focal = 28*u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    sky_coords = utils.camera_to_sky(pos_x, pos_y, focal, pointing_alt, pointing_az)
    np.testing.assert_allclose(sky_coords.alt, pointing_alt, rtol=1e-4)
    np.testing.assert_allclose(sky_coords.az, pointing_az, rtol=1e-4)


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
    timestamp_tai = 1579376359.3225002
    leap_seconds = 37
    utc_time = utils.unix_tai_to_utc(timestamp_tai)
    np.testing.assert_allclose(utc_time.unix, timestamp_tai - leap_seconds, rtol=1e-12)
