from lstchain.reco import utils
import astropy.units as u
import numpy as np





def test_camera_to_sky():
    pos_x = np.array([0, 0]) * u.m
    pos_y = np.array([0, 0]) * u.m
    focal = 28*u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    sky_coords = utils.camera_to_sky(pos_x, pos_y, focal, pointing_alt, pointing_az)
    np.testing.assert_allclose(sky_coords.alt, pointing_alt)
    np.testing.assert_allclose(sky_coords.az, pointing_az)


def test_reco_source_position_sky():
    cog_x = np.array([2, 1]) * u.m
    cog_y = np.array([-1, 1]) * u.m
    disp_dx = np.array([-2, -1]) * u.m
    disp_dy = np.array([1, -1]) * u.m
    focal_length = 28 *u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    sky_coords = utils.reco_source_position_sky(cog_x, cog_y, disp_dx, disp_dy, focal_length, pointing_alt, pointing_az)
    np.testing.assert_allclose(sky_coords.alt, pointing_alt)
    np.testing.assert_allclose(sky_coords.az, pointing_az)


def test_sky_to_camera():
    alt = np.array([1, 1]) * u.rad
    az = np.array([0.2, 0.5]) * u.rad
    focal = 28*u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    camera_coords = utils.sky_to_camera(alt, az, focal, pointing_alt, pointing_az)
    np.testing.assert_allclose(camera_coords.x.value, np.array([0, 0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(camera_coords.y.value, np.array([0, 0]), rtol=1e-5, atol=1e-5)

