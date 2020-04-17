from ctapipe.utils import get_dataset_path
import numpy as np
import pytest
import tempfile
import pandas as pd
from lstchain.io.io import dl1_params_lstcam_key, dl2_params_lstcam_key
from lstchain.reco.utils import filter_events
import astropy.units as u
from pathlib import Path


mc_gamma_testfile = Path(get_dataset_path('gamma_test_large.simtel.gz'))


@pytest.fixture(scope='session')
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


custom_config = {
    "events_filters": {
        "intensity": [0.3, np.inf],
        "width": [0, 10],
        "length": [0, 10],
        "wl": [0, 1],
        "r": [0, 1],
        "leakage": [0, 1]
    },
    "tailcut": {
        "picture_thresh": 6,
        "boundary_thresh": 2,
        "keep_isolated_pixels": True,
        "min_number_picture_neighbors": 1
    },
    "random_forest_regressor_args": {
        "max_depth": 5,
        "min_samples_leaf": 2,
        "n_jobs": 4,
        "n_estimators": 15,
    },
    "random_forest_classifier_args": {
        "max_depth": 5,
        "min_samples_leaf": 2,
        "n_jobs": 4,
        "n_estimators": 10,
    },
    "regression_features": [
        "intensity",
        "width",
        "length",
        "x",
        "y",
        "wl",
        "skewness",
        "kurtosis",
    ],
    "classification_features": [
        "intensity",
        "width",
        "length",
        "x",
        "y",
        "log_reco_energy",
        "reco_disp_dx",
        "reco_disp_dy"
    ],

    "allowed_tels": [1, 2, 3, 4],
    "image_extractor": "GlobalPeakWindowSum",
    "image_extractor_config": {},
    "gain_selector": "ThresholdGainSelector",
    "gain_selector_config": {
        "threshold":  4094
    }
}


def test_import_calib():
    from lstchain import calib  # noqa


def test_import_reco():
    from lstchain import reco  # noqa


def test_import_visualization():
    from lstchain import visualization  # noqa


def test_import_lstio():
    from lstchain import io  # noqa


@pytest.mark.run(order=1)
def test_r0_to_dl1(temp_dir):
    from lstchain.reco.r0_to_dl1 import r0_to_dl1
    r0_to_dl1(
        str(mc_gamma_testfile),
        custom_config=custom_config,
        output_filename=temp_dir / ('dl1_' + mc_gamma_testfile.stem + '.h5')
    )


@pytest.mark.run(order=2)
def test_build_models(temp_dir):
    from lstchain.reco.dl1_to_dl2 import build_models
    infile = temp_dir / ('dl1_' + mc_gamma_testfile.stem + '.h5')

    reg_energy, reg_disp, cls_gh = build_models(
        infile, infile, custom_config=custom_config, save_models=False
    )

    from sklearn.externals import joblib
    joblib.dump(reg_energy, temp_dir / 'energy.pkl')
    joblib.dump(reg_disp, temp_dir / 'disp.pkl')
    joblib.dump(cls_gh, temp_dir / 'gh.pkl')


@pytest.mark.run(order=3)
def test_apply_models(temp_dir):
    from lstchain.reco.dl1_to_dl2 import apply_models
    from sklearn.externals import joblib

    dl1_file = temp_dir / ('dl1_' + mc_gamma_testfile.stem + '.h5')
    dl2_file = dl1_file.with_name(dl1_file.name.replace('dl1', 'dl2', 1))

    dl1 = pd.read_hdf(dl1_file, key=dl1_params_lstcam_key)
    dl1 = filter_events(dl1, filters=custom_config["events_filters"])

    reg_energy = joblib.load(temp_dir / 'energy.pkl')
    reg_disp = joblib.load(temp_dir / 'disp.pkl')
    reg_cls_gh = joblib.load(temp_dir / 'gh.pkl')

    dl2 = apply_models(dl1, reg_cls_gh, reg_energy, reg_disp, custom_config=custom_config)
    dl2.to_hdf(dl2_file, key=dl2_params_lstcam_key)


def produce_fake_dl2_proton_file(temp_dir):
    """
    Produce a fake dl2 proton file by copying the dl2 gamma test file
    and changing mc_type
    """
    dl1_file = temp_dir / ('dl1_' + mc_gamma_testfile.stem + '.h5')
    dl2_file = dl1_file.with_name(dl1_file.name.replace('dl1', 'dl2', 1))
    events = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    events.mc_type = 101
    events.to_hdf(temp_dir / 'dl2_fake_protons.h5', key=dl2_params_lstcam_key)


@pytest.mark.run(after='produce_fake_dl2_proton_file')
def test_sensitivity(temp_dir):
    from lstchain.mc.sensitivity import find_best_cuts_sensitivity, sensitivity 

    produce_fake_dl2_proton_file(temp_dir)

    eb = 10  # Number of energy bins
    gb = 11  # Number of gammaness bins
    tb = 10  # Number of theta2 bins
    obstime = 50 * 3600 * u.s
    noff = 2

    dl1_file = temp_dir / ('dl1_' + mc_gamma_testfile.stem + '.h5')
    dl2_file = dl1_file.with_name(dl1_file.name.replace('dl1', 'dl2', 1))
    fake_dl2_proton_file = temp_dir / 'dl2_fake_protons.h5'

    E, best_sens, result, units, gcut, tcut = find_best_cuts_sensitivity(
        dl1_file,
        dl1_file,
        dl2_file,
        fake_dl2_proton_file,
        1, 1,
        eb, gb, tb, noff,
        obstime,
    )

    E, best_sens, result, units, dl2 = sensitivity(
        dl1_file,
        dl1_file,
        dl2_file,
        fake_dl2_proton_file,
        1, 1,
        eb, gcut, tcut * (u.deg ** 2), noff,
        obstime,
    )


def test_disp_vector():
    from lstchain.reco.disp import disp_vector
    dx = np.cos(np.pi/3 * np.ones(3))
    dy = np.sin(np.pi/3 * np.ones(3))
    disp_angle = np.pi/3 * np.ones(3)
    disp_norm = np.ones(3)
    disp_sign = np.ones(3)
    disp_dx, disp_dy = disp_vector(disp_norm, disp_angle, disp_sign)
    np.testing.assert_array_equal([dx, dy], [disp_dx, disp_dy])


def test_disp_to_pos():
    from lstchain.reco.disp import disp_to_pos
    x = np.random.rand(3)
    y = np.random.rand(3)
    cog_x = np.random.rand(3)
    cog_y = np.random.rand(3)
    X, Y = disp_to_pos(x, y, cog_x, cog_y)
    np.testing.assert_array_equal([X, Y], [x+cog_x, y+cog_y])


def test_change_frame_camera_sky():
    from lstchain.reco.utils import sky_to_camera, camera_to_sky
    import astropy.units as u
    x = np.random.rand(1) * u.m
    y = np.random.rand(1) * u.m
    focal_length = 5 * u.m
    pointing_alt = np.pi/3. * u.rad
    pointing_az = 0. * u.rad

    sky_pos = camera_to_sky(x, y, focal_length, pointing_alt, pointing_az)
    cam_pos = sky_to_camera(sky_pos.alt, sky_pos.az, focal_length, pointing_alt, pointing_az)
    np.testing.assert_almost_equal([x, y], [cam_pos.x, cam_pos.y], decimal=4)


def test_polar_cartesian():
    from lstchain.reco.utils import polar_to_cartesian, cartesian_to_polar
    X = [-0.5, 0.5]
    Y = [-0.5, 0.5]
    for x in X:
        for y in Y:
            p = cartesian_to_polar(x, y)
            np.testing.assert_almost_equal((x, y), polar_to_cartesian(*p))


def test_version_not_unkown():
    """
    Test that lstchain.__version__ is not unkown
    """
    import lstchain
    assert lstchain.__version__ != 'unknown'
