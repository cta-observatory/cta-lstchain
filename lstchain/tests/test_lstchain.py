from ctapipe.utils import get_dataset_path
import numpy as np
import pytest
import os
import pandas as pd
from lstchain.io.io import dl1_params_lstcam_key, dl2_params_lstcam_key
from lstchain.reco.utils import filter_events

test_dir = 'testfiles'

os.makedirs(test_dir, exist_ok=True)

mc_gamma_testfile = get_dataset_path('gamma_test_large.simtel.gz')
dl1_file = os.path.join(test_dir, 'dl1_gamma_test_large.simtel.h5')
dl2_file = os.path.join(test_dir, 'dl2_gamma_test_large.simtel.h5')
file_model_energy = os.path.join(test_dir, 'reg_energy.sav')
file_model_disp = os.path.join(test_dir, 'reg_disp_vector.sav')
file_model_gh_sep = os.path.join(test_dir, 'cls_gh.sav')

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
        "picture_thresh":6,
        "boundary_thresh":2,
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
    from lstchain import calib

def test_import_reco():
    from lstchain import reco

def test_import_visualization():
    from lstchain import visualization

def test_import_lstio():
    from lstchain import io

@pytest.mark.run(order=1)
def test_dl0_to_dl1():
    from lstchain.reco.dl0_to_dl1 import r0_to_dl1
    infile = mc_gamma_testfile
    r0_to_dl1(infile, custom_config=custom_config, output_filename=dl1_file)

@pytest.mark.run(order=2)
def test_build_models():
    from lstchain.reco.dl1_to_dl2 import build_models
    infile = dl1_file

    reg_energy, reg_disp, cls_gh = build_models(infile, infile, custom_config=custom_config, save_models=False)

    from sklearn.externals import joblib
    joblib.dump(reg_energy, file_model_energy)
    joblib.dump(reg_disp, file_model_disp)
    joblib.dump(cls_gh, file_model_gh_sep)


@pytest.mark.run(order=3)
def test_apply_models():
    from lstchain.reco.dl1_to_dl2 import apply_models
    from sklearn.externals import joblib

    dl1 = pd.read_hdf(dl1_file, key=dl1_params_lstcam_key)
    dl1 = filter_events(dl1, filters=custom_config["events_filters"])

    reg_energy = joblib.load(file_model_energy)
    reg_disp = joblib.load(file_model_disp)
    reg_cls_gh = joblib.load(file_model_gh_sep)

    dl2 = apply_models(dl1, reg_cls_gh, reg_energy, reg_disp, custom_config=custom_config)
    dl2.to_hdf(dl2_file, key=dl2_params_lstcam_key)


@pytest.mark.last
def test_clean_test_files():
    """
    Function to clean the test files created by the previous test
    """
    import shutil
    shutil.rmtree(test_dir)


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
    assert lstchain.__version__ is not 'unknown'
