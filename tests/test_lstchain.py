from ctapipe.utils import get_dataset_path
import numpy as np

def test_import_calib():
    from lstchain import calib

def test_import_reco():
    from lstchain import reco

def test_import_visualization():
    from lstchain import visualization

def test_import_lstio():
    from lstchain import io

def test_dl0_to_dl1():
    from lstchain.reco.dl0_to_dl1 import r0_to_dl1
    infile = get_dataset_path('gamma_test_large.simtel.gz')
    r0_to_dl1(infile)

def test_build_models():
    from lstchain.reco.dl1_to_dl2 import build_models
    infile = 'dl1_gamma_test_large.h5'
    features = ['intensity', 'width', 'length']

    reg_energy, reg_disp, cls_gh = build_models(
        infile, infile,
        features,
        save_models=True)

    from sklearn.externals import joblib
    joblib.dump(reg_energy, 'rf_energy.pkl')
    joblib.dump(reg_disp, 'rf_disp.pkl')
    joblib.dump(cls_gh, 'rf_cls_gh.pkl')


def test_apply_models():
    from lstchain.reco.dl1_to_dl2 import apply_models
    import pandas as pd
    from sklearn.externals import joblib

    dl1_file = 'dl1_gamma_test_large.h5'
    dl1 = pd.read_hdf(dl1_file)
    features = ['intensity', 'width', 'length']
    # Load the trained RF for reconstruction:
    file_energy = 'rf_energy.pkl'
    file_disp = 'rf_disp.pkl'
    file_cls_gh = 'rf_cls_gh.pkl'

    reg_energy = joblib.load(file_energy)
    reg_disp = joblib.load(file_disp)
    reg_cls_gh = joblib.load(file_cls_gh)

    apply_models(dl1, features, reg_cls_gh, reg_energy, reg_disp)


def test_clean_test_files():
    """
    Function to clean the test files created by the previous test
    """
    import os
    os.remove('dl1_gamma_test_large.h5')
    os.remove('cls_gh.sav')
    os.remove('reg_disp.sav')
    os.remove('reg_energy.sav')
    os.remove('rf_disp.pkl')
    os.remove('rf_energy.pkl')
    os.remove('rf_cls_gh.pkl')


def test_disp_vector():
    from lstchain.reco.utils import disp_vector
    dx = np.cos(np.pi/3 * np.ones(3))
    dy = np.sin(np.pi/3 * np.ones(3))
    disp_angle = np.pi/3 * np.ones(3)
    disp_norm = np.ones(3)
    disp_sign = np.ones(3)
    disp_dx, disp_dy = disp_vector(disp_norm, disp_angle, disp_sign)
    np.testing.assert_array_equal([dx, dy], [disp_dx, disp_dy])

def test_disp_to_pos():
    from lstchain.reco.utils import disp_to_pos
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
    np.testing.assert_almost_equal([x, y], [cam_pos.x, cam_pos.y])

