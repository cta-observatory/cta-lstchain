import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import tables

from lstchain.io import standard_config
from lstchain.io.io import dl1_params_lstcam_key, dl2_params_lstcam_key, dl1_images_lstcam_key
from lstchain.reco.utils import filter_events


test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
test_r0_path = test_data / 'real/R0/20200218/LST-1.1.Run02008.0000_first50.fits.fz'
test_r0_path2 = test_data / 'real/R0/20200218/LST-1.1.Run02008.0100_first50.fits.fz'
test_drs4_r0_path = test_data / 'real/R0/20200218/LST-1.1.Run02005.0000_first50.fits.fz'
test_calib_path = test_data / 'real/calibration/20200218/v05/calibration.Run2006.0000.hdf5'
test_drs4_pedestal_path = test_data / 'real/calibration/20200218/v05/drs4_pedestal.Run2005.0000.fits'
test_time_calib_path = test_data / 'real/calibration/20200218/v05/time_calibration.Run2006.0000.hdf5'
test_drive_report = test_data / 'real/monitoring/DrivePositioning/drive_log_20200218.txt'


def test_import_calib():
    from lstchain import calib


def test_import_reco():
    from lstchain import reco


def test_import_visualization():
    from lstchain import visualization


def test_import_lstio():
    from lstchain import io


@pytest.mark.run(order=1)
def test_r0_to_dl1(tmp_path, mc_gamma_testfile):
    from lstchain.reco.r0_to_dl1 import r0_to_dl1
    infile = mc_gamma_testfile
    r0_to_dl1(infile, custom_config=standard_config, output_filename=tmp_path / "dl1_gamma.h5")


@pytest.mark.private_data
def test_r0_to_dl1_observed(tmp_path):
    from lstchain.reco.r0_to_dl1 import r0_to_dl1

    output_path = tmp_path / ('dl1_' + test_r0_path.stem + '.h5')

    config = standard_config
    lst_event_source = config['source_config']['LSTEventSource']
    lst_event_source['PointingSource']['drive_report_path'] = test_drive_report
    lst_event_source['LSTR0Corrections']['drs4_pedestal_path'] = \
        test_drs4_pedestal_path
    lst_event_source['LSTR0Corrections']['calibration_path'] = \
        test_calib_path
    lst_event_source['LSTR0Corrections']['drs4_time_calibration_path']\
        = test_time_calib_path


    r0_to_dl1(
        test_r0_path,
        output_filename=output_path,
        custom_config=config
    )

    with tables.open_file(output_path, 'r') as f:
        images_table = f.root[dl1_images_lstcam_key]
        params_table = f.root[dl1_params_lstcam_key]
        assert 'image' in images_table.colnames
        assert 'peak_time' in images_table.colnames
        assert 'tel_id' in images_table.colnames
        assert 'obs_id' in images_table.colnames
        assert 'event_id' in images_table.colnames
        assert 'tel_id' in params_table.colnames
        assert 'event_id' in params_table.colnames
        assert 'obs_id' in params_table.colnames


@pytest.mark.private_data
def test_r0_available():
    assert test_r0_path.is_file()
    assert test_r0_path2.is_file()


@pytest.mark.run(after='test_r0_to_dl1')
def test_content_dl1(simulated_dl1_file):
    # test presence of images and parameters
    with tables.open_file(simulated_dl1_file, 'r') as f:
        images_table = f.root[dl1_images_lstcam_key]
        params_table = f.root[dl1_params_lstcam_key]
        assert 'image' in images_table.colnames
        assert 'peak_time' in images_table.colnames
        assert 'tel_id' in images_table.colnames
        assert 'obs_id' in images_table.colnames
        assert 'event_id' in images_table.colnames
        assert 'tel_id' in params_table.colnames
        assert 'event_id' in params_table.colnames
        assert 'obs_id' in params_table.colnames


def test_get_source_dependent_parameters(simulated_dl1_file):
    from lstchain.reco.dl1_to_dl2 import get_source_dependent_parameters

    dl1_params = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    src_dep_df = get_source_dependent_parameters(dl1_params, standard_config)


@pytest.mark.run(order=2)
def test_build_models(simulated_dl1_file, rf_models):
    from lstchain.reco.dl1_to_dl2 import build_models
    infile = simulated_dl1_file

    reg_energy, reg_disp, cls_gh = build_models(
        infile,
        infile,
        custom_config=standard_config,
        save_models=False
    )

    import joblib

    joblib.dump(reg_energy, rf_models["energy"])
    joblib.dump(reg_disp, rf_models["disp"])
    joblib.dump(cls_gh, rf_models["gh_sep"])


@pytest.mark.run(order=3)
def test_apply_models(simulated_dl1_file, simulated_dl2_file, rf_models):
    from lstchain.reco.dl1_to_dl2 import apply_models
    import joblib

    dl1 = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    dl1 = filter_events(
        dl1,
        filters=standard_config["events_filters"],
        finite_params=standard_config['regression_features'] + standard_config['classification_features']
    )

    reg_energy = joblib.load(rf_models["energy"])
    reg_disp = joblib.load(rf_models["disp"])
    reg_cls_gh = joblib.load(rf_models["gh_sep"])

    dl2 = apply_models(dl1, reg_cls_gh, reg_energy, reg_disp, custom_config=standard_config)
    dl2.to_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)


@pytest.fixture(scope="session")
def fake_dl2_proton_file(temp_dir_simulated_files, simulated_dl2_file):
    """
    Produce a fake dl2 proton file by copying the dl2 gamma test file
    and changing mc_type.
    """
    dl2_proton_file = temp_dir_simulated_files / 'dl2_fake_proton.simtel.h5'
    events = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    events.mc_type = 101
    events.to_hdf(dl2_proton_file, key=dl2_params_lstcam_key)
    return dl2_proton_file

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
    from lstchain.reco.utils import sky_to_camera, camera_to_altaz
    import astropy.units as u
    x = np.random.rand(1) * u.m
    y = np.random.rand(1) * u.m
    focal_length = 5 * u.m
    pointing_alt = np.pi/3. * u.rad
    pointing_az = 0. * u.rad

    sky_pos = camera_to_altaz(x, y, focal_length, pointing_alt, pointing_az)
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


def test_version_not_unknown():
    """
    Test that lstchain.__version__ is not unknown
    """
    import lstchain
    assert lstchain.__version__ != 'unknown'
