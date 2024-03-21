import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import tables
from copy import deepcopy

from lstchain.io import standard_config, srcdep_config
from lstchain.io.io import dl1_params_lstcam_key, dl2_params_lstcam_key, dl1_images_lstcam_key
from lstchain.reco.utils import filter_events
from lstchain.reco.dl1_to_dl2 import build_models

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()
test_r0_path = test_data / 'real/R0/20200218/LST-1.1.Run02008.0000_first50.fits.fz'
test_r0_path2 = test_data / 'real/R0/20200218/LST-1.1.Run02008.0100_first50.fits.fz'
test_drs4_r0_path = test_data / 'real/R0/20200218/LST-1.1.Run02005.0000_first50.fits.fz'

calib_path = test_data / 'real/monitoring/PixelCalibration/Cat-A'
calib_version = 'ctapipe-v0.17'
test_calib_path = calib_path / f'calibration/20200218/{calib_version}/calibration_filters_52.Run02006.0000.h5'
test_drs4_pedestal_path = calib_path / f'drs4_baseline/20200218/{calib_version}/drs4_pedestal.Run02005.0000.h5'
test_time_calib_path = calib_path / f'drs4_time_sampling_from_FF/20191124/{calib_version}/time_calibration.Run01625.0000.h5'
test_drive_report = test_data / 'real/monitoring/DrivePositioning/DrivePosition_log_20200218.txt'
test_run_summary_path = test_data / 'real/monitoring/RunSummary/RunSummary_20200218.ecsv'
test_systematics_path = test_data / 'real/monitoring/PixelCalibration/Cat-A/ffactor_systematics/20200725/ctapipe-v0.17/ffactor_systematics_20200725.h5'


def test_r0_to_dl1(tmp_path, mc_gamma_testfile):
    from lstchain.reco.r0_to_dl1 import r0_to_dl1

    infile = mc_gamma_testfile
    r0_to_dl1(
        infile,
        custom_config=standard_config,
        output_filename=tmp_path / "dl1_gamma.h5"
    )


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


def test_r0_to_dl1_lhfit_mc(tmp_path, mc_gamma_testfile):
    from lstchain.reco.r0_to_dl1 import r0_to_dl1
    config = deepcopy(standard_config)
    config['source_config']['EventSource']['max_events'] = 5
    config['source_config']['EventSource']['allowed_tels'] = [1]
    config['lh_fit_config'] = {
        "sigma_s": [
            ["type", "*", 1.0],
            ["type", "LST_LST_LSTCam", 0.3282]
        ],
        "crosstalk": [
            ["type", "*", 0.0],
            ["type", "LST_LST_LSTCam", 0.0]
        ],
        "spatial_selection": "hillas",
        "dvr_pic_threshold": 8,
        "dvr_pix_for_full_image": 500,
        "sigma_space": 3,
        "sigma_time": 4,
        "time_before_shower": [
            ["type", "*", 0.0],
            ["type", "LST_LST_LSTCam", 0.0]
        ],
        "time_after_shower": [
            ["type", "*", 20.0],
            ["type", "LST_LST_LSTCam", 20.0]
        ],
        "n_peaks": 20,
        "no_asymmetry": False,
        "use_interleaved": False,
        "verbose": 4
    }
    os.makedirs('./event', exist_ok=True)
    r0_to_dl1(mc_gamma_testfile, custom_config=config, output_filename=tmp_path / "tmp.h5")
    assert len(os.listdir('./event')) > 1
    for path in os.listdir('./event'):
        os.remove('./event/'+path)
    os.rmdir('./event')
    os.remove(tmp_path / "tmp.h5")

    config['source_config']['EventSource']['allowed_tels'] = [1]
    config['lh_fit_config']["no_asymmetry"] = True
    config['lh_fit_config']["verbose"] = 0
    r0_to_dl1(mc_gamma_testfile, custom_config=config, output_filename=tmp_path / "tmp.h5")
    os.remove(tmp_path / "tmp.h5")
    config['lh_fit_config']["spatial_selection"] = 'dvr'
    config['lh_fit_config']["use_interleaved"] = True
    config['waveform_nsb_tuning']['nsb_tuning'] = True
    r0_to_dl1(mc_gamma_testfile, custom_config=config, output_filename=tmp_path / "tmp.h5")


@pytest.mark.private_data
def test_r0_to_dl1_lhfit_observed(tmp_path):
    from lstchain.reco.r0_to_dl1 import r0_to_dl1
    config = deepcopy(standard_config)
    config['source_config']['EventSource']['max_events'] = None
    config['source_config']['EventSource']['allowed_tels'] = [1]
    config['lh_fit_config'] = {
        "sigma_s": [
            ["type", "*", 1.0],
            ["type", "LST_LST_LSTCam", 0.3282]
        ],
        "crosstalk": [
            ["type", "*", 0.0],
            ["type", "LST_LST_LSTCam", 0.0]
        ],
        "spatial_selection": "hillas",
        "dvr_pic_threshold": 8,
        "dvr_pix_for_full_image": 500,
        "sigma_space": 3,
        "sigma_time": 4,
        "time_before_shower": [
            ["type", "*", 0.0],
            ["type", "LST_LST_LSTCam", 0.0]
        ],
        "time_after_shower": [
            ["type", "*", 20.0],
            ["type", "LST_LST_LSTCam", 20.0]
        ],
        "n_peaks": 0,
        "no_asymmetry": False,
        # test data doesn't contain interleaved events
        "use_interleaved": False,
        "verbose": 0
    }
    r0_to_dl1(test_r0_path, custom_config=config, output_filename=tmp_path / "tmp2.h5")
    os.remove(tmp_path / "tmp2.h5")
    config['lh_fit_config']["spatial_selection"] = 'dvr'
    r0_to_dl1(test_r0_path, custom_config=config, output_filename=tmp_path / "tmp2.h5")


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


def test_get_source_dependent_parameters_mc(simulated_dl1_file):
    from lstchain.reco.dl1_to_dl2 import get_source_dependent_parameters

    # for gamma MC
    dl1_params = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    src_dep_df_gamma = get_source_dependent_parameters(dl1_params, srcdep_config)

    # for proton MC
    dl1_params.mc_type = 101
    src_dep_df_proton = get_source_dependent_parameters(dl1_params, srcdep_config)

    assert 'alpha' in src_dep_df_gamma['on'].columns
    assert 'dist' in src_dep_df_gamma['on'].columns
    assert 'time_gradient_from_source' in src_dep_df_gamma['on'].columns
    assert 'skewness_from_source' in src_dep_df_gamma['on'].columns
    assert (src_dep_df_gamma['on']['expected_src_x'] == dl1_params['src_x']).all()
    assert (src_dep_df_gamma['on']['expected_src_y'] == dl1_params['src_y']).all()

    np.testing.assert_allclose(
        src_dep_df_proton['on']['expected_src_x'], 0.205, atol=1e-2
    )
    np.testing.assert_allclose(
        src_dep_df_proton['on']['expected_src_y'], 0., atol=1e-2
    )

@pytest.mark.private_data
def test_get_source_dependent_parameters_observed(observed_dl1_files):
    from lstchain.reco.dl1_to_dl2 import get_source_dependent_parameters

    # on observation data
    srcdep_config['observation_mode']='on'
    dl1_params = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    src_dep_df_on = get_source_dependent_parameters(dl1_params, srcdep_config)

    # wobble observation data
    srcdep_config['observation_mode']='wobble'
    dl1_params['alt_tel'] += np.deg2rad(0.4)
    src_dep_df_wobble = get_source_dependent_parameters(dl1_params, srcdep_config)

    assert 'alpha' in src_dep_df_on['on'].columns
    assert 'dist' in src_dep_df_on['on'].columns
    assert 'time_gradient_from_source' in src_dep_df_on['on'].columns
    assert 'skewness_from_source' in src_dep_df_on['on'].columns
    assert (src_dep_df_on['on']['expected_src_x'] == 0).all()
    assert (src_dep_df_on['on']['expected_src_y'] == 0).all()

    np.testing.assert_allclose(
        src_dep_df_wobble['on']['expected_src_x'], -0.205, atol=1e-2
    )
    np.testing.assert_allclose(
        src_dep_df_wobble['on']['expected_src_y'], 0., atol=1e-2
    )
    np.testing.assert_allclose(
        src_dep_df_wobble['off_180']['expected_src_x'], 0.205, atol=1e-2
    )
    np.testing.assert_allclose(
        src_dep_df_wobble['off_180']['expected_src_y'], 0., atol=1e-2
    )

@pytest.mark.private_data
def test_get_interleaved_r1_file(interleaved_r1_file):
    assert interleaved_r1_file.is_file()


def test_build_models(simulated_dl1_file, rf_models):
    infile = simulated_dl1_file
    custom_config = {
        "n_training_events": {
            "gamma_regressors": 0.99,
            "gamma_tmp_regressors": 0.78,
            "gamma_classifier": 0.21,
            "proton_classifier": 0.98
        }
    }
    reg_energy, reg_disp_norm, cls_disp_sign, cls_gh = build_models(
        infile,
        infile,
        save_models=False,
        free_model_memory=False,
        custom_config=custom_config
    )

    build_models(
        infile,
        infile,
        save_models=True,
        free_model_memory=True,
        custom_config=custom_config
    )

    import joblib

    joblib.dump(reg_energy, rf_models["energy"], compress=3)
    joblib.dump(cls_gh, rf_models["gh_sep"], compress=3)
    joblib.dump(reg_disp_norm, rf_models["disp_norm"], compress=3)
    joblib.dump(cls_disp_sign, rf_models["disp_sign"], compress=3)


@pytest.mark.xfail(raises=ValueError)
def test_fail_build_models(simulated_dl1_file):
    custom_config = {
        "n_training_events": {
            "gamma_regressors": 0.99,
            "gamma_tmp_regressors": 0.78,
            "gamma_classifier": 0.31,
            "proton_classifier": 0.98
        }
    }
    _, _, _, _ = build_models(
        simulated_dl1_file,
        simulated_dl1_file,
        save_models=False,
        custom_config=custom_config
    )


def test_apply_models(simulated_dl1_file, simulated_dl2_file, rf_models):
    from lstchain.reco.dl1_to_dl2 import apply_models
    import joblib

    dl1 = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    dl1 = filter_events(
        dl1,
        filters=standard_config["events_filters"],
        finite_params=standard_config['energy_regression_features']
        + standard_config['disp_regression_features']
        + standard_config['particle_classification_features']
        + standard_config['disp_classification_features']
    )

    reg_energy = joblib.load(rf_models["energy"])
    reg_cls_gh = joblib.load(rf_models["gh_sep"])
    reg_disp_norm = joblib.load(rf_models["disp_norm"])
    cls_disp_sign = joblib.load(rf_models["disp_sign"])

    dl2 = apply_models(dl1, reg_cls_gh, reg_energy, reg_disp_norm=reg_disp_norm,
                       cls_disp_sign=cls_disp_sign, custom_config=standard_config)

    dl2 = apply_models(dl1, rf_models["gh_sep"], rf_models["energy"], reg_disp_norm=rf_models["disp_norm"],
                       cls_disp_sign=rf_models["disp_sign"], custom_config=standard_config)

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
    disp_vec = disp_vector(disp_norm, disp_angle, disp_sign)
    disp_dx = disp_vec[:, 0]
    disp_dy = disp_vec[:, 1]
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
