import tempfile
from pathlib import Path

import pandas as pd
import pytest

from lstchain.io.io import dl1_params_lstcam_key
from lstchain.scripts.tests.test_lstchain_scripts import run_program
from ctapipe.utils import get_dataset_path
from lstchain.tests.test_lstchain import (
    test_drive_report,
    test_drs4_pedestal_path,
    test_calib_path,
    test_time_calib_path,
    test_r0_path,
    test_r0_path2,
    test_data,
)


# add a marker for the tests that need private data and don't run them
# by default
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "private_data: mark tests that needs the private test data"
    )

    if 'private_data' not in config.option.markexpr:
        if config.option.markexpr:
            config.option.markexpr += ' and '
        else:
            config.option.markexpr += 'not private_data'


@pytest.fixture(scope="session")
def temp_dir():
    """Shared temporal directory for the tests."""
    with tempfile.TemporaryDirectory(prefix='test_lstchain') as d:
        yield Path(d)


@pytest.fixture(scope="session")
def temp_dir_simulated_files():
    """Temporal common directory for processing simulated data."""
    with tempfile.TemporaryDirectory(prefix='test_lstchain') as d:
        yield Path(d)


@pytest.mark.private_data
@pytest.fixture(scope="session")
def temp_dir_observed_files():
    """Temporal common directory for processing observed data."""
    with tempfile.TemporaryDirectory(prefix='test_lstchain') as d:
        yield Path(d)


@pytest.fixture(scope="session")
def mc_gamma_testfile():
    """Get a simulated test file."""
    return get_dataset_path('gamma_test_large.simtel.gz')


@pytest.fixture(scope="session")
def simulated_dl1_file(temp_dir_simulated_files, mc_gamma_testfile):
    """Produce a dl1 file from simulated data."""
    output_dl1_path = temp_dir_simulated_files / "dl1_gamma_test_large.h5"
    run_program(
        "lstchain_mc_r0_to_dl1",
        "-f", mc_gamma_testfile,
        "-o", temp_dir_simulated_files
    )
    return output_dl1_path


@pytest.fixture(scope='session')
def run_summary_path(temp_dir_observed_files):
    date = "20200218"
    r0_path = test_data / "real/R0"
    run_summary_path = temp_dir_observed_files / f"RunSummary_{date}.ecsv"
    run_program(
        "lstchain_create_run_summary",
        "--date", date,
        "--r0-path", r0_path,
        "--output-dir", temp_dir_observed_files
    )

    return run_summary_path


@pytest.mark.private_data
@pytest.fixture(scope="session")
def observed_dl1_files(temp_dir_observed_files, run_summary_path):
    """
    Produce dl1, datacheck and muons files from real observed data.
    The initial timestamps and counters used for the first set of files
    here are extracted from the night summary. In this case these values
    correspond to the third event. A second set of files are produced
    without using the first valid timestamps.
    """
    # FIXME: naming criteria (suffixes, no stream) of dl1, dl2,
    #  muons and datacheck files should be coherent

    # First set of files to be produced
    dl1_output_path1 = temp_dir_observed_files / "dl1_LST-1.Run02008.0000.h5"
    muons_file1 = temp_dir_observed_files / "muons_LST-1.Run02008.0000.fits"
    datacheck_file1 = temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.0000.h5"

    # Second set of files
    dl1_output_path2 = temp_dir_observed_files / "dl1_LST-1.Run02008.0100.h5"
    muons_file2 = temp_dir_observed_files / "muons_LST-1.Run02008.0100.fits"
    datacheck_file2 = temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.0100.h5"

    run_program(
        "lstchain_data_r0_to_dl1",
        "-f",
        test_r0_path,
        "-o",
        temp_dir_observed_files,
        "--pedestal-file",
        test_drs4_pedestal_path,
        "--calibration-file",
        test_calib_path,
        "--time-calibration-file",
        test_time_calib_path,
        "--pointing-file",
        test_drive_report,
        "--dragon-reference-time",
        "1582059789516351903",
        "--dragon-reference-counter",
        "2516351600",
        "--dragon-module-id",
        "132",
    )

    run_program(
            "lstchain_check_dl1",
            "-b",
            "--omit-pdf",
            "--output-dir",
            temp_dir_observed_files,
            "--input-file",
            dl1_output_path1
    )

    run_program(
        "lstchain_data_r0_to_dl1",
        "-f",
        test_r0_path2,
        "-o",
        temp_dir_observed_files,
        "--pedestal-file",
        test_drs4_pedestal_path,
        "--calibration-file",
        test_calib_path,
        "--time-calibration-file",
        test_time_calib_path,
        "--pointing-file",
        test_drive_report,
        '--run-summary-path',
        run_summary_path,
    )

    run_program(
            "lstchain_check_dl1",
            "-b",
            "--omit-pdf",
            "--output-dir",
            temp_dir_observed_files,
            "--input-file",
            dl1_output_path2
    )

    return {
        'dl1_file1': dl1_output_path1,
        'muons1': muons_file1,
        'datacheck1': datacheck_file1,
        'dl1_file2': dl1_output_path2,
        'muons2': muons_file2,
        'datacheck2': datacheck_file2
    }


@pytest.fixture(scope="session")
def simulated_dl2_file(temp_dir_simulated_files, simulated_dl1_file, rf_models):
    """
    Produce the test dl2 file from the simulated dl1 test file
    using the random forest test models.
    """
    dl2_file = temp_dir_simulated_files / "dl2_gamma_test_large.h5"
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        simulated_dl1_file,
        "--path-models",
        rf_models["path"],
        "--output-dir",
        temp_dir_simulated_files
    )
    return dl2_file


@pytest.fixture(scope="session")
def fake_dl1_proton_file(temp_dir_simulated_files, simulated_dl1_file):
    """
    Produce a fake dl1 proton file by copying the dl2 gamma test file
    and changing mc_type.
    """
    dl1_proton_file = temp_dir_simulated_files / 'dl1_fake_proton.simtel.h5'
    events = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    events.mc_type = 101
    events.to_hdf(dl1_proton_file, key=dl1_params_lstcam_key)
    return dl1_proton_file


@pytest.fixture(scope="session")
def rf_models(temp_dir_simulated_files, simulated_dl1_file):
    """Produce test random forest models."""
    gamma_file = simulated_dl1_file
    proton_file = simulated_dl1_file
    models_path = temp_dir_simulated_files
    file_model_energy = models_path / "reg_energy.sav"
    file_model_disp = models_path / "reg_disp_vector.sav"
    file_model_gh_sep = models_path / "cls_gh.sav"

    run_program(
        "lstchain_mc_trainpipe", "--fg", gamma_file, "--fp", proton_file, "-o", models_path
    )
    return {
        'energy': file_model_energy,
        'disp': file_model_disp,
        'gh_sep': file_model_gh_sep,
        'path': models_path
    }


@pytest.fixture(scope="session")
@pytest.mark.private_data
def observed_dl2_file(temp_dir_observed_files, observed_dl1_files,  rf_models):
    """Produce a dl2 file from an observed dl1 file."""
    real_data_dl2_file = temp_dir_observed_files / (observed_dl1_files["dl1_file1"].name.replace("dl1", "dl2"))
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        observed_dl1_files["dl1_file1"],
        "--path-models",
        rf_models["path"],
        "--output-dir",
        temp_dir_observed_files
    )
    return real_data_dl2_file
