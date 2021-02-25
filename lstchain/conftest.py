import tempfile
from pathlib import Path

import pandas as pd
import pytest

from lstchain.io.io import dl1_params_lstcam_key
from lstchain.scripts.tests.test_lstchain_scripts import run_program
from ctapipe.utils import get_dataset_path


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
