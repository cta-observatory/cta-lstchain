from pathlib import Path
import os
from astropy.io import fits

from .test_lstchain_scripts import run_program


test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))


def test_lstchain_create_drs4_pedestal_file(tmp_path):
    '''Test that the drs4 pedestal file creation script runs through

    We cannot test that the resulting calibration coefficients are ok,
    since we have far to few events in the test file, but better than nothing
    '''
    input_path = test_data / "real/R0/20200218/LST-1.1.Run02005.0000_first50.fits.fz"
    output_path = tmp_path / "test_drs4.fits.gz"

    run_program(
        "lstchain_data_create_drs4_pedestal_file",
        "--input-file", str(input_path),
        "--output-file", str(output_path),
    )

    with fits.open(output_path, "readonly") as f:
        assert "pedestal array" in f
        assert "failing pixels" in f
        assert f["pedestal array"].data.shape == (2, 1855, 4096)
        assert f["failing pixels"].data.shape == (1855, )

