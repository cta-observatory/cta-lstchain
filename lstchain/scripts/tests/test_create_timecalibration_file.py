from pathlib import Path
import os
import subprocess as sp


test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
calib_path = test_data / 'real/monitoring/PixelCalibration/Cat-A'
calib_version = 'ctapipe-v0.17'



def test_lstchain_create_timecalibration_file(tmp_path):
    '''Test that the timen calibration file creation script runs at least
    until the finalize step.

    Because we have way to few events in the test file and a full run would
    take too long, we expect the exception about some capacitors not having
    enough data, but the rest of the script is expected to work.
    '''
    input_path = test_data / "real/R0/20200218/LST-1.1.Run02005.*_first50.fits.fz"
    pedestal_path = calib_path / f'drs4_baseline/20200218/{calib_version}/drs4_pedestal.Run02005.0000.h5'
    run_summary_path = test_data / "real/monitoring/RunSummary/RunSummary_20200218.ecsv"
    output_path = tmp_path / "test_timecalibration.h5"

    ret = sp.run(
        [
            "lstchain_data_create_time_calibration_file",
            "--input-file", str(input_path),
            "--output-file", str(output_path),
            "--pedestal-file", str(pedestal_path),
            "--run-summary-path" , str(run_summary_path)
        ],
        stdout=sp.PIPE,
        stderr=sp.STDOUT,  # redirect stderr to stdout, to have both in one string
        encoding='utf-8',
    )

    if ret.returncode != 0:
        # make sure it was the expected error and not something else
        assert 'RuntimeError: No data available for some capacitors.' in ret.stdout, ret.stdout

