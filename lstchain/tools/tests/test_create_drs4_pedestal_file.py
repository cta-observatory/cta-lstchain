import os
from pathlib import Path

import numpy as np

from ctapipe.core import run_tool
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL
from ctapipe.io import read_table


test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()


def test_create_drs4_pedestal_file(tmp_path):
    '''Test the lstchain_drs4_pedestal_file tool'''
    from lstchain.tools.lstchain_create_drs4_pedestal_file import DRS4PedestalAndSpikeHeight

    input_path = test_data / "real/R0/20200218/LST-1.1.Run02005.0000_first50.fits.fz"
    output_path = tmp_path / "drs4_pedestal_02005.h5"

    ret = run_tool(
        DRS4PedestalAndSpikeHeight(),
        argv=[
            f"--input={input_path}",
            f"--output={output_path}",
            "--overwrite",
        ],
        cwd=tmp_path,
    )

    assert ret == 0, 'Running DRS4PedestalAndSpikeHeight tool failed'
    assert output_path.is_file(), 'Output file not written'

    drs4_data = read_table(output_path, '/r1/monitoring/drs4_baseline/tel_001')[0]
    baseline_mean = drs4_data['baseline_mean']
    baseline_counts = drs4_data['baseline_std']

    assert baseline_mean.dtype == np.int16
    assert baseline_mean.shape == (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)
    assert np.isclose(np.average(baseline_mean[baseline_counts > 0], weights=baseline_counts[baseline_counts > 0]), 400, rtol=0.05)

    spike_height = drs4_data['spike_height']
    assert spike_height.dtype == np.int16
    mean_spike_height = np.nanmean(spike_height, axis=(0, 1))

    # these are the expected spike heights, but due to the low statistics,
    # we need to use a rather large atol
    assert np.allclose(mean_spike_height, [46, 53, 7], atol=2)
