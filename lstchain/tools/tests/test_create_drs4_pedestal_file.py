import os
from pathlib import Path

import numpy as np

from ctapipe.core import run_tool
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL
import tables


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

    with tables.open_file(output_path, 'r') as f:
        baseline_mean = f.root.baseline.mean[:].astype(np.float32) / 100
        baseline_counts = f.root.baseline.counts[:]

        assert baseline_mean.shape == (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)
        assert np.isclose(np.average(baseline_mean, weights=baseline_counts), 400, rtol=0.05)

        for sample, expected in zip([1, 2, 3], [47, 53, 6]):
            spike_mean = f.root[f'spike{sample}'].mean[:].astype(np.float32) / 100
            spike_mean[spike_mean < 0] = np.nan
            assert np.isclose(np.nanmean(spike_mean), expected, atol=5)
