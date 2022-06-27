import numpy as np
import pytest

from lstchain.scripts.tests.test_lstchain_scripts import run_program
from lstchain.tests.test_lstchain import test_drs4_pedestal_path
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL


@pytest.mark.private_data
def test_convert_drs4_pedestal_file(tmp_path):
    outfile = tmp_path / 'pedestal_for_evb.dat'

    run_program(
        "lstchain_convert_drs4_pedestal_to_evb",
        str(test_drs4_pedestal_path),
        outfile,
    )

    pedestal = np.fromfile(outfile, dtype='<u2')
    assert len(pedestal) == N_GAINS * N_PIXELS * N_CAPACITORS_PIXEL
    assert np.isclose(pedestal.mean(), 400, atol=2)
