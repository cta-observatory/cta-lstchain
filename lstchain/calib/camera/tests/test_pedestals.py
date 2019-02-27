from ctapipe.utils import get_dataset_path
from ctapipe.io.nectarcameventsource import NectarCAMEventSource
from lstchain.calib.camera.pedestals import PedestalIntegrator
from .. import pedestals

import numpy as np

def test_pedestal_calculator():

    example_file_path = get_dataset_path("NectarCAM.Run0890.10events.fits.fz")

    inputfile_reader = NectarCAMEventSource(
        input_url=example_file_path,
        max_events=10
    )

    ped_calculator = PedestalIntegrator(sample_size=3, tel_id=0)

    for event in inputfile_reader:

        ped_data = ped_calculator.calculate_pedestals(event)

        if ped_calculator.num_events_seen == ped_calculator.sample_size:
            assert ped_data



def test_calc_pedestals_from_traces():

    # create some test data (all ones, but with a 2 stuck in for good measure):
    npix = 1000
    nsamp = 32
    traces = np.ones(shape=(npix, nsamp), dtype=np.int32)
    traces[:, 2] = 2

    # calculate pedestals over samples 0-10
    peds, pedvars = pedestals.calc_pedestals_from_traces(traces, 0, 10)

    # check the results. All pedestals should be 1.1 and vars 0.09:

    assert np.all(peds == 1.1)
    assert np.all(pedvars == 0.09)
    assert peds.shape[0] == npix
    assert pedvars.shape[0] == npix

    # try another sample range, where there should be no variances and
    # all 1.0 peds:
    peds, pedvars = pedestals.calc_pedestals_from_traces(traces, 12, 32)

    assert np.all(peds == 1.0)
    assert np.all(pedvars == 0)
