import numpy as np
import astropy.units as u
from ctapipe.containers import ArrayEventContainer
from traitlets.config import Config
from astropy.time import Time
from copy import deepcopy

def test_pedestal_calculator(lst1_subarray):
    """ test of PedestalIntegrator """
    from lstchain.calib.camera.pedestals import PedestalIntegrator

    tel_id = 1
    n_events = 1000
    n_gain = 2
    n_pixels = 1855
    ped_level = 300
    ped_std = 10

    subarray = deepcopy(lst1_subarray)
    subarray.tel[tel_id].camera.readout.reference_pulse_shape = np.ones((1, 2))
    subarray.tel[tel_id].camera.readout.reference_pulse_sample_width = u.Quantity(1, u.ns)

    config = Config({
        "FixedWindowSum": {
          "apply_integration_correction": False,
        }
    })
    ped_calculator = PedestalIntegrator(charge_product="FixedWindowSum",
                                        config=config,
                                        sample_size=n_events,
                                        tel_id=tel_id,
                                        subarray=subarray)
    # create one event
    data = ArrayEventContainer()
    data.meta['origin'] = 'test'

    # fill the values necessary for the pedestal calculation
    data.mon.tel[tel_id].pixel_status.hardware_failing_pixels = np.zeros(
        (n_gain, n_pixels), dtype=bool
    )
    rng = np.random.default_rng(0)
    data.trigger.time = Time(0, format='mjd', scale='tai')


    window_width = ped_calculator.extractor.window_width.tel[tel_id]
    while ped_calculator.num_events_seen < n_events:
        data.r1.tel[tel_id].waveform = rng.normal(ped_level, ped_std, (2, n_pixels, 40))
        if ped_calculator.calculate_pedestals(data):
            assert data.mon.tel[tel_id].pedestal
            assert np.isclose(
                np.mean(data.mon.tel[tel_id].pedestal.charge_median),
                window_width * ped_level,
                rtol=0.01,
            )
            assert np.isclose(
                np.mean(data.mon.tel[tel_id].pedestal.charge_std),
                np.sqrt(window_width) * ped_std,
                rtol=0.05,
            )


