import numpy as np
import astropy.units as u
from ctapipe.containers import ArrayEventContainer
from ctapipe.instrument import SubarrayDescription, TelescopeDescription
from traitlets.config import Config
from astropy.time import Time

def test_pedestal_calculator():
    """ test of PedestalIntegrator """
    from lstchain.calib.camera.pedestals import PedestalIntegrator

    tel_id = 0
    n_events = 10
    n_gain = 2
    n_pixels = 1855
    ped_level = 300

    subarray = SubarrayDescription(
        "test array",
        tel_positions={0: np.zeros(3) * u.m},
        tel_descriptions={
            0: TelescopeDescription.from_name(
                optics_name="SST-ASTRI", camera_name="CHEC"
            ),
        },
    )
    subarray.tel[0].camera.readout.reference_pulse_shape = np.ones((1, 2))
    subarray.tel[0].camera.readout.reference_pulse_sample_width = u.Quantity(1, u.ns)

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
    data.r1.tel[tel_id].waveform = np.full((2, n_pixels, 40), ped_level)
    data.trigger.time = Time(0, format='mjd', scale='tai')
    while ped_calculator.num_events_seen < n_events:
        if ped_calculator.calculate_pedestals(data):
            assert data.mon.tel[tel_id].pedestal
            assert np.mean(data.mon.tel[tel_id].pedestal.charge_median) == (
                ped_calculator.extractor.window_width.tel[tel_id] * ped_level
            )
            assert np.mean(data.mon.tel[tel_id].pedestal.charge_std) == 0


