import numpy as np
import pytest
from ctapipe.containers import ArrayEventContainer

def test_get_ped_thresh():
    from ..pixel_threshold_estimation import get_ped_thresh

    tel_id = 1
    sigma = 2
    event = ArrayEventContainer()
    event.mon.tel[tel_id].pedestal.charge_mean = np.full((2, 2, 1855), 4, dtype='int')
    event.mon.tel[tel_id].pedestal.charge_std = np.full((2, 2, 1855), 3, dtype='int')
    event.mon.tel[tel_id].calibration.dc_to_pe = np.full((2, 2, 1855), 2, dtype='int')
    event.mon.tel[tel_id].calibration.unusable_pixels = np.full((1855,), False, dtype='bool')

    ped_thresh = get_ped_thresh(tel_id=tel_id, event=event, sigma_clean=sigma)
    ped_thresh_real = np.full((1855,), 20, dtype='int')

    assert (ped_thresh.all() == ped_thresh_real.all())

@pytest.mark.private_data
def test_get_bias_and_std(observed_dl1_files):
    from ..pixel_threshold_estimation import get_bias_and_std

    file = observed_dl1_files["dl1_file1"]
    ped_charge_mean_pe, ped_charge_std_pe = get_bias_and_std(file)

    assert (ped_charge_mean_pe.shape[2] == 1855)
    assert (ped_charge_std_pe.shape[2] == 1855)


@pytest.mark.private_data
def test_get_threshold_from_dl1_file(observed_dl1_files):
    from ..pixel_threshold_estimation import get_threshold_from_dl1_file

    file = observed_dl1_files["dl1_file1"]
    sigma = 2.5
    ped_thresh = get_threshold_from_dl1_file(file, sigma_clean=sigma)
    
    assert (ped_thresh.shape == (1855,))
    assert (np.max(ped_thresh) > 0)


@pytest.mark.private_data
def test_get_unusable_pixels(observed_dl1_files):
    from ..pixel_threshold_estimation import get_unusable_pixels

    file = observed_dl1_files["dl1_file1"]
    unusable_pixels = get_unusable_pixels(file, pedestal_id=0)
    
    assert (len(unusable_pixels[0]) == 2)

