import numpy as np
import pytest
# import pandas as pd # uncomment when the tests loading dl2_file are written
from lstchain.mc import (
    read_sim_par,
    calculate_sensitivity,
    calculate_sensitivity_lima,
    bin_definition,
    ring_containment,
)

from lstchain.tests.test_lstchain import dl1_file
# from lstchain.tests.test_lstchain import dl2_file, dl2_params_lstcam_key  # uncomment when the tests loading dl2_file are written


@pytest.mark.run(after='test_dl0_to_dl1')
def test_read_sim_par():
    par = read_sim_par(dl1_file)

    assert np.isclose(par['emin'].to_value(), 0.003)
    assert np.isclose(par['emax'].to_value(), 330.0)
    assert np.isclose(par['sp_idx'], -2.0)
    assert np.isclose(par['sim_ev'], 400000)
    assert np.isclose(par['area_sim'].to_value(), 28274333.88)
    assert np.isclose(par['cone'].to_value(), 10.0)


@pytest.mark.run(after='test_apply_models')
def test_process_mc():
    # TODO: write a test for `test_process_mc` using `dl2` dataframe
    # dl2 = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    pass

def test_calculate_sensitivity():

    np.testing.assert_allclose(calculate_sensitivity(
        50, 10, 0.2), 14.14, rtol = 1.e-3) 
    np.testing.assert_allclose(calculate_sensitivity(
        200, 50, 1), 17.67, rtol = 1.e-3)
    # Testing an array
    np.testing.assert_allclose(calculate_sensitivity(
        [10, 100], [50,100], 1), [353.55,  50.], rtol = 1.e-3)

def test_calculate_sensitivity_lima():
    
    np.testing.assert_allclose(calculate_sensitivity_lima(
            50, 10, 0.2, 1, 0, 0),
                               ([13.48],[26.97]), rtol = 1.e-3)
    np.testing.assert_allclose(calculate_sensitivity_lima(
            200, 50, 1, 0, 1, 0),
                               ([63.00],[31.5]), rtol = 1.e-3)
    # Testing an array
    np.testing.assert_allclose(calculate_sensitivity_lima(
            [10, 100], [50,100], 1, 1, 1, 0),
                               ([63.00, 83.57],[630.07,  83.57]), rtol = 1.e-3)

def test_bin_definition():

    gammaness_bins, theta2_bins = bin_definition(11,10)
    gammaness_bins_test = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    theta2_bins_test = [0.005, 0.01 , 0.015, 0.02 , 0.025, 0.03 , 0.035, 0.04, 
            0.045, 0.05]
    np.testing.assert_allclose(gammaness_bins, gammaness_bins_test) 
    np.testing.assert_allclose(theta2_bins.to_value(), theta2_bins_test) 

def test_ring_containment():

    # Event contained
    ring_containment(0.4, 0.5, 0.1)

    # Event not contained
    ring_containment(0.1, 0.4, 0.1)

    # Testing an array
    ring_containment(np.linspace(0.1,1,10), 0.6, 0.2)

def test_ring_containment():

    # Event contained
    contained, area = ring_containment(0.4**2, 0.5, 0.101)
    np.testing.assert_allclose(area, 0.57, rtol=1.e-3)
    np.testing.assert_equal(contained, True)

    # Event not contained
    contained, area = ring_containment(0.1**2, 0.4, 0.1)
    np.testing.assert_allclose(area, 0.44, rtol=1.e-3)
    np.testing.assert_equal(contained, False)

    # Testing an array
    contained, area = ring_containment(np.linspace(0.1,1,10), 0.6, 0.2)
    np.testing.assert_allclose(area, 1.256, rtol=1.e-3)
    np.testing.assert_equal(contained, [False,  True,  True,  True, True, 
                                        False, False, False, False, False])


@pytest.mark.run(after='test_apply_models')
def test_sens():
    # TODO: define test for sens
    # dl2 = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    pass



