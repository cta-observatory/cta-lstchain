import numpy as np
from ctapipe.utils import get_dataset_path 
from lstchain.mc import (
    read_sim_par,
    process_mc,
    calculate_sensitivity,
    calculate_sensitivity_lima,
    bin_definition,
    ring_containment,
    sens
)
from eventio.simtel.simtelfile import SimTelFile


def test_read_sim_par():
    gamma_test_path = get_dataset_path("gamma_test.simtel.gz") 
    source = SimTelFile(gamma_test_path)

    par = read_sim_par(source)

    assert np.isclose(par['emin'], 0.003)
    assert np.isclose(par['emax'], 330.0)
    assert np.isclose(par['spectral_idx'], -2.0)
    assert np.isclose(par['n_showers'], 100000)
    assert np.isclose(par['n_use'], 10)
    assert np.isclose(par['max_impact'], 2500.0)
    assert np.isclose(par['cone'], 0.0)


#def test_process_mc():
    # need to define a dl2 test_dataset

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
            50, 10, 0.2), 26.97, rtol = 1.e-3)
    np.testing.assert_allclose(calculate_sensitivity_lima(
            200, 50, 1), 31.5, rtol = 1.e-3)
    # Testing an array
    np.testing.assert_allclose(calculate_sensitivity_lima(
            [10, 100], [50,100], 1), [630.07,  83.57], rtol = 1.e-3) 

def test_bin_definition():

    gb, tb = bin_definition(11,10)
    gbin = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    tbin = [0.005, 0.01 , 0.015, 0.02 , 0.025, 0.03 , 0.035, 0.04, 
            0.045, 0.05]
    np.testing.assert_allclose(gb, gbin) 
    np.testing.assert_allclose(tb.to_value(), tbin) 

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


#def test_sens():
    # need to define a dl2 test_dataset



