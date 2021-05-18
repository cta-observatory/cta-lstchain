import numpy as np
from lstchain.mc import (
    power_law_integrated_distribution,
    int_diff_sp,
    rate,
    weight
)

def test_integrated_distribution():

    emin = 50.     # u.GeV
    emax = 500.e3  # u.GeV
    nevents = 1e6
    spectral_index = -2.5
    bins = 30

    b , y = power_law_integrated_distribution(
        emin, emax, nevents, spectral_index, bins)

    np.testing.assert_allclose(nevents,np.sum(y),rtol=1.e-10)

def test_diff_sp():

    emin = 30.     # u.GeV
    emax = 100.e3  # u.GeV
    spectral_index = -2.
    e0 = 1000.     # u.GeV

    integral_e = int_diff_sp(emin, emax, spectral_index, e0)

    np.testing.assert_allclose(integral_e, 33323, rtol=1e-3)

def test_rate():
    
    shape = 'PowerLaw'
    emin = 20.     # u.GeV
    emax = 300.e3  # u.GeV
    param = {'f0':1.e-11, 'e0':300, 'alpha': -3}
    area = 1.e9
    cone = 0

    np.testing.assert_allclose(rate(shape, emin, emax, param, 
                                    cone, area), 
                               337.5, rtol=1e-3)

def test_weight():

    shape = 'PowerLaw'
    emin = 10.     # u.GeV
    emax = 50.e3  # u.GeV
    sim_sp_idx = -2.
    w_param = {'f0':1.e-11, 'e0':1000, 'alpha': -2.6}
    rate = 8.
    nev = 1.e6

    np.testing.assert_allclose(weight(shape, emin, emax, sim_sp_idx, 
                                      rate, nev, w_param), 
                               8.07e-7, rtol=1e-3)
