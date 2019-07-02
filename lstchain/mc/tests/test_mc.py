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
    
    emin = 20.     # u.GeV
    emax = 300.e3  # u.GeV
    spectral_index = -3.
    e0 = 300.     # u.GeV
    area = 1.e9
    cone = 0
    norm = 1.e-11

    np.testing.assert_allclose(rate(emin, emax, spectral_index, 
                                    cone, area, norm, e0), 
                               337.5, rtol=1e-3)

def test_weight():
    
    emin = 10.     # u.GeV
    emax = 50.e3  # u.GeV
    sim_sp_idx = -2.
    w_sp_idx = -2.6
    e0 = 1000.     # u.GeV
    rate = 8.
    nev = 1.e6

    np.testing.assert_allclose(weight(emin, emax, sim_sp_idx, 
                                      w_sp_idx, rate, nev, e0), 
                               8.07e-7, rtol=1e-3)
