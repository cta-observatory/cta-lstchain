import numpy as np
from lstchain.mc import (power_law_integrated_distribution,
                  int_diff_sp,
                  rate,
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
    omega = 1.
    norm = 1.e-11

    integral_e = int_diff_sp(emin, emax, spectral_index, e0)
    rate = norm * area * omega * integral_e

    np.testing.assert_allclose(rate, 337.5, rtol=1e-3)
