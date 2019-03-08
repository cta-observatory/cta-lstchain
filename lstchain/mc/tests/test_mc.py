import numpy as np
from numpy.testing import assert_allclose
from lstchain.mc import power_law_integrated_distribution

def test_integrated_distribution():

    emin = 50.     # u.GeV
    emax = 500.e3  # u.GeV
    Nevents = 1e6
    spectral_index = -2
    bins = 30

    b ,y = power_law_integrated_distribution(
        emin, emax, Nevents, spectral_index, bins)

    assert_allclose(Nevents,np.sum(y),rtol=1.e-2)
