from lstchain.mc import plot_utils
import astropy.units as u
import numpy as np


def test_plot_Crab_SED():

    plot_utils.plot_Crab_SED(100, 1*u.GeV, 1*u.PeV, label=r'Crab')


def test_sensitivity_plot_comparison():

    energy = np.geomspace(0.01, 100, 20) * u.TeV
    sensitivity = energy**-1 * 1e-12 / (u.cm**2 * u.s)

    plot_utils.sensitivity_plot_comparison(energy, sensitivity)

