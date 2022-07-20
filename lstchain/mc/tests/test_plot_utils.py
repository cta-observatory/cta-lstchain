import matplotlib.pyplot as plt
from lstchain.mc import plot_utils
import astropy.units as u
import numpy as np


def test_plot_Crab_SED():
    plt.figure()
    plot_utils.plot_Crab_SED(1*u.GeV, 1*u.PeV, percentage=45, label=r'Crab')


def test_sensitivity_plot_comparison():
    energy = np.geomspace(0.01, 100, 20) * u.TeV
    e_center = np.sqrt(energy[:-1] * energy[1:])
    sensitivity = e_center**-1 * 1e-12 * u.TeV**2/(u.cm**2 * u.s)
    plt.figure()
    plot_utils.sensitivity_plot_comparison(energy, sensitivity)

