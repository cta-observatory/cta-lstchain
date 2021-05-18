import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lstchain.spectra.crab import crab_magic, crab_hegra
from lstchain.visualization.plot_dl2 import plot_pos
import numpy as np
import astropy.units as u
import pandas as pd
from ctaplot.plots import plot_sensitivity_magic_performance
from pyirf.spectral import CRAB_MAGIC_JHEAP2015

__all__ = [
    'fill_bin_content',
    'format_axes_ebin',
    'format_axes_array',
    'format_axes_sensitivity',
    'plot_Crab_SED',
    'plot_LST_preliminary_sensitivity',
    'plot_sensitivity',
    'sensitivity_minimization_plot',
    'sensitivity_plot_comparison',
    'plot_positions_survived_events',
    ]


def fill_bin_content(ax, sensitivity, energy_bin, n_bins_gammaness, n_bins_theta2):
    """
    Function to fill bin content to be plotted in the case of an
    optimized figure array

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`
    sensitivity:  `numpy.ndarray`  sensitivity array
    energy_bin:    `int`  energy bin number
    n_bins_gammaness:    `int`  number of bins in gammaness
    n_bins_theta2:    `int`  number of bins in theta2

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`

    """

    for i in range(0, n_bins_gammaness):
        for j in range(0, n_bins_theta2):
            # With theta2, we start the binning in 0.005 deg^2 and increase up to n_bins_theta2 steps
            # up to reach 0.05 deg^2
            theta2 = 0.005 + 0.005 / 2 + ((0.05 - 0.005) / n_bins_theta2) * j
            # With gammaness, we start with 0.05 and increase up to n_bins_gammaness steps
            # up to reaching 1
            gammaness = 0.1 / 2 + (1 / n_bins_gammaness) * i
            text = ax.text(theta2, gammaness, "%.2f %%" % sensitivity[energy_bin][i][j],
                           ha = "center", va = "center", color = "w", size = 8)
    return ax


def format_axes_ebin(ax, img):
    """
    Format axes for the theta2 and gammaness optimization per energy bin

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`
    img:    `matplotlib.image.AxesImage`

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`

    """

    ax.set_aspect('auto')

    ax.set_ylabel(r'Gammaness', fontsize = 15)
    ax.set_xlabel(r'$\theta^2$ (deg$^2$)', fontsize = 15)

    starty, endy = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(endy, starty, 0.1)[::-1])
    startx, endx = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(startx, endx, 0.01))

    fig = ax.get_figure()
    #cbaxes = fig.add_axes([0.9, 0.125, 0.03, 0.755])
    cbar = fig.colorbar(img)#, cax=cbaxes)
    cbar.set_label('Sensitivity (% Crab)', fontsize = 15)


def format_axes_array(ax, arr_i, arr_j, plot):
    """
    Format axes for the theta2 and gammaness optimization for a
    figure array with all energy bins together

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`
    arr_i: `int` i index for the square plot
    arr_j: `int` j index for the square plot
    plot:  `matplotlib.pyplot.figure`

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`
    """
    ax.set_aspect(0.5)
    if ((arr_i == 0) and (arr_j == 0)):
        ax.set_ylabel(r'Gammaness', fontsize=15)
    if ((arr_i == 3) and (arr_j == 2)):
        ax.set_xlabel(r'$\theta^2$ (deg$^2$)', fontsize=15)

    starty, endy = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(endy, starty, 0.1)[::-1])
    startx, endx = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(startx, endx, 0.1))

    fig = ax.get_figure()
    cbaxes = fig.add_axes([0.91, 0.125, 0.03, 0.755])
    cbar = fig.colorbar(plot, cax=cbaxes)
    cbar.set_label('Sensitivity (% Crab)', fontsize=15)


def format_axes_sensitivity(ax):
    """
    Format axes of the sensitivity plot

    Parameters
    ----------
    ax: `matplotlib.pyplot.axis`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """

    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    # ax.set_xlim(5e-2, 9.e1)
    # ax.set_ylim(1.e-14, 5.e-10)
    ax.set_xlabel("Energy [TeV]")
    ax.set_ylabel(r'E$^2$ $\frac{\mathrm{dN}}{\mathrm{dE}}$ [TeV cm$^{-2}$ s$^{-1}$]')
    ax.grid(ls='--', alpha=.5)


def plot_Crab_SED(emin, emax, percentage=100, ax=None, **kwargs):
    """
    Plot a percentage of the Crab SED

    Parameters
    --------
    emin: `astropy.units.quantity.Quantity` compatible with energies
    emax:  astropy.units.quantity.Quantity compatible with energies
    percentage:  `float`  percentage of the Crab Nebula to be plotted
    ax:    `matplotlib.pyplot.axis`
    kwargs: kwargs for `matplotlib.pyplot.plot`

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    energy = np.geomspace(emin.to_value(u.TeV), emax.to_value(u.TeV), 40) * u.TeV

    if percentage==100:
        kwargs.setdefault('label', f'Crab (MAGIC JHEAP 2015)')
    else:
        kwargs.setdefault('label', f'{percentage}% Crab (MAGIC JHEAP 2015)')

    kwargs.setdefault('color', 'gray')
    ax.plot(energy.to_value(u.TeV),
            percentage/100. * (energy**2 * CRAB_MAGIC_JHEAP2015(energy)).to_value(u.TeV / (u.cm * u.cm * u.s)),
            **kwargs
            )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Energy [TeV]")
    ax.set_ylabel(r'E$^2$ $\frac{\mathrm{dN}}{\mathrm{dE}}$ [TeV cm$^{-2}$ s$^{-1}$]')
    ax.legend()
    return ax


def plot_sensitivity(energy, sensitivity, ax=None, **kwargs):
    """
    Plot the achieved sensitivity

    Parameters
    --------
    ax:          `matplotlib.pyplot.axis`
    energy:      `astropy.units.quantity.Quantity`  energy array
    sensitivity: `numpy.ndarray`  sensitivity array (bins of energy)

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`

    """
    ax = plt.gca() if ax is None else ax

    mask = sensitivity < 1e100 * sensitivity.unit
    egeom = np.sqrt(energy[1:] * energy[:-1])
    binsize = (energy[1:] - energy[:-1]) / 2

    dFdE = crab_hegra(egeom[mask])

    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.errorbar(egeom[mask].to_value(),
                (sensitivity[mask] / 100 * (dFdE[0] * egeom[mask] \
                                            * egeom[mask]).to(u.TeV / (u.cm * u.cm * u.s))).to_value(),
                xerr=binsize[mask].to_value(), marker='o', color='C3', label='Sensitivity')

    return ax


def sensitivity_minimization_plot(n_bins_energy, n_bins_gammaness, n_bins_theta2, energy, sensitivity_3Darray):
    """
    Plot the sensitivity minimization plots in different
    energy bins to check that the theta2 and gammaness
    cuts were properly applied

    TODO: Save plots!
    Parameters
    --------
    n_bins_energy:    `int`  number of bins in energy
    n_bins_gammaness:    `int`  number of bins in gammaness
    n_bins_theta2:    `int`  number of bins in theta2
    energy:  `numpy.ndarray`  energy array
    sensitivity_3Darray: `numpy.ndarray`  sensitivity array (bins of energy, gammaness and theta2)

    Returns
    --------
    figarr: `matplotlib.pyplot.figure`

    """

    #TODO : To be changed!!!
    # if (n_bins_energy == 12):
    #     figarr, axarr = plt.subplots(4,3, sharex=True, sharey=True, figsize=(13.2,18))

    figarr, axarr = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(13.2, 18))

    for i in range(0, n_bins_energy):
        for j in range(0, n_bins_gammaness):
            for k in range(0, n_bins_theta2):
                conditions = (not np.isfinite(sensitivity_3Darray[i, j, k])) or (sensitivity_3Darray[i, j, k] <= 0)
                if conditions:
                    sensitivity_3Darray[i, j, k] = 1

    for ebin in range(0, n_bins_energy):
        if (figarr):
            arr_i = int(ebin / 4)
            arr_j = ebin - int(ebin / 4) * 4
            plot = axarr[arr_i, arr_j].imshow(sensitivity_3Darray[ebin],
                                              cmap='viridis_r',
                                              extent=[0.005, 0.05, 1., 0.],
                                              norm=LogNorm(vmin=sensitivity_3Darray.min(),
                                                           vmax=sensitivity_3Darray.max()),
                                              aspect='auto',
                                              )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Ebin: %.2f - %.2f %s" % (energy[ebin].to_value(),
                                               energy[ebin + 1].to_value(), energy.unit.name))
        img = ax.imshow(sensitivity_3Darray[ebin], cmap='viridis', extent=[0.005, 0.05, 1., 0.], aspect='auto')

        fill_bin_content(ax, sensitivity_3Darray, ebin, n_bins_gammaness, n_bins_theta2)
        format_axes_ebin(ax, img)
        fig.savefig("Ebin%d.png" % ebin)
        if (figarr):
            figarr.subplots_adjust(hspace=0, wspace=0)
            format_axes_array(axarr[arr_i, arr_j], arr_i, arr_j, plot)

    return figarr


def sensitivity_plot_comparison(energy, sensitivity, ax=None):
    """
    Main sensitivity plot.
    We plot the sensitivity achieved, MAGIC sensitivity and Crab SEDs

    Parameters
    --------
    n_bins_energy:   `int`  number of bins in energy
    energy:    `numpy.ndarray`  sensitivity array
    sens: `numpy.ndarray`  sensitivity array (bins of energy, gammaness and theta2)

    Returns
    --------
    fig_sens: `matplotlib.pyplot.figure` Figure containing sensitivity plot

    """

    # Final sensitivity plot
    ax = plt.gca() if ax is None else ax

    ax = plot_sensitivity(energy, sensitivity, ax=ax)

    emin = 10 * u.GeV
    emax = 100 * u.TeV

    plot_Crab_SED(emin, emax, percentage=100, ax=ax, label=r'Crab')
    plot_Crab_SED(emin, emax, percentage=1, ax=ax, ls='dotted', label='1% Crab')
    plot_Crab_SED(emin, emax, percentage=10, ax=ax, ls='-.', label='10% Crab')

    plot_sensitivity_magic_performance(ax=ax)
    format_axes_sensitivity(ax)
    ax.legend(numpoints=1, prop={'size': 9}, ncol=2, loc='upper right')

    return ax


def plot_positions_survived_events(df_gammas,
                                   df_protons,
                                   gammaness_g, gammaness_p,
                                   theta2_g, p_contained, sensitivity, energy,
                                   n_bins_energy, gammaness_bins, theta2_bins,
                                   save_figure=False):
    """
    Plot positions of surviving events after cuts

    Parameters
    --------
    df_gammas: `pandas.DataFrame` gammas dl2 parameters
    df_protons: `pandas.DataFrame` protons dl2 parameters
    gammaness_g: `numpy.ndarray`  gammaness array of gamma events
    gammaness_p: `numpy.ndarray`  gammaness array of proton events
    theta2_g: `numpy.ndarray`  theta2 array of gamma events
    p_contained: `numpy.ndarray`  containment of proton events inside
                  the ring established in camera coordinates
    sensitivity: `numpy.ndarray`  array with sensitivity values in energy bins
    energy: `numpy.ndarray`  energy edge bins (size n_bins_energy + 1)
    n_bins_energy: `int`  number of bins in energy
    gammaness_bins: `numpy.ndarray`  gammaness bins
    theta2_bins: `numpy.ndarray`  theta2 bins

    Returns
    --------

    """

    e_reco_g = df_gammas.reco_energy
    e_reco_p = df_protons.reco_energy
    for i in range(0, n_bins_energy):
        fig, ax = plt.subplots()
        print("Energy range [GeV]: ", energy[i], energy[i + 1])
        ind = np.unravel_index(np.nanargmin(sensitivity[i], axis=None), sensitivity[i].shape)
        events_g = df_gammas[(e_reco_g < energy[i + 1]) & (e_reco_g > energy[i]) \
                             & (gammaness_g > gammaness_bins[ind[0]]) & (theta2_g < theta2_bins[ind[1]])]

        events_p = df_protons[(e_reco_p < energy[i + 1]) & (e_reco_p > energy[i]) \
                              & (gammaness_p > gammaness_bins[ind[0]]) & p_contained]
        events_p.intensity.hist()
        ax.set_xlabel("Log(10) Intensity Protons")
        fig.savefig("intensity_prot%d" % i)

        df = pd.concat([events_g, events_p], ignore_index=True)
        plot_pos(df, True)

        if (save_figure):
            fig.savefig("srcpos_bin%d" % i)
